// WebGPU-accelerated Qwen3 inference engine
// Handles Q8_0 matmuls on GPU; k-quant and attention remain on CPU

// ============================================================
//  WGSL Shader Sources
// ============================================================

const WGSL_MATMUL_Q8_0 = /* wgsl */`
struct Params { inDim: u32, outDim: u32 }
@group(0) @binding(0) var<storage, read> weight: array<u32>;
@group(0) @binding(1) var<storage, read> in_vec: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_vec: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

fn rb(byteOff: u32) -> u32 {
    return (weight[byteOff >> 2u] >> ((byteOff & 3u) << 3u)) & 0xFFu;
}
fn rf16(byteOff: u32) -> f32 {
    // unpack2x16float: lower 16 bits -> .x, upper 16 bits -> .y
    return unpack2x16float(rb(byteOff) | (rb(byteOff + 1u) << 8u)).x;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= p.outDim) { return; }
    let bpc = (p.inDim + 31u) >> 5u;
    let colBase = c * bpc * 34u;
    var dot: f32 = 0.0;
    for (var b = 0u; b < bpc; b++) {
        let bb = colBase + b * 34u;
        let d = rf16(bb);
        let eb = b << 5u;
        let maxI = min(32u, p.inDim - eb);
        var bd: f32 = 0.0;
        for (var i = 0u; i < maxI; i++) {
            let qb = rb(bb + 2u + i);
            // sign-extend u8 -> i32: shift to MSB, arithmetic right shift back
            let q = f32(bitcast<i32>(qb << 24u) >> 24u);
            bd += in_vec[eb + i] * q;
        }
        dot += bd * d;
    }
    out_vec[c] = dot;
}`;

const WGSL_MATMUL_KQUANT = /* wgsl */`
struct Params { inDim: u32, outDim: u32, qtype: u32, blockBytes: u32 }
@group(0) @binding(0) var<storage, read> weight: array<u32>;
@group(0) @binding(1) var<storage, read> in_vec: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_vec: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

fn rb(o: u32) -> u32 { return (weight[o >> 2u] >> ((o & 3u) << 3u)) & 0xFFu; }
fn rf16(o: u32) -> f32 { return unpack2x16float(rb(o) | (rb(o + 1u) << 8u)).x; }
fn ri8(o: u32) -> f32 { return f32(bitcast<i32>(rb(o) << 24u) >> 24u); }

// Q4_K/Q5_K scale+min (j=0..7, 12-byte scales at scaleBase)
fn q4k_scale_min(scaleBase: u32, j: u32, sc: ptr<function,f32>, mn: ptr<function,f32>) {
    var s: u32; var m: u32;
    if (j < 4u) {
        s = rb(scaleBase + j) & 63u;
        m = rb(scaleBase + j + 4u) & 63u;
    } else {
        let jj = j - 4u;
        s = (rb(scaleBase + j + 4u) & 0xFu) | ((rb(scaleBase + jj) >> 6u) << 4u);
        m = (rb(scaleBase + j + 4u) >> 4u) | ((rb(scaleBase + j) >> 6u) << 4u);
    }
    *sc = f32(s); *mn = f32(m);
}

// Q3_K 6-bit signed scale (sub=0..15, 12-byte scales at scaleBase)
// s[0..7]: one lo4 per byte; s[8..11]: hi2 packed 4-per-byte
// byte(lo4)=(sub&7), shift(lo4)=(sub>>3)<<2; byte(hi2)=(sub&3)+8, shift(hi2)=(sub>>2)*2
fn q3k_scale(scaleBase: u32, sub: u32) -> f32 {
    // GGML: 6-bit unsigned (0..63), dequant applies (scale - 32) as bias subtraction
    let lo4 = (rb(scaleBase + (sub & 7u)) >> ((sub >> 3u) << 2u)) & 0xFu;
    let hi2 = (rb(scaleBase + 8u + (sub & 3u)) >> ((sub >> 2u) * 2u)) & 0x3u;
    return f32(lo4 | (hi2 << 4u)) - 32.0;  // maps 0..63 → -32..31
}

fn dequant(bb: u32, e: u32) -> f32 {
    var result: f32 = 0.0;
    switch p.qtype {
        case 10u: { // Q2_K: scales[16]@0 qs[64]@16 d(f16)@80 dmin(f16)@82
            let d = rf16(bb + 80u); let dm = rf16(bb + 82u);
            let sb = rb(bb + (e >> 4u));  // scales @ 0, index = e/16
            // qs: same interleaved layout as Q4_K — NOT a flat 2-bit array
            let qb = bb + 16u + (e >> 7u)*32u + ((e >> 4u) & 1u)*16u + (e & 15u);
            let q = (rb(qb) >> (((e >> 5u) & 3u) * 2u)) & 3u;
            result = d * f32(sb & 0xFu) * f32(q) - dm * f32((sb >> 4u) & 0xFu);
        }
        case 11u: { // Q3_K: hmask[32]@0 qs[64]@32 scales[12]@96 d(2)@108
            let d = rf16(bb + 108u);
            let scale = q3k_scale(bb + 96u, e >> 4u);
            // hmask: each byte covers 8 groups; byte = l_half*16+l, bit = e>>5
            let hb = (rb(bb + ((e >> 4u) & 1u)*16u + (e & 15u)) >> (e >> 5u)) & 1u;
            // qs: same interleaved layout as Q2_K
            let qb = bb + 32u + (e >> 7u)*32u + ((e >> 4u) & 1u)*16u + (e & 15u);
            let low2 = (rb(qb) >> (((e >> 5u) & 3u) * 2u)) & 3u;
            result = d * scale * f32(i32(low2 | (hb << 2u)) - 4);
        }
        case 12u: { // Q4_K: d(2) dmin(2) scales[12]@4 qs[128]@16
            // chunk=e>>6 (4 chunks of 64); lower half → even sub, upper half → odd sub
            let d = rf16(bb); let dm = rf16(bb + 2u);
            var sc: f32; var mn: f32;
            let sub = (e >> 6u) * 2u + ((e >> 5u) & 1u);
            q4k_scale_min(bb + 4u, sub, &sc, &mn);
            // each 64-elem chunk uses 32 qs bytes; e and e+32 share the same byte
            let qb = rb(bb + 16u + (e >> 6u) * 32u + (e & 31u));
            let q = select((qb >> 4u) & 0xFu, qb & 0xFu, ((e >> 5u) & 1u) == 0u);
            result = d * sc * f32(q) - dm * mn;
        }
        case 13u: { // Q5_K: d(2) dmin(2) scales[12]@4 qh[32]@16 qs[128]@48
            let d = rf16(bb); let dm = rf16(bb + 2u);
            var sc: f32; var mn: f32;
            let sub = (e >> 6u) * 2u + ((e >> 5u) & 1u);
            q4k_scale_min(bb + 4u, sub, &sc, &mn);
            // qh: 32 bytes, byte=e&31, bit=(e>>6)*2+((e>>5)&1)
            let qh_bit = (e >> 6u) * 2u + ((e >> 5u) & 1u);
            let hb = (rb(bb + 16u + (e & 31u)) >> qh_bit) & 1u;
            let qb = rb(bb + 48u + (e >> 6u) * 32u + (e & 31u));
            let low4 = select((qb >> 4u) & 0xFu, qb & 0xFu, ((e >> 5u) & 1u) == 0u);
            result = d * sc * f32(low4 | (hb << 4u)) - dm * mn;
        }
        case 14u: { // Q6_K: ql[128]@0 qh[64]@128 scales[16]@192 d(2)@208
            // two 128-elem chunks (e>>7), 4-interleaved within each chunk
            let d = rf16(bb + 208u);
            // ql: chunk*64 + (qgroup&1)*32 + l   where l=e&31, qgroup=(e>>5)&3
            let ql_idx = (e >> 7u) * 64u + ((e >> 5u) & 1u) * 32u + (e & 31u);
            let qlb = rb(bb + ql_idx);
            // nibble: low for qgroups 0,1 (e&127 < 64), high for 2,3
            let low4 = select((qlb >> 4u) & 0xFu, qlb & 0xFu, ((e >> 6u) & 1u) == 0u);
            // qh: chunk*32 + l, shift = qgroup*2
            let qh_idx = (e >> 7u) * 32u + (e & 31u);
            let qh_shift = ((e >> 5u) & 3u) * 2u;
            let high2 = (rb(bb + 128u + qh_idx) >> qh_shift) & 3u;
            // scale: chunk*8 + qgroup*2 + (l>>4)
            let sc_idx = (e >> 7u) * 8u + ((e >> 5u) & 3u) * 2u + ((e & 31u) >> 4u);
            let scaleByte = rb(bb + 192u + sc_idx);
            let sc = f32(bitcast<i32>(scaleByte << 24u) >> 24u);
            result = d * sc * f32(i32(low4 | (high2 << 4u)) - 32);
        }
        default: { result = 0.0; }
    }
    return result;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= p.outDim) { return; }
    let bpc = (p.inDim + 255u) / 256u;
    let colBase = c * bpc * p.blockBytes;
    var dot: f32 = 0.0;
    for (var b = 0u; b < bpc; b++) {
        let bb = colBase + b * p.blockBytes;
        let eb = b * 256u;
        let maxI = min(256u, p.inDim - eb);
        for (var i = 0u; i < maxI; i++) {
            dot += dequant(bb, i) * in_vec[eb + i];
        }
    }
    out_vec[c] = dot;
}`;

const WGSL_RMSNORM = /* wgsl */`
struct P { headDim: u32, eps: f32 }
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<uniform> p: P;

var<workgroup> wg: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let hd = p.headDim;
    let off = wid.x * hd;
    let li = lid.x;
    var s: f32 = 0.0;
    for (var i = li; i < hd; i += 256u) { let x = data[off + i]; s += x * x; }
    wg[li] = s;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (li < stride) { wg[li] += wg[li + stride]; }
        workgroupBarrier();
    }
    let inv = 1.0 / sqrt(wg[0] / f32(hd) + p.eps);
    for (var i = li; i < hd; i += 256u) {
        data[off + i] = weight[i] * data[off + i] * inv;
    }
}`;

const WGSL_ROPE = /* wgsl */`
struct P { nHeads: u32, headDim: u32, position: u32, freqBase: f32 }
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half = p.headDim / 2u;
    let h = gid.x / half;
    let i = gid.x % half;
    if (h >= p.nHeads) { return; }
    let freq = 1.0 / pow(p.freqBase, f32(i) / f32(half));
    let theta = f32(p.position) * freq;
    let c = cos(theta); let s = sin(theta);
    let off = h * p.headDim;
    let x0 = data[off + i];
    let x1 = data[off + i + half];
    data[off + i]        = x0 * c - x1 * s;
    data[off + i + half] = x0 * s + x1 * c;
}`;

const WGSL_ELEMENTWISE = /* wgsl */`
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<uniform> n: u32;

// SiLU(a) * b in-place on a
@compute @workgroup_size(256)
fn silu_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= n) { return; }
    let x = a[i];
    a[i] = (x / (1.0 + exp(-x))) * b[i];
}

// a += b
@compute @workgroup_size(256)
fn add_residual(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= n) { return; }
    a[i] += b[i];
}`;

// a += scale * b  (used for MoE expert accumulation — all on GPU, no CPU roundtrip)
const WGSL_AXPY = /* wgsl */`
struct P { n: u32, scale: f32 }
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    a[i] += p.scale * b[i];
}`;

// GQA decode attention — one workgroup per query head
// Workgroup memory: scores[2048] + reduce[256] ≈ 9 KB (within 16 KB limit)
const WGSL_ATTENTION = /* wgsl */`
struct AttnP {
    nHeads: u32, nHeadKV: u32,
    headDimQ: u32, headDimKV: u32,
    seqLen: u32, maxCtx: u32,
    scale: f32, _pad: u32,
}
@group(0) @binding(0) var<storage, read>       q:       array<f32>; // [nHeads * headDimQ]
@group(0) @binding(1) var<storage, read>       k_cache: array<f32>; // [maxCtx * nHeadKV * headDimKV]
@group(0) @binding(2) var<storage, read>       v_cache: array<f32>; // [maxCtx * nHeadKV * headDimKV]
@group(0) @binding(3) var<storage, read_write> out:     array<f32>; // [nHeads * headDimKV]
@group(0) @binding(4) var<uniform>             p:       AttnP;

var<workgroup> scores: array<f32, 2048>; // max seqLen supported
var<workgroup> reduce: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h  = wid.x;
    let li = lid.x;
    let kvHead  = (h * p.nHeadKV) / p.nHeads;
    let seqLen  = min(p.seqLen, 2048u);
    let qOff    = h * p.headDimQ;
    let kvStride = p.nHeadKV * p.headDimKV;

    // Step 1: QK^T scores
    for (var t = li; t < seqLen; t += 256u) {
        let kOff = t * kvStride + kvHead * p.headDimKV;
        var dot: f32 = 0.0;
        for (var d = 0u; d < p.headDimKV; d++) {
            dot += q[qOff + d] * k_cache[kOff + d];
        }
        scores[t] = dot * p.scale;
    }
    workgroupBarrier();

    // Step 2: find max
    var lmax: f32 = -1e30;
    for (var t = li; t < seqLen; t += 256u) { lmax = max(lmax, scores[t]); }
    reduce[li] = lmax;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (li < s) { reduce[li] = max(reduce[li], reduce[li + s]); }
        workgroupBarrier();
    }
    let gmax = reduce[0];

    // Step 3: exp + sum
    var lsum: f32 = 0.0;
    for (var t = li; t < seqLen; t += 256u) {
        let e = exp(scores[t] - gmax);
        scores[t] = e;
        lsum += e;
    }
    reduce[li] = lsum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (li < s) { reduce[li] += reduce[li + s]; }
        workgroupBarrier();
    }
    let inv_sum = 1.0 / reduce[0];

    // Step 4: normalize
    for (var t = li; t < seqLen; t += 256u) { scores[t] *= inv_sum; }
    workgroupBarrier();

    // Step 5: weighted V sum → output
    let outOff = h * p.headDimKV;
    for (var d = li; d < p.headDimKV; d += 256u) {
        var acc: f32 = 0.0;
        for (var t = 0u; t < seqLen; t++) {
            acc += scores[t] * v_cache[t * kvStride + kvHead * p.headDimKV + d];
        }
        out[outOff + d] = acc;
    }
}`;

const WGSL_COPY = /* wgsl */`
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> n: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= n) { return; }
    dst[i] = src[i];
}`;

// ============================================================
//  WebGPU Device wrapper
// ============================================================

class GPUDevice {
    constructor(device) {
        this.device = device;
        this._pipelineCache = new Map();
    }

    static async create() {
        if (!navigator.gpu) throw new Error('WebGPU not supported');
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) throw new Error('No WebGPU adapter found');
        const device = await adapter.requestDevice({
            requiredLimits: {
                maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
                maxBufferSize: adapter.limits.maxBufferSize,
            }
        });
        device.addEventListener('uncapturederror', e => console.error('[WebGPU]', e.error.message));
        return new GPUDevice(device);
    }

    // Upload ArrayBuffer/TypedArray data to a STORAGE + COPY_DST buffer
    uploadBuffer(data, extraUsage = 0) {
        const bytes = data instanceof ArrayBuffer ? data : data.buffer;
        const offset = data instanceof ArrayBuffer ? 0 : data.byteOffset;
        const size = data instanceof ArrayBuffer ? data.byteLength : data.byteLength;
        const aligned = Math.ceil(size / 4) * 4;
        const buf = this.device.createBuffer({
            size: aligned,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | extraUsage,
            mappedAtCreation: true,
        });
        new Uint8Array(buf.getMappedRange()).set(new Uint8Array(bytes, offset, size));
        buf.unmap();
        return buf;
    }

    // Create an empty GPU buffer (STORAGE | COPY_SRC | COPY_DST)
    createBuffer(size, extraUsage = 0) {
        const aligned = Math.ceil(size / 4) * 4;
        return this.device.createBuffer({
            size: aligned,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | extraUsage,
        });
    }

    createUniformBuffer(data) {
        const bytes = data instanceof ArrayBuffer ? new Uint8Array(data)
                    : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        const size = Math.ceil(bytes.length / 16) * 16;  // uniform buffers require 16-byte alignment
        const buf = this.device.createBuffer({
            size,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Uint8Array(buf.getMappedRange()).set(bytes);
        buf.unmap();
        return buf;
    }

    writeUniform(buf, data) {
        const bytes = data instanceof ArrayBuffer ? new Uint8Array(data)
                    : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        this.device.queue.writeBuffer(buf, 0, bytes);
    }

    // Read a GPUBuffer back to CPU as Float32Array
    async readF32(gpuBuf, nElems) {
        const byteSize = nElems * 4;
        const staging = this.device.createBuffer({
            size: byteSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(gpuBuf, 0, staging, 0, byteSize);
        this.device.queue.submit([enc.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(staging.getMappedRange().slice(0));
        staging.unmap();
        staging.destroy();
        return data;
    }

    async getOrCreatePipeline(label, wgsl, entryPoint = 'main') {
        const key = `${label}:${entryPoint}`;
        if (this._pipelineCache.has(key)) return this._pipelineCache.get(key);
        const module = this.device.createShaderModule({ label, code: wgsl });
        // Surface line-level compile errors before the GPU error event fires
        const info = await module.getCompilationInfo();
        let hasError = false;
        for (const msg of info.messages) {
            const prefix = `[WGSL:${label}] line ${msg.lineNum}:${msg.linePos}`;
            if (msg.type === 'error') { console.error(prefix, msg.message); hasError = true; }
            else if (msg.type === 'warning') { console.warn(prefix, msg.message); }
        }
        if (hasError) throw new Error(`Shader compilation failed: ${label}`);
        const pipeline = this.device.createComputePipeline({
            label,
            layout: 'auto',
            compute: { module, entryPoint },
        });
        this._pipelineCache.set(key, pipeline);
        return pipeline;
    }

    // Submit a compute dispatch (or accumulate into the current batch encoder)
    dispatch(pipeline, bindEntries, dispatchX, dispatchY = 1, dispatchZ = 1) {
        const bg = this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: bindEntries.map((e, i) => ({ binding: i, resource: e })),
        });
        const enc = this._batchEnc ?? this.device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
        pass.end();
        if (!this._batchEnc) this.device.queue.submit([enc.finish()]);
    }

    // Batch mode: collect all GPU commands into one encoder, submit once at endBatch().
    // This avoids the driver overhead of ~700 separate submit() calls per forward pass.
    beginBatch() {
        this._batchEnc      = this.device.createCommandEncoder();
        this._batchDeferred = [];   // uniform buffers to destroy after submit
    }
    endBatch() {
        if (!this._batchEnc) return;
        this.device.queue.submit([this._batchEnc.finish()]);
        this._batchEnc = null;
        for (const b of this._batchDeferred) b.destroy();
        this._batchDeferred = null;
    }
    // Destroy a buffer immediately, or defer it until after the current batch submits.
    deferDestroy(buf) {
        if (this._batchDeferred) this._batchDeferred.push(buf);
        else buf.destroy();
    }

    buf(b) { return { buffer: b }; }  // shorthand for binding resource
    ubuf(b) { return { buffer: b }; }
}

// ============================================================
//  Qwen3 GPU Engine
// ============================================================

class Qwen3GPUEngine {
    constructor() {}

    static async create(gguf) {
        const eng = new Qwen3GPUEngine();
        eng._gpu = await GPUDevice.create();
        eng._cpu = new Qwen3Engine(gguf);  // CPU engine for tokenizer, attention, etc.

        // Copy relevant fields from CPU engine
        eng.nLayers   = eng._cpu.nLayers;
        eng.nEmb      = eng._cpu.nEmb;
        eng.nHeads    = eng._cpu.nHeads;
        eng.nHeadKV   = eng._cpu.nHeadKV;
        eng.nQ        = eng._cpu.nQ;
        eng.nKV       = eng._cpu.nKV;
        eng.nFF       = eng._cpu.nFF;
        eng.nVocab    = eng._cpu.nVocab;
        eng.headDimQ  = eng._cpu.headDimQ;
        eng.headDimKV = eng._cpu.headDimKV;
        eng.eps       = eng._cpu.eps;
        eng.ropeFreqBase = eng._cpu.ropeFreqBase;
        eng.maxCtx    = eng._cpu.maxCtx;
        eng.kvCache   = eng._cpu.kvCache;
        eng.tokenizer = eng._cpu.tokenizer;
        eng.isMoE     = eng._cpu.isMoE;
        eng.layers    = eng._cpu.layers;  // needed for nExperts, nExpertsUsed per layer

        // Create pipelines (async — surfaces WGSL compile errors with line numbers)
        console.log('[GPU] Compiling shaders...');
        const [matmulQ80, matmulKQ, rmsnorm, rope, siluMul, addResidual, copy, attention, axpy] = await Promise.all([
            eng._gpu.getOrCreatePipeline('matmul_q8_0',  WGSL_MATMUL_Q8_0),
            eng._gpu.getOrCreatePipeline('matmul_kquant', WGSL_MATMUL_KQUANT),
            eng._gpu.getOrCreatePipeline('rmsnorm',       WGSL_RMSNORM),
            eng._gpu.getOrCreatePipeline('rope',          WGSL_ROPE),
            eng._gpu.getOrCreatePipeline('silu_mul',      WGSL_ELEMENTWISE, 'silu_mul'),
            eng._gpu.getOrCreatePipeline('add_residual',  WGSL_ELEMENTWISE, 'add_residual'),
            eng._gpu.getOrCreatePipeline('copy',          WGSL_COPY),
            eng._gpu.getOrCreatePipeline('attention',     WGSL_ATTENTION),
            eng._gpu.getOrCreatePipeline('axpy',          WGSL_AXPY),
        ]);
        eng._pipelines = { matmulQ80, matmulKQ, rmsnorm, rope, siluMul, addResidual, copy, attention, axpy };
        console.log('[GPU] Shaders compiled OK.');

        // Upload weight tensors to GPU
        console.log('[GPU] Uploading weights...');
        await eng._uploadWeights(gguf);
        console.log('[GPU] Ready.');

        // Allocate working GPU buffers
        eng._allocGPUBuffers();

        // Allocate GPU-side KV cache (replaces CPU KV cache for attention)
        eng._gpuMaxCtx = Math.min(eng._cpu.maxCtx, 2048);
        eng._allocGPUKVCache();

        return eng;
    }

    async _uploadWeights(gguf) {
        const g = this._gpu;
        const cpu = this._cpu;

        const upload = (meta) => {
            if (!meta) return null;
            // Upload just this tensor's data from the shared buffer
            const slice = new Uint8Array(meta.buffer, meta.offset, meta.nbytes);
            return g.uploadBuffer(slice);
        };

        this._gpuW = {
            tokEmbd:    upload(cpu.tokEmbd),
            outputNorm: upload(cpu.outputNorm),
            output:     upload(cpu.output),
            layers:     [],
        };

        for (let i = 0; i < this.nLayers; i++) {
            const l = cpu.layers[i];
            this._gpuW.layers.push({
                attnNorm: upload(l.attnNorm),
                wq: upload(l.wq),
                wk: upload(l.wk),
                wv: upload(l.wv),
                wo: upload(l.wo),
                qNorm: upload(l.qNorm),
                kNorm: upload(l.kNorm),
                ffnNorm: upload(l.ffnNorm),
                // Dense
                ffnGate: upload(l.ffnGate),
                ffnUp:   upload(l.ffnUp),
                ffnDown: upload(l.ffnDown),
                // MoE
                isMoE: l.isMoE,
                routerWeight:  l.routerWeight,  // keep CPU ref (F32, small)
                ffnGateExps: upload(l.ffnGateExps),
                ffnUpExps:   upload(l.ffnUpExps),
                ffnDownExps: upload(l.ffnDownExps),
                // Original CPU meta for type/shape info
                _cpuLayer: l,
            });
            if (i % 4 === 0) await yieldToBrowser();
        }
    }

    _allocGPUBuffers() {
        const g = this._gpu;
        const f32 = n => g.createBuffer(n * 4);
        this._buf = {
            hidden:    f32(this.nEmb),
            normed:    f32(this.nEmb),
            q:         f32(this.nQ),
            k:         f32(this.nKV),
            v:         f32(this.nKV),
            attnOut:   f32(this.nQ),
            proj:      f32(this.nEmb),
            gate:      f32(this.nFF),
            up:        f32(this.nFF),
            down:      f32(this.nEmb),
            moeAcc:    f32(this.nEmb),
            logits:    f32(this.nVocab),
        };
    }

    _allocGPUKVCache() {
        const g = this._gpu;
        // Per-layer GPU KV cache: [gpuMaxCtx * nHeadKV * headDimKV] f32 values = kvBytes bytes
        const floatCount = this._gpuMaxCtx * this.nHeadKV * this.headDimKV;
        const kvBytes    = floatCount * 4;
        this._kvCacheK = [];
        this._kvCacheV = [];
        for (let l = 0; l < this.nLayers; l++) {
            this._kvCacheK.push(g.createBuffer(kvBytes));  // createBuffer takes bytes
            this._kvCacheV.push(g.createBuffer(kvBytes));
        }
        const totalMB = (this.nLayers * 2 * kvBytes / 1024 / 1024).toFixed(0);
        console.log(`[GPU] KV cache: ${this.nLayers}L × 2 × ${(kvBytes/1024).toFixed(0)} KB = ${totalMB} MB`);
    }

    // ---- GPU helper: dispatch matmul (handles Q8_0 and k-quants) ----
    _matmul(weightBuf, inputBuf, outputBuf, meta, outDim) {
        const g = this._gpu;
        const inDim = meta.shape[0];
        const type = meta.type;

        if (type === 8) {
            // Q8_0
            const uBuf = g.createUniformBuffer(new Uint32Array([inDim, outDim]));
            g.dispatch(this._pipelines.matmulQ80,
                [g.buf(weightBuf), g.buf(inputBuf), g.buf(outputBuf), g.ubuf(uBuf)],
                Math.ceil(outDim / 256));
            uBuf.destroy();
        } else if (type >= 10 && type <= 14) {
            // k-quant
            const blockBytes = [0,0,0,0,0,0,0,0,0,0, 84,110,144,176,210][type];
            const uBuf = g.createUniformBuffer(new Uint32Array([inDim, outDim, type, blockBytes]));
            g.dispatch(this._pipelines.matmulKQ,
                [g.buf(weightBuf), g.buf(inputBuf), g.buf(outputBuf), g.ubuf(uBuf)],
                Math.ceil(outDim / 256));
            uBuf.destroy();
        } else {
            throw new Error(`GPU matmul: unsupported type ${type}`);
        }
    }

    // ---- GPU helper: dispatch rmsnorm (nHeads workgroups, each processes headDim elements) ----
    _rmsnorm(dataBuf, weightBuf, totalN, headDim = null) {
        const g = this._gpu;
        const hd = headDim ?? totalN;
        const nHeads = totalN / hd;
        // Pack Params: headDim(u32) + eps(f32) = 8 bytes
        const params = new ArrayBuffer(8);
        new Uint32Array(params)[0] = hd;
        new Float32Array(params)[1] = this.eps;
        const pBuf = g.createUniformBuffer(params);
        g.dispatch(this._pipelines.rmsnorm,
            [g.buf(dataBuf), g.buf(weightBuf), g.ubuf(pBuf)],
            nHeads);
        pBuf.destroy();
    }

    // ---- GPU helper: RoPE ----
    _rope(dataBuf, nHeads, headDim, position) {
        const g = this._gpu;
        const params = new ArrayBuffer(16);
        const u32 = new Uint32Array(params);
        const f32 = new Float32Array(params);
        u32[0] = nHeads; u32[1] = headDim; u32[2] = position; f32[3] = this.ropeFreqBase;
        const pBuf = g.createUniformBuffer(params);
        const nPairs = nHeads * (headDim / 2);
        g.dispatch(this._pipelines.rope,
            [g.buf(dataBuf), g.ubuf(pBuf)],
            Math.ceil(nPairs / 64));
        pBuf.destroy();
    }

    // ---- GPU helper: add residual (a += b) ----
    _addResidual(aBuf, bBuf, n) {
        const g = this._gpu;
        const uBuf = g.createUniformBuffer(new Uint32Array([n]));
        g.dispatch(this._pipelines.addResidual,
            [g.buf(aBuf), g.buf(bBuf), g.ubuf(uBuf)],
            Math.ceil(n / 256));
        uBuf.destroy();
    }

    // ---- GPU helper: a += scale * b  (MoE expert accumulation, fully on GPU) ----
    _axpy(aBuf, bBuf, n, scale) {
        const g = this._gpu;
        const params = new ArrayBuffer(8);
        new Uint32Array(params)[0] = n;
        new Float32Array(params)[1] = scale;
        const uBuf = g.createUniformBuffer(params);
        g.dispatch(this._pipelines.axpy,
            [g.buf(aBuf), g.buf(bBuf), g.ubuf(uBuf)],
            Math.ceil(n / 256));
        uBuf.destroy();
    }

    // ---- GPU helper: SiLU(a) * b in-place on a ----
    _siluMul(aBuf, bBuf, n) {
        const g = this._gpu;
        const uBuf = g.createUniformBuffer(new Uint32Array([n]));
        g.dispatch(this._pipelines.siluMul,
            [g.buf(aBuf), g.buf(bBuf), g.ubuf(uBuf)],
            Math.ceil(n / 256));
        uBuf.destroy();
    }

    // ---- GPU helper: copy buffer ----
    _copy(srcBuf, dstBuf, n) {
        const enc = this._gpu.device.createCommandEncoder();
        enc.copyBufferToBuffer(srcBuf, 0, dstBuf, 0, n * 4);
        this._gpu.device.queue.submit([enc.finish()]);
    }

    // ---- Store current K/V buffers into GPU KV cache at given position ----
    _storeKVGPU(layerIdx, position) {
        if (position >= this._gpuMaxCtx) return;  // silently ignore beyond GPU max ctx
        const byteOffset = position * this.nHeadKV * this.headDimKV * 4;  // bytes
        const byteSize   = this.nHeadKV * this.headDimKV * 4;              // bytes
        const enc = this._gpu.device.createCommandEncoder();
        enc.copyBufferToBuffer(this._buf.k, 0, this._kvCacheK[layerIdx], byteOffset, byteSize);
        enc.copyBufferToBuffer(this._buf.v, 0, this._kvCacheV[layerIdx], byteOffset, byteSize);
        this._gpu.device.queue.submit([enc.finish()]);
    }

    // ---- GPU GQA attention using GPU KV cache, output → _buf.attnOut ----
    _attentionGPU(layerIdx, position) {
        const g = this._gpu;
        const seqLen = position + 1;
        const scale  = 1.0 / Math.sqrt(this.headDimKV);

        const params = new ArrayBuffer(32);  // 8 × 4 bytes
        const u32 = new Uint32Array(params);
        const f32 = new Float32Array(params);
        u32[0] = this.nHeads;    u32[1] = this.nHeadKV;
        u32[2] = this.headDimQ;  u32[3] = this.headDimKV;
        u32[4] = seqLen;         u32[5] = this._gpuMaxCtx;
        f32[6] = scale;          u32[7] = 0;  // padding

        const pBuf = g.createUniformBuffer(params);
        g.dispatch(this._pipelines.attention, [
            g.buf(this._buf.q),
            g.buf(this._kvCacheK[layerIdx]),
            g.buf(this._kvCacheV[layerIdx]),
            g.buf(this._buf.attnOut),
            g.ubuf(pBuf),
        ], this.nHeads);  // one workgroup per Q head
        pBuf.destroy();
    }

    // ============================================================
    //  Forward pass (one token) — fully on GPU
    //  Only GPU→CPU sync: 1× logits readback (when !skipLogits)
    //  Only CPU→GPU upload: 1× embedding per token
    // ============================================================
    async forward(tokenId, position, skipLogits = false) {
        const g = this._gpu;
        const cpu = this._cpu;

        // 1. Embedding lookup on CPU → GPU hidden state
        const emb = embeddingLookupGeneric(cpu.tokEmbd, tokenId);
        g.device.queue.writeBuffer(this._buf.hidden, 0, emb);

        for (let l = 0; l < this.nLayers; l++) {
            const gl = this._gpuW.layers[l];
            const cl = cpu.layers[l];

            // --- Attention norm (pre-cached F32 buffer, no alloc/free) ---
            this._copy(this._buf.hidden, this._buf.normed, this.nEmb);
            this._rmsnorm(this._buf.normed, gl.attnNorm, this.nEmb);

            // --- Q / K / V projections ---
            this._matmul(gl.wq, this._buf.normed, this._buf.q, cl.wq, this.nQ);
            this._matmul(gl.wk, this._buf.normed, this._buf.k, cl.wk, this.nKV);
            this._matmul(gl.wv, this._buf.normed, this._buf.v, cl.wv, this.nKV);

            // --- Per-head RMSNorm on Q and K ---
            this._rmsnorm(this._buf.q, gl.qNorm, this.nQ,  this.headDimQ);
            this._rmsnorm(this._buf.k, gl.kNorm, this.nKV, this.headDimKV);

            // --- RoPE ---
            this._rope(this._buf.q, this.nHeads,  this.headDimQ,  position);
            this._rope(this._buf.k, this.nHeadKV, this.headDimKV, position);

            // --- Store K/V into GPU KV cache (GPU→GPU copy, no CPU roundtrip) ---
            this._storeKVGPU(l, position);

            // --- GPU attention (GQA) → _buf.attnOut ---
            this._attentionGPU(l, position);

            // --- Output projection ---
            this._matmul(gl.wo, this._buf.attnOut, this._buf.proj, cl.wo, this.nEmb);

            // --- Residual ---
            this._addResidual(this._buf.hidden, this._buf.proj, this.nEmb);

            // --- FFN norm ---
            this._copy(this._buf.hidden, this._buf.normed, this.nEmb);
            this._rmsnorm(this._buf.normed, gl.ffnNorm, this.nEmb);

            // --- FFN (dense or MoE) ---
            if (cl.isMoE) {
                await this._moEFFNGPU(l, gl, cl, position);
            } else {
                this._matmul(gl.ffnGate, this._buf.normed, this._buf.gate, cl.ffnGate, this.nFF);
                this._matmul(gl.ffnUp,   this._buf.normed, this._buf.up,   cl.ffnUp,   this.nFF);
                this._siluMul(this._buf.gate, this._buf.up, this.nFF);
                this._matmul(gl.ffnDown, this._buf.gate, this._buf.down, cl.ffnDown, this.nEmb);
                this._addResidual(this._buf.hidden, this._buf.down, this.nEmb);
            }

            if (l % 8 === 7) await yieldToBrowser();
        }

        if (!skipLogits) {
            // Final norm + lm_head — only GPU→CPU sync in the whole forward pass
            this._copy(this._buf.hidden, this._buf.normed, this.nEmb);
            this._rmsnorm(this._buf.normed, this._gpuW.outputNorm, this.nEmb);
            this._matmul(this._gpuW.output, this._buf.normed, this._buf.logits, cpu.output, this.nVocab);
            const logits = await g.readF32(this._buf.logits, this.nVocab);
            cpu.bufLogits.set(logits);
        }

        return skipLogits ? null : cpu.bufLogits;
    }

    // ---- MoE FFN on GPU (router on CPU, expert matmuls on GPU) ----
    async _moEFFNGPU(layerIdx, gl, cl, position) {
        const g = this._gpu;

        // Read normed hidden for router (CPU F32 matmul)
        const normedCPU = await g.readF32(this._buf.normed, this.nEmb);

        // Router: F32 matmul on CPU
        const routerLogits = matmulGeneric(cl.routerWeight, normedCPU);
        softmaxInPlace(routerLogits, 0, routerLogits.length);

        // Top-K selection
        const topK = cl.nExpertsUsed;
        const sorted = Array.from(routerLogits).map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
        const topIndices = sorted.slice(0, topK).map(x => x[1]);
        const topWeights = sorted.slice(0, topK).map(x => x[0]);
        let sumW = topWeights.reduce((a, b) => a + b, 0);
        const invSum = sumW > 0 ? 1 / sumW : 0;

        // Notify visualization
        if (this.onRouterUpdate) {
            this.onRouterUpdate({
                layer: layerIdx,
                probs: Array.from(routerLogits),
                selected: topIndices,
                weights: topWeights.map(w => w * invSum),
            });
        }

        // Clear MoE accumulator on GPU
        const enc0 = g.device.createCommandEncoder();
        enc0.clearBuffer(this._buf.moeAcc);
        g.device.queue.submit([enc0.finish()]);

        const hidDim = cl.ffnGateExps.shape[1];  // expert hidden dim (e.g. 768)
        const inDim  = cl.ffnGateExps.shape[0];  // input dim (e.g. 2048)

        for (let k = 0; k < topK; k++) {
            const expertIdx = topIndices[k];
            if (this.expertMask && this.expertMask.has(expertIdx)) continue;
            const w = topWeights[k] * invSum;

            // Get GPU buffer slices for this expert
            const gateSlice = this._expertGPUSlice(gl.ffnGateExps, cl.ffnGateExps, expertIdx, hidDim, inDim);
            const upSlice   = this._expertGPUSlice(gl.ffnUpExps,   cl.ffnUpExps,   expertIdx, hidDim, inDim);
            const downSlice = this._expertGPUSlice(gl.ffnDownExps, cl.ffnDownExps, expertIdx, this.nEmb, hidDim);

            // gate = matmul(normed, gate_exps_expert)
            this._matmulSlice(gateSlice.buf, gateSlice.offset, this._buf.normed, this._buf.gate, cl.ffnGateExps, hidDim, inDim);
            // up  = matmul(normed, up_exps_expert)
            this._matmulSlice(upSlice.buf, upSlice.offset, this._buf.normed, this._buf.up, cl.ffnUpExps, hidDim, inDim);
            // silu(gate) * up
            this._siluMul(this._buf.gate, this._buf.up, hidDim);
            // down = matmul(gate, down_exps_expert)
            this._matmulSlice(downSlice.buf, downSlice.offset, this._buf.gate, this._buf.down, cl.ffnDownExps, this.nEmb, hidDim);

            // Accumulate: moeAcc += w * down — fully on GPU, no CPU roundtrip
            this._axpy(this._buf.moeAcc, this._buf.down, this.nEmb, w);
        }

        // Add MoE output to hidden
        this._addResidual(this._buf.hidden, this._buf.moeAcc, this.nEmb);
    }

    _expertGPUSlice(gpuBuf, cpuMeta, expertIdx, outDim, inDim) {
        const blockBytes = this._cpu._blockSizeForType(cpuMeta.type);
        const elemsPerBlock = cpuMeta.type >= 10 ? 256 : 32;
        const blocksPerCol = Math.ceil(inDim / elemsPerBlock);
        const bytesPerExpert = outDim * blocksPerCol * blockBytes;
        return { buf: gpuBuf, offset: expertIdx * bytesPerExpert };
    }

    // Dispatch matmul using a slice of a larger GPU buffer (with byte offset)
    _matmulSlice(gpuBuf, byteOffset, inputBuf, outputBuf, cpuMeta, outDim, inDim) {
        // For now, fall back to CPU for sliced matmuls (GPU version needs offset support)
        // TODO: implement GPU-side offset support in the shader (pass byteOffset as uniform)
        const g = this._gpu;
        const type = cpuMeta.type;
        const blockBytes = [0,0,0,0,0,0,0,0,34,0, 84,110,144,176,210][type];
        const uBuf = (type === 8)
            ? g.createUniformBuffer(new Uint32Array([inDim, outDim]))
            : g.createUniformBuffer(new Uint32Array([inDim, outDim, type, blockBytes]));

        // We need a view buffer starting at byteOffset — create a sub-buffer view
        // WebGPU doesn't support buffer views directly, so we use the offset in bind group
        // Actually, we need to use a different approach: bind the buffer with offset
        const pipeline = type === 8 ? this._pipelines.matmulQ80 : this._pipelines.matmulKQ;
        const layout = pipeline.getBindGroupLayout(0);
        const bg = g.device.createBindGroup({
            layout,
            entries: [
                { binding: 0, resource: { buffer: gpuBuf, offset: byteOffset } },
                { binding: 1, resource: { buffer: inputBuf } },
                { binding: 2, resource: { buffer: outputBuf } },
                { binding: 3, resource: { buffer: uBuf } },
            ],
        });
        const enc = g.device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(outDim / 256));
        pass.end();
        g.device.queue.submit([enc.finish()]);
        uBuf.destroy();
    }

    // ---- CPU attention core (QK softmax + weighted V, no wo projection) ----
    // Returns cpu.bufAttnOut (Float32Array, size nQ = nHeads*headDimQ)
    _selfAttentionCPU(layerIdx, position, qData) {
        const cpu = this._cpu;
        cpu.bufQ.set(qData);
        return cpu._selfAttentionCore(layerIdx, position);  // returns bufAttnOut, no wo
    }

    // ============================================================
    //  Generation loop (same API as Qwen3Engine)
    // ============================================================
    async generate(tokenIds, options = {}) {
        const {
            maxSteps = 512,
            temperature = 0.7,
            topP = 0.9,
            topK = 40,
            onToken,
            onFinish,
            abortSignal,
        } = options;

        // GPU KV cache is overwritten from position 0 each generate() call;
        // no explicit clear needed since positions are always written before read.
        // CPU KV cache reset (kept for compatibility but not used by GPU attention)
        this.kvCache.reset();

        for (let i = 0; i < tokenIds.length - 1; i++) {
            if (abortSignal?.aborted) return;
            await this.forward(tokenIds[i], i, true);
            if (i % 8 === 0 && tokenIds.length > 16) await yieldToBrowser();
        }

        if (abortSignal?.aborted) return;
        await this.forward(tokenIds[tokenIds.length - 1], tokenIds.length - 1, false);

        const cpu = this._cpu;
        let currentToken = temperature <= 0
            ? sampleGreedy(cpu.bufLogits)
            : sampleTopPTopK(cpu.bufLogits, temperature, topP, topK);

        const generatedTokens = [];
        for (let step = 0; step < maxSteps; step++) {
            if (currentToken === cpu.tokenizer.eosTokenId) break;
            if (cpu.tokenizer.stopTokenIds.has(currentToken)) break;
            if (abortSignal?.aborted) return;

            await this.forward(currentToken, tokenIds.length + step);
            generatedTokens.push(currentToken);
            if (onToken) onToken(currentToken);

            currentToken = temperature <= 0
                ? sampleGreedy(cpu.bufLogits)
                : sampleTopPTopK(cpu.bufLogits, temperature, topP, topK);

            await yieldToBrowser();
        }
        if (onFinish) onFinish(generatedTokens);
    }

    formatChat(messages, systemPrompt) {
        return this._cpu.formatChat(messages, systemPrompt);
    }
}

// ============================================================
//  WGSL Shader: LFM2 depthwise causal short-conv
//  Each thread handles one dimension d independently.
//  Computes: bx=B*x, conv1d with rolling state, y=C*conv_out
//  Then shifts state left and appends bx — no cross-thread deps.
// ============================================================
const WGSL_SHORTCONV = /* wgsl */`
struct P { nEmb: u32, lCache: u32 }
@group(0) @binding(0) var<storage, read>       bcx:    array<f32>;   // [3*nEmb]: B|C|x from in_proj
@group(0) @binding(1) var<storage, read_write> state:  array<f32>;   // [(lCache-1)*nEmb]: rolling
@group(0) @binding(2) var<storage, read>       kernel: array<f32>;   // [nEmb*lCache]: [d*L+k]
@group(0) @binding(3) var<storage, read_write> conv_y: array<f32>;   // [nEmb]: output
@group(0) @binding(4) var<uniform>             p:      P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    if (d >= p.nEmb) { return; }
    let L     = p.lCache;    // = 3
    let dConv = L - 1u;     // = 2

    let B_d  = bcx[d];
    let C_d  = bcx[p.nEmb + d];
    let x_d  = bcx[2u * p.nEmb + d];
    let bx_d = B_d * x_d;

    // causal conv: sum past states then add current bx
    var conv_out: f32 = 0.0;
    for (var k = 0u; k < dConv; k++) {
        conv_out += kernel[d * L + k] * state[k * p.nEmb + d];
    }
    conv_out += kernel[d * L + dConv] * bx_d;

    // shift state: oldest drops off, newest = bx
    for (var k = 0u; k + 1u < dConv; k++) {
        state[k * p.nEmb + d] = state[(k + 1u) * p.nEmb + d];
    }
    if (dConv > 0u) {
        state[(dConv - 1u) * p.nEmb + d] = bx_d;
    }

    conv_y[d] = C_d * conv_out;
}`;

// ============================================================
//  WGSL Shader: F32 × F32 matrix-vector product
//  Used for LFM2 router weights (small, often F32 in GGUF)
// ============================================================
const WGSL_MATMUL_F32 = /* wgsl */`
struct P { inDim: u32, outDim: u32 }
@group(0) @binding(0) var<storage, read>       weight:  array<f32>;
@group(0) @binding(1) var<storage, read>        in_vec:  array<f32>;
@group(0) @binding(2) var<storage, read_write>  out_vec: array<f32>;
@group(0) @binding(3) var<uniform>              p:       P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= p.outDim) { return; }
    var dot: f32 = 0.0;
    let base = c * p.inDim;
    for (var i = 0u; i < p.inDim; i++) {
        dot += weight[base + i] * in_vec[i];
    }
    out_vec[c] = dot;
}`;

// ============================================================
//  WGSL Shader: MoE sigmoid routing + top-K (GPU-only)
//  Computes: sigmoid(logits + bias), selects top-K by biased score,
//  normalizes selected weights, stores result buffer:
//    [0..topK-1]              = selected expert indices (f32)
//    [topK..2*topK-1]         = normalized weights (f32)
//    [2*topK..2*topK+nE-1]    = all sigmoid probs (f32, for viz)
//  Supports up to 64 experts (workgroup size 64).
// ============================================================
const WGSL_MOE_SIGMOID_TOPK = /* wgsl */`
struct STP { nExperts: u32, topK: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<storage, read>       logits: array<f32>;
@group(0) @binding(1) var<storage, read>        bias:   array<f32>;
@group(0) @binding(2) var<storage, read_write>  result: array<f32>;
@group(0) @binding(3) var<uniform>              p:      STP;

var<workgroup> wg_probs: array<f32, 64>;
var<workgroup> wg_sel:   array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let e  = lid.x;
    let nE = p.nExperts;
    if (e < nE) {
        let prob  = 1.0 / (1.0 + exp(-logits[e]));
        wg_probs[e] = prob;
        wg_sel[e]   = prob + bias[e];
    } else {
        wg_probs[e] = 0.0;
        wg_sel[e]   = -1e38;
    }
    workgroupBarrier();
    if (e != 0u) { return; }
    // Thread 0: sequential top-K (nE=32, topK=4 — tiny)
    var sumW: f32 = 1e-6;
    for (var k = 0u; k < p.topK; k++) {
        var best: f32 = -1e38;
        var bestI: u32 = 0u;
        for (var i = 0u; i < nE; i++) {
            if (wg_sel[i] > best) { best = wg_sel[i]; bestI = i; }
        }
        result[k]           = f32(bestI);
        result[p.topK + k]  = wg_probs[bestI];
        sumW               += wg_probs[bestI];
        wg_sel[bestI]       = -1e38;
    }
    let inv = 1.0 / sumW;
    for (var k = 0u; k < p.topK; k++) { result[p.topK + k] *= inv; }
    for (var i = 0u; i < nE; i++) { result[2u * p.topK + i] = wg_probs[i]; }
}`;

// ============================================================
//  WGSL Shader: Q8_0 expert matmul — expert index from routing buf
//  Params: inDim, outDim, slotK (which top-K slot), bytesPerExpert
//  Reads expertIdx = u32(routingResult[slotK]) at runtime.
// ============================================================
const WGSL_MATMUL_EXPERT_Q80 = /* wgsl */`
struct P { inDim: u32, outDim: u32, slotK: u32, bytesPerExpert: u32 }
@group(0) @binding(0) var<storage, read>       weight:        array<u32>;
@group(0) @binding(1) var<storage, read>        in_vec:        array<f32>;
@group(0) @binding(2) var<storage, read_write>  out_vec:       array<f32>;
@group(0) @binding(3) var<uniform>              p:             P;
@group(0) @binding(4) var<storage, read>        routingResult: array<f32>;

fn rb(o: u32) -> u32 { return (weight[o >> 2u] >> ((o & 3u) << 3u)) & 0xFFu; }
fn rf16(o: u32) -> f32 { return unpack2x16float(rb(o) | (rb(o + 1u) << 8u)).x; }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= p.outDim) { return; }
    let expertIdx  = u32(routingResult[p.slotK]);
    let expertBase = expertIdx * p.bytesPerExpert;
    let bpc        = (p.inDim + 31u) >> 5u;
    let colBase    = expertBase + c * bpc * 34u;
    var dot: f32   = 0.0;
    for (var b = 0u; b < bpc; b++) {
        let bb   = colBase + b * 34u;
        let d    = rf16(bb);
        let eb   = b << 5u;
        let maxI = min(32u, p.inDim - eb);
        var bd: f32 = 0.0;
        for (var i = 0u; i < maxI; i++) {
            let qb = rb(bb + 2u + i);
            let q  = f32(bitcast<i32>(qb << 24u) >> 24u);
            bd += in_vec[eb + i] * q;
        }
        dot += bd * d;
    }
    out_vec[c] = dot;
}`;

// ============================================================
//  WGSL Shader: K-quant expert matmul — expert index from routing buf
//  Same dequant logic as WGSL_MATMUL_KQUANT, plus expertBase offset.
// ============================================================
const WGSL_MATMUL_EXPERT_KQUANT = /* wgsl */`
struct P { inDim: u32, outDim: u32, qtype: u32, blockBytes: u32,
           slotK: u32, bytesPerExpert: u32, _p2: u32, _p3: u32 }
@group(0) @binding(0) var<storage, read>       weight:        array<u32>;
@group(0) @binding(1) var<storage, read>        in_vec:        array<f32>;
@group(0) @binding(2) var<storage, read_write>  out_vec:       array<f32>;
@group(0) @binding(3) var<uniform>              p:             P;
@group(0) @binding(4) var<storage, read>        routingResult: array<f32>;

fn rb(o: u32) -> u32 { return (weight[o >> 2u] >> ((o & 3u) << 3u)) & 0xFFu; }
fn rf16(o: u32) -> f32 { return unpack2x16float(rb(o) | (rb(o + 1u) << 8u)).x; }
fn ri8(o: u32) -> f32 { return f32(bitcast<i32>(rb(o) << 24u) >> 24u); }

fn q4k_scale_min(scaleBase: u32, j: u32, sc: ptr<function,f32>, mn: ptr<function,f32>) {
    var s: u32; var m: u32;
    if (j < 4u) {
        s = rb(scaleBase + j) & 63u;
        m = rb(scaleBase + j + 4u) & 63u;
    } else {
        let jj = j - 4u;
        s = (rb(scaleBase + j + 4u) & 0xFu) | ((rb(scaleBase + jj) >> 6u) << 4u);
        m = (rb(scaleBase + j + 4u) >> 4u) | ((rb(scaleBase + j) >> 6u) << 4u);
    }
    *sc = f32(s); *mn = f32(m);
}
fn q3k_scale(scaleBase: u32, sub: u32) -> f32 {
    let lo4 = (rb(scaleBase + (sub & 7u)) >> ((sub >> 3u) << 2u)) & 0xFu;
    let hi2 = (rb(scaleBase + 8u + (sub & 3u)) >> ((sub >> 2u) * 2u)) & 0x3u;
    return f32(lo4 | (hi2 << 4u)) - 32.0;
}

fn dequant(bb: u32, e: u32) -> f32 {
    var result: f32 = 0.0;
    switch p.qtype {
        case 10u: {
            let d = rf16(bb + 80u); let dm = rf16(bb + 82u);
            let sb = rb(bb + (e >> 4u));
            let qb = bb + 16u + (e >> 7u)*32u + ((e >> 4u) & 1u)*16u + (e & 15u);
            let q = (rb(qb) >> (((e >> 5u) & 3u) * 2u)) & 3u;
            result = d * f32(sb & 0xFu) * f32(q) - dm * f32((sb >> 4u) & 0xFu);
        }
        case 11u: {
            let d = rf16(bb + 108u);
            let scale = q3k_scale(bb + 96u, e >> 4u);
            let hb = (rb(bb + ((e >> 4u) & 1u)*16u + (e & 15u)) >> (e >> 5u)) & 1u;
            let qb = bb + 32u + (e >> 7u)*32u + ((e >> 4u) & 1u)*16u + (e & 15u);
            let low2 = (rb(qb) >> (((e >> 5u) & 3u) * 2u)) & 3u;
            result = d * scale * f32(i32(low2 | (hb << 2u)) - 4);
        }
        case 12u: {
            let d = rf16(bb); let dm = rf16(bb + 2u);
            var sc: f32; var mn: f32;
            let sub = (e >> 6u) * 2u + ((e >> 5u) & 1u);
            q4k_scale_min(bb + 4u, sub, &sc, &mn);
            let qb = rb(bb + 16u + (e >> 6u) * 32u + (e & 31u));
            let q = select((qb >> 4u) & 0xFu, qb & 0xFu, ((e >> 5u) & 1u) == 0u);
            result = d * sc * f32(q) - dm * mn;
        }
        case 13u: {
            let d = rf16(bb); let dm = rf16(bb + 2u);
            var sc: f32; var mn: f32;
            let sub = (e >> 6u) * 2u + ((e >> 5u) & 1u);
            q4k_scale_min(bb + 4u, sub, &sc, &mn);
            let qh_bit = (e >> 6u) * 2u + ((e >> 5u) & 1u);
            let hb = (rb(bb + 16u + (e & 31u)) >> qh_bit) & 1u;
            let qb = rb(bb + 48u + (e >> 6u) * 32u + (e & 31u));
            let low4 = select((qb >> 4u) & 0xFu, qb & 0xFu, ((e >> 5u) & 1u) == 0u);
            result = d * sc * f32(low4 | (hb << 4u)) - dm * mn;
        }
        case 14u: {
            let d = rf16(bb + 208u);
            let ql_idx = (e >> 7u) * 64u + ((e >> 5u) & 1u) * 32u + (e & 31u);
            let qlb = rb(bb + ql_idx);
            let low4 = select((qlb >> 4u) & 0xFu, qlb & 0xFu, ((e >> 6u) & 1u) == 0u);
            let qh_idx = (e >> 7u) * 32u + (e & 31u);
            let qh_shift = ((e >> 5u) & 3u) * 2u;
            let high2 = (rb(bb + 128u + qh_idx) >> qh_shift) & 3u;
            let sc_idx = (e >> 7u) * 8u + ((e >> 5u) & 3u) * 2u + ((e & 31u) >> 4u);
            let scaleByte = rb(bb + 192u + sc_idx);
            let sc = f32(bitcast<i32>(scaleByte << 24u) >> 24u);
            result = d * sc * f32(i32(low4 | (high2 << 4u)) - 32);
        }
        default: { result = 0.0; }
    }
    return result;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= p.outDim) { return; }
    let expertIdx  = u32(routingResult[p.slotK]);
    let expertBase = expertIdx * p.bytesPerExpert;
    let bpc        = (p.inDim + 255u) / 256u;
    let colBase    = expertBase + c * bpc * p.blockBytes;
    var dot: f32   = 0.0;
    for (var b = 0u; b < bpc; b++) {
        let bb   = colBase + b * p.blockBytes;
        let eb   = b * 256u;
        let maxI = min(256u, p.inDim - eb);
        for (var i = 0u; i < maxI; i++) {
            dot += dequant(bb, i) * in_vec[eb + i];
        }
    }
    out_vec[c] = dot;
}`;

// ============================================================
//  WGSL Shader: Expert-weight AXPY — weight comes from routing buf
//  out[i] += routingResult[topK + slotK] * b[i]
// ============================================================
const WGSL_AXPY_EXPERT = /* wgsl */`
struct P { n: u32, slotK: u32, topK: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read_write>  a:             array<f32>;
@group(0) @binding(1) var<storage, read>         b:             array<f32>;
@group(0) @binding(2) var<uniform>               p:             P;
@group(0) @binding(3) var<storage, read>         routingResult: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let w  = routingResult[p.topK + p.slotK];
    a[i]  += w * b[i];
}`;

// ============================================================
//  Lfm2GPUEngine — WebGPU-accelerated LFM2/LFM2-MoE inference
//
//  Architecture vs Qwen3GPU:
//    - 18 recurrent (short-conv) + 6 attention layers
//    - Short-conv: GPU matmuls (in_proj/out_proj) + GPU conv shader
//    - Conv states stored in GPU buffers, cleared on reset
//    - Conv kernels pre-dequantized to F32, uploaded at load time
//    - KV cache: 6 layers only, indexed by attnCacheIdx
//    - MoE routing: fully on GPU (sigmoid+topK shader), batch readback after each token
// ============================================================
class Lfm2GPUEngine {
    constructor() {}

    static async create(gguf) {
        const eng = new Lfm2GPUEngine();
        eng._gpu = await GPUDevice.create();
        eng._cpu = new Lfm2Engine(gguf);
        const cpu = eng._cpu;

        // Mirror CPU engine fields
        eng.nLayers      = cpu.nLayers;
        eng.nEmb         = cpu.nEmb;
        eng.nHeads       = cpu.nHeads;
        eng.nHeadKV      = cpu.nHeadKV;
        eng.headDim      = cpu.headDim;
        eng.nKV          = cpu.nKV;
        eng.nFF          = cpu.nFF;
        eng.nFFExp       = cpu.nFFExp;
        eng.nExperts     = cpu.nExperts;
        eng.nExpertsUsed = cpu.nExpertsUsed;
        eng.nVocab       = cpu.nVocab;
        eng.eps          = cpu.eps;
        eng.ropeFreqBase = cpu.ropeFreqBase;
        eng.maxCtx       = cpu.maxCtx;
        eng.lCache       = cpu.lCache;
        eng.dConv        = cpu.dConv;
        eng.nAttnLayers  = cpu.nAttnLayers;
        eng.isMoE        = true;
        eng.layers       = cpu.layers;
        eng.tokenizer    = cpu.tokenizer;
        eng.onRouterUpdate  = null;
        eng.expertMask      = null;
        eng.batchRouterUpdate = true;   // routing data is batch-read after each token
        eng._gpuRoutingBufs     = [];   // per-MoE-layer routing result buffers
        eng._moeAbsLayerIndices = [];   // maps moeIdx → absolute layer index

        console.log('[GPU-LFM2] Compiling shaders...');
        const [matmulQ80, matmulKQ, matmulF32, rmsnorm, rope, siluMul, addResidual, copy, attention, axpy,
               shortconv, moeSigmoidTopK, matmulExpertQ80, matmulExpertKQ, axpyExpert] =
            await Promise.all([
                eng._gpu.getOrCreatePipeline('matmul_q8_0',          WGSL_MATMUL_Q8_0),
                eng._gpu.getOrCreatePipeline('matmul_kquant',         WGSL_MATMUL_KQUANT),
                eng._gpu.getOrCreatePipeline('matmul_f32',            WGSL_MATMUL_F32),
                eng._gpu.getOrCreatePipeline('rmsnorm',               WGSL_RMSNORM),
                eng._gpu.getOrCreatePipeline('rope',                  WGSL_ROPE),
                eng._gpu.getOrCreatePipeline('silu_mul',              WGSL_ELEMENTWISE, 'silu_mul'),
                eng._gpu.getOrCreatePipeline('add_residual',          WGSL_ELEMENTWISE, 'add_residual'),
                eng._gpu.getOrCreatePipeline('copy',                  WGSL_COPY),
                eng._gpu.getOrCreatePipeline('attention',             WGSL_ATTENTION),
                eng._gpu.getOrCreatePipeline('axpy',                  WGSL_AXPY),
                eng._gpu.getOrCreatePipeline('shortconv',             WGSL_SHORTCONV),
                eng._gpu.getOrCreatePipeline('moe_sigmoid_topk',      WGSL_MOE_SIGMOID_TOPK),
                eng._gpu.getOrCreatePipeline('matmul_expert_q80',     WGSL_MATMUL_EXPERT_Q80),
                eng._gpu.getOrCreatePipeline('matmul_expert_kquant',  WGSL_MATMUL_EXPERT_KQUANT),
                eng._gpu.getOrCreatePipeline('axpy_expert',           WGSL_AXPY_EXPERT),
            ]);
        eng._pipelines = { matmulQ80, matmulKQ, matmulF32, rmsnorm, rope, siluMul, addResidual, copy, attention, axpy,
                           shortconv, moeSigmoidTopK, matmulExpertQ80, matmulExpertKQ, axpyExpert };
        console.log('[GPU-LFM2] Shaders compiled OK.');

        console.log('[GPU-LFM2] Uploading weights...');
        await eng._uploadWeights();
        console.log('[GPU-LFM2] Ready.');

        eng._allocGPUBuffers();
        eng._gpuMaxCtx = Math.min(cpu.maxCtx, 2048);
        eng._allocGPUKVCache();

        return eng;
    }

    async _uploadWeights() {
        const g   = this._gpu;
        const cpu = this._cpu;

        const upload = (meta) => {
            if (!meta) return null;
            const slice = new Uint8Array(meta.buffer, meta.offset, meta.nbytes);
            return g.uploadBuffer(slice);
        };

        this._gpuW = {
            tokEmbd:    upload(cpu.tokEmbd),
            outputNorm: upload(cpu.outputNorm),
            output:     upload(cpu.output),
            layers:     [],
        };

        // Shared zero-bias buffer for MoE layers without expProbsBias
        const zeroBiasF32 = new Float32Array(cpu.nExperts);
        this._gpuZeroBias = g.uploadBuffer(zeroBiasF32);

        // Per-layer GPU conv states (only for recurrent layers)
        this._gpuConvStates = new Array(cpu.nLayers).fill(null);

        for (let i = 0; i < cpu.nLayers; i++) {
            const l = cpu.layers[i];
            const gl = {
                isRecurrent: l.isRecurrent,
                isMoE:       l.isMoE,
                isDense:     l.isDense,
                _cpuLayer:   l,
            };

            gl.attnNorm = upload(l.attnNorm);
            gl.ffnNorm  = upload(l.ffnNorm);

            if (l.isRecurrent) {
                gl.shortconvInProj  = upload(l.shortconvInProj);
                gl.shortconvOutProj = upload(l.shortconvOutProj);
                // Conv kernel pre-dequantized to F32 at construction time — upload directly
                gl.convKernel = g.uploadBuffer(l.convKernelF32);

                // Zero-initialised GPU conv state: [dConv * nEmb] f32
                const stateElems = cpu.dConv * cpu.nEmb;
                this._gpuConvStates[i] = g.createBuffer(stateElems * 4);
                const enc = g.device.createCommandEncoder();
                enc.clearBuffer(this._gpuConvStates[i]);
                g.device.queue.submit([enc.finish()]);
            } else {
                gl.wq    = upload(l.wq);
                gl.wk    = upload(l.wk);
                gl.wv    = upload(l.wv);
                gl.wo    = upload(l.wo);
                gl.qNorm = upload(l.qNorm);
                gl.kNorm = upload(l.kNorm);
            }

            if (l.isDense) {
                gl.ffnGate = upload(l.ffnGate);
                gl.ffnUp   = upload(l.ffnUp);
                gl.ffnDown = upload(l.ffnDown);
            } else {
                // ---- GPU MoE routing ----
                const moeIdx = this._gpuRoutingBufs.length;
                gl._moeIdx   = moeIdx;
                this._moeAbsLayerIndices.push(i);

                // Router weight (F32 or quantized) — uploaded as-is, shader handles type
                gl.routerWeightGPU = upload(l.routerWeight);

                // Expert selection bias (F32 vector, may be absent)
                if (l.expProbsBias) {
                    const biasF32 = _readWeightF32(l.expProbsBias, cpu.nExperts);
                    gl.expProbsBiasGPU = g.uploadBuffer(biasF32);
                } else {
                    gl.expProbsBiasGPU = this._gpuZeroBias;  // shared zeros
                }

                // Routing result buffer: [topK indices | topK weights | nExperts probs]
                const resultFloats = 2 * cpu.nExpertsUsed + cpu.nExperts;
                this._gpuRoutingBufs.push(g.createBuffer(resultFloats * 4));

                // Expert weight tensors
                gl.ffnGateExps = upload(l.ffnGateExps);
                gl.ffnUpExps   = upload(l.ffnUpExps);
                gl.ffnDownExps = upload(l.ffnDownExps);
            }

            this._gpuW.layers.push(gl);
            if (i % 4 === 0) await yieldToBrowser();
        }
    }

    _allocGPUBuffers() {
        const g = this._gpu;
        const f32 = n => g.createBuffer(n * 4);
        const cpu = this._cpu;
        const nQ  = cpu.nHeads  * cpu.headDim;
        const nKV = cpu.nHeadKV * cpu.headDim;
        this._buf = {
            hidden:       f32(cpu.nEmb),
            normed:       f32(cpu.nEmb),
            bcx:          f32(3 * cpu.nEmb),   // in_proj output for short-conv
            convY:        f32(cpu.nEmb),        // short-conv result, fed to out_proj
            q:            f32(nQ),
            k:            f32(nKV),
            v:            f32(nKV),
            attnOut:      f32(nQ),
            proj:         f32(cpu.nEmb),
            gate:         f32(Math.max(cpu.nFF, cpu.nFFExp)),
            up:           f32(Math.max(cpu.nFF, cpu.nFFExp)),
            down:         f32(cpu.nEmb),
            moeAcc:       f32(cpu.nEmb),
            routerLogits: f32(cpu.nExperts),   // router matmul output (nExperts)
            logits:       f32(cpu.nVocab),
        };
    }

    _allocGPUKVCache() {
        const g = this._gpu;
        const cpu = this._cpu;
        const floatCount = this._gpuMaxCtx * cpu.nHeadKV * cpu.headDim;
        const kvBytes    = floatCount * 4;
        this._kvCacheK = [];
        this._kvCacheV = [];
        for (let l = 0; l < cpu.nAttnLayers; l++) {
            this._kvCacheK.push(g.createBuffer(kvBytes));
            this._kvCacheV.push(g.createBuffer(kvBytes));
        }
        const totalMB = (cpu.nAttnLayers * 2 * kvBytes / 1024 / 1024).toFixed(0);
        console.log(`[GPU-LFM2] KV cache: ${cpu.nAttnLayers}L × 2 × ${(kvBytes / 1024).toFixed(0)} KB = ${totalMB} MB`);
    }

    // ---- GPU helper methods (same logic as Qwen3GPUEngine) ----

    _matmul(weightBuf, inputBuf, outputBuf, meta, outDim) {
        const g = this._gpu;
        const inDim = meta.shape[0];
        const type  = meta.type;
        if (type === 0) {
            // F32 weight (common for router weights)
            const uBuf = g.createUniformBuffer(new Uint32Array([inDim, outDim]));
            g.dispatch(this._pipelines.matmulF32,
                [g.buf(weightBuf), g.buf(inputBuf), g.buf(outputBuf), g.ubuf(uBuf)],
                Math.ceil(outDim / 256));
            g.deferDestroy(uBuf);
        } else if (type === 8) {
            const uBuf = g.createUniformBuffer(new Uint32Array([inDim, outDim]));
            g.dispatch(this._pipelines.matmulQ80,
                [g.buf(weightBuf), g.buf(inputBuf), g.buf(outputBuf), g.ubuf(uBuf)],
                Math.ceil(outDim / 256));
            g.deferDestroy(uBuf);
        } else if (type >= 10 && type <= 14) {
            const blockBytes = [0,0,0,0,0,0,0,0,0,0, 84,110,144,176,210][type];
            const uBuf = g.createUniformBuffer(new Uint32Array([inDim, outDim, type, blockBytes]));
            g.dispatch(this._pipelines.matmulKQ,
                [g.buf(weightBuf), g.buf(inputBuf), g.buf(outputBuf), g.ubuf(uBuf)],
                Math.ceil(outDim / 256));
            g.deferDestroy(uBuf);
        } else {
            throw new Error(`GPU-LFM2 matmul: unsupported type ${type}`);
        }
    }

    _rmsnorm(dataBuf, weightBuf, totalN, headDim = null) {
        const g = this._gpu;
        const hd = headDim ?? totalN;
        const nHeads = totalN / hd;
        const params = new ArrayBuffer(8);
        new Uint32Array(params)[0] = hd;
        new Float32Array(params)[1] = this.eps;
        const pBuf = g.createUniformBuffer(params);
        g.dispatch(this._pipelines.rmsnorm,
            [g.buf(dataBuf), g.buf(weightBuf), g.ubuf(pBuf)],
            nHeads);
        g.deferDestroy(pBuf);
    }

    _rope(dataBuf, nHeads, headDim, position) {
        const g = this._gpu;
        const params = new ArrayBuffer(16);
        const u32 = new Uint32Array(params);
        const f32 = new Float32Array(params);
        u32[0] = nHeads; u32[1] = headDim; u32[2] = position; f32[3] = this.ropeFreqBase;
        const pBuf = g.createUniformBuffer(params);
        g.dispatch(this._pipelines.rope,
            [g.buf(dataBuf), g.ubuf(pBuf)],
            Math.ceil(nHeads * (headDim / 2) / 64));
        g.deferDestroy(pBuf);
    }

    _addResidual(aBuf, bBuf, n) {
        const g = this._gpu;
        const uBuf = g.createUniformBuffer(new Uint32Array([n]));
        g.dispatch(this._pipelines.addResidual,
            [g.buf(aBuf), g.buf(bBuf), g.ubuf(uBuf)],
            Math.ceil(n / 256));
        g.deferDestroy(uBuf);
    }

    _axpy(aBuf, bBuf, n, scale) {
        const g = this._gpu;
        const params = new ArrayBuffer(8);
        new Uint32Array(params)[0] = n;
        new Float32Array(params)[1] = scale;
        const uBuf = g.createUniformBuffer(params);
        g.dispatch(this._pipelines.axpy,
            [g.buf(aBuf), g.buf(bBuf), g.ubuf(uBuf)],
            Math.ceil(n / 256));
        g.deferDestroy(uBuf);
    }

    _siluMul(aBuf, bBuf, n) {
        const g = this._gpu;
        const uBuf = g.createUniformBuffer(new Uint32Array([n]));
        g.dispatch(this._pipelines.siluMul,
            [g.buf(aBuf), g.buf(bBuf), g.ubuf(uBuf)],
            Math.ceil(n / 256));
        g.deferDestroy(uBuf);
    }

    // Use the batch encoder if available, otherwise create/submit its own
    _copy(srcBuf, dstBuf, n) {
        const g = this._gpu;
        const enc = g._batchEnc ?? g.device.createCommandEncoder();
        enc.copyBufferToBuffer(srcBuf, 0, dstBuf, 0, n * 4);
        if (!g._batchEnc) g.device.queue.submit([enc.finish()]);
    }

    // ---- Short-conv GPU dispatch ----
    _shortconvGPU(layerIdx, gl) {
        const g = this._gpu;
        const params = new Uint32Array([this.nEmb, this.lCache]);
        const pBuf = g.createUniformBuffer(params);
        g.dispatch(this._pipelines.shortconv, [
            g.buf(this._buf.bcx),
            g.buf(this._gpuConvStates[layerIdx]),
            g.buf(gl.convKernel),
            g.buf(this._buf.convY),
            g.ubuf(pBuf),
        ], Math.ceil(this.nEmb / 256));
        g.deferDestroy(pBuf);
    }

    // ---- Store K/V into GPU KV cache (kvIdx = attnCacheIdx, 0-5) ----
    _storeKVGPU(kvIdx, position) {
        if (position >= this._gpuMaxCtx) return;
        const g = this._gpu;
        const byteOffset = position * this.nHeadKV * this.headDim * 4;
        const byteSize   = this.nHeadKV * this.headDim * 4;
        const enc = g._batchEnc ?? g.device.createCommandEncoder();
        enc.copyBufferToBuffer(this._buf.k, 0, this._kvCacheK[kvIdx], byteOffset, byteSize);
        enc.copyBufferToBuffer(this._buf.v, 0, this._kvCacheV[kvIdx], byteOffset, byteSize);
        if (!g._batchEnc) g.device.queue.submit([enc.finish()]);
    }

    // ---- GPU GQA attention (uses attnCacheIdx-indexed KV cache) ----
    _attentionGPU(kvIdx, position) {
        const g = this._gpu;
        const seqLen = position + 1;
        const scale  = 1.0 / Math.sqrt(this.headDim);
        const params = new ArrayBuffer(32);
        const u32 = new Uint32Array(params);
        const f32 = new Float32Array(params);
        u32[0] = this.nHeads;  u32[1] = this.nHeadKV;
        u32[2] = this.headDim; u32[3] = this.headDim;
        u32[4] = seqLen;       u32[5] = this._gpuMaxCtx;
        f32[6] = scale;        u32[7] = 0;
        const pBuf = g.createUniformBuffer(params);
        g.dispatch(this._pipelines.attention, [
            g.buf(this._buf.q),
            g.buf(this._kvCacheK[kvIdx]),
            g.buf(this._kvCacheV[kvIdx]),
            g.buf(this._buf.attnOut),
            g.ubuf(pBuf),
        ], this.nHeads);
        g.deferDestroy(pBuf);
    }

    // ---- Dispatch GPU sigmoid+topK for one MoE layer ----
    _moeSigmoidTopK(moeIdx, gl) {
        const g   = this._gpu;
        const cpu = this._cpu;
        const params = new Uint32Array([cpu.nExperts, cpu.nExpertsUsed, 0, 0]);
        const pBuf = g.createUniformBuffer(params);
        g.dispatch(this._pipelines.moeSigmoidTopK, [
            g.buf(this._buf.routerLogits),
            g.buf(gl.expProbsBiasGPU),
            g.buf(this._gpuRoutingBufs[moeIdx]),
            g.ubuf(pBuf),
        ], 1);
        g.deferDestroy(pBuf);
    }

    // ---- Compute bytesPerExpert for a packed expert weight tensor ----
    _bytesPerExpert(cpuMeta, outDim, inDim) {
        const blockBytes    = this._cpu._blockSizeForType(cpuMeta.type);
        const elemsPerBlock = cpuMeta.type >= 10 ? 256 : 32;
        const blocksPerCol  = Math.ceil(inDim / elemsPerBlock);
        return outDim * blocksPerCol * blockBytes;
    }

    // ---- Expert matmul: reads expert index from routing buffer at runtime ----
    _matmulExpert(slotK, moeIdx, gpuWeightBuf, inputBuf, outputBuf, cpuMeta, outDim, inDim) {
        const g    = this._gpu;
        const type = cpuMeta.type;
        const bpe  = this._bytesPerExpert(cpuMeta, outDim, inDim);
        const routingBuf = this._gpuRoutingBufs[moeIdx];
        if (type === 8) {
            const pBuf = g.createUniformBuffer(new Uint32Array([inDim, outDim, slotK, bpe]));
            g.dispatch(this._pipelines.matmulExpertQ80, [
                g.buf(gpuWeightBuf), g.buf(inputBuf), g.buf(outputBuf),
                g.ubuf(pBuf), g.buf(routingBuf),
            ], Math.ceil(outDim / 256));
            g.deferDestroy(pBuf);
        } else if (type >= 10 && type <= 14) {
            const blockBytes = [0,0,0,0,0,0,0,0,0,0, 84,110,144,176,210][type];
            const pBuf = g.createUniformBuffer(
                new Uint32Array([inDim, outDim, type, blockBytes, slotK, bpe, 0, 0]));
            g.dispatch(this._pipelines.matmulExpertKQ, [
                g.buf(gpuWeightBuf), g.buf(inputBuf), g.buf(outputBuf),
                g.ubuf(pBuf), g.buf(routingBuf),
            ], Math.ceil(outDim / 256));
            g.deferDestroy(pBuf);
        } else {
            throw new Error(`GPU-LFM2 expert matmul: unsupported weight type ${type}`);
        }
    }

    // ---- Expert AXPY: weight = routingResult[topK + slotK] ----
    _axpyExpert(slotK, topK, moeIdx, accBuf, addBuf, n) {
        const g = this._gpu;
        const pBuf = g.createUniformBuffer(new Uint32Array([n, slotK, topK, 0]));
        g.dispatch(this._pipelines.axpyExpert, [
            g.buf(accBuf), g.buf(addBuf),
            g.ubuf(pBuf), g.buf(this._gpuRoutingBufs[moeIdx]),
        ], Math.ceil(n / 256));
        g.deferDestroy(pBuf);
    }

    // ---- MoE FFN: fully on GPU, zero CPU roundtrips during forward pass ----
    //  1. Router matmul on GPU → routerLogits
    //  2. Sigmoid + top-K on GPU → routing buffer (read back later in batch)
    //  3. Expert matmuls use indices from routing buffer — no CPU involvement
    _moEFFNGPU(layerIdx, gl, cl) {
        const g   = this._gpu;
        const cpu = this._cpu;
        const moeIdx = gl._moeIdx;

        // 1. Router matmul: normed → routerLogits
        this._matmul(gl.routerWeightGPU, this._buf.normed, this._buf.routerLogits,
                     cl.routerWeight, cpu.nExperts);

        // 2. GPU sigmoid + top-K → routing buffer for this layer
        this._moeSigmoidTopK(moeIdx, gl);

        // 3. Clear MoE accumulator (use batch encoder if active)
        const clearEnc = g._batchEnc ?? g.device.createCommandEncoder();
        clearEnc.clearBuffer(this._buf.moeAcc);
        if (!g._batchEnc) g.device.queue.submit([clearEnc.finish()]);

        const topK   = cpu.nExpertsUsed;
        const hidDim = cl.ffnGateExps.shape[1];
        const inDim  = cl.ffnGateExps.shape[0];

        // 4. For each top-K slot, dispatch expert matmuls (indices resolved GPU-side)
        for (let k = 0; k < topK; k++) {
            this._matmulExpert(k, moeIdx, gl.ffnGateExps, this._buf.normed, this._buf.gate,
                               cl.ffnGateExps, hidDim, inDim);
            this._matmulExpert(k, moeIdx, gl.ffnUpExps,   this._buf.normed, this._buf.up,
                               cl.ffnUpExps,   hidDim, inDim);
            this._siluMul(this._buf.gate, this._buf.up, hidDim);
            this._matmulExpert(k, moeIdx, gl.ffnDownExps, this._buf.gate, this._buf.down,
                               cl.ffnDownExps, this.nEmb, hidDim);
            this._axpyExpert(k, topK, moeIdx, this._buf.moeAcc, this._buf.down, this.nEmb);
        }

        this._addResidual(this._buf.hidden, this._buf.moeAcc, this.nEmb);
        // No await — no GPU→CPU sync during forward pass!
    }

    // ---- Batch-read all MoE routing results after each token ----
    //  Called once per generated token (not per layer).
    //  Uses Promise.all for parallel GPU→CPU readback across all MoE layers.
    async _readbackRoutingData() {
        if (!this.onRouterUpdate) return;
        const cpu       = this._cpu;
        const topK      = cpu.nExpertsUsed;
        const nExperts  = cpu.nExperts;
        const resultFloats = 2 * topK + nExperts;

        const results = await Promise.all(
            this._gpuRoutingBufs.map(buf => this._gpu.readF32(buf, resultFloats))
        );

        for (let m = 0; m < results.length; m++) {
            const res      = results[m];
            const selected = Array.from({ length: topK },  (_, k) => Math.round(res[k]));
            const weights  = Array.from({ length: topK },  (_, k) => res[topK + k]);
            const probs    = Array.from(res.subarray(2 * topK, 2 * topK + nExperts));
            this.onRouterUpdate({
                layer:    this._moeAbsLayerIndices[m],
                probs,
                selected,
                weights,
            });
        }
    }

    // ================================================================
    //  Forward pass — one token, fully on GPU except:
    //    - embedding lookup (CPU → GPU upload, ~8KB)
    //    - logits (1× GPU→CPU readback when !skipLogits, nVocab×4B)
    //  MoE routing is now GPU-only (no CPU roundtrip per layer).
    //  Routing results are batch-read in _readbackRoutingData() after generate().
    // ================================================================
    async forward(tokenId, position, skipLogits = false) {
        const g   = this._gpu;
        const cpu = this._cpu;

        // Embedding lookup on CPU, then upload to GPU hidden buffer
        const emb = embeddingLookupGeneric(cpu.tokEmbd, tokenId);
        g.device.queue.writeBuffer(this._buf.hidden, 0, emb);

        const nQ  = cpu.nHeads  * cpu.headDim;
        const nKV = cpu.nHeadKV * cpu.headDim;

        // Begin batching: all GPU commands for this forward pass go into one encoder.
        // This eliminates ~700 separate submit() calls and their driver overhead.
        g.beginBatch();

        for (let l = 0; l < this.nLayers; l++) {
            const gl = this._gpuW.layers[l];
            const cl = cpu.layers[l];

            // Pre-norm (operator block)
            this._copy(this._buf.hidden, this._buf.normed, this.nEmb);
            this._rmsnorm(this._buf.normed, gl.attnNorm, this.nEmb);

            if (cl.isRecurrent) {
                // --- Short-conv recurrent block ---
                // in_proj: normed → bcx [3*nEmb]
                this._matmul(gl.shortconvInProj, this._buf.normed, this._buf.bcx, cl.shortconvInProj, 3 * this.nEmb);
                // conv shader: bcx + state + kernel → conv_y; updates state in-place
                this._shortconvGPU(l, gl);
                // out_proj: conv_y → proj [nEmb]
                this._matmul(gl.shortconvOutProj, this._buf.convY, this._buf.proj, cl.shortconvOutProj, this.nEmb);
            } else {
                // --- Full attention block ---
                this._matmul(gl.wq, this._buf.normed, this._buf.q, cl.wq, nQ);
                this._matmul(gl.wk, this._buf.normed, this._buf.k, cl.wk, nKV);
                this._matmul(gl.wv, this._buf.normed, this._buf.v, cl.wv, nKV);
                this._rmsnorm(this._buf.q, gl.qNorm, nQ,  cpu.headDim);
                this._rmsnorm(this._buf.k, gl.kNorm, nKV, cpu.headDim);
                this._rope(this._buf.q, cpu.nHeads,  cpu.headDim, position);
                this._rope(this._buf.k, cpu.nHeadKV, cpu.headDim, position);
                const kvIdx = cl.attnCacheIdx;    // 0..5 (attention layers only)
                this._storeKVGPU(kvIdx, position);
                this._attentionGPU(kvIdx, position);
                this._matmul(gl.wo, this._buf.attnOut, this._buf.proj, cl.wo, this.nEmb);
            }

            // Residual
            this._addResidual(this._buf.hidden, this._buf.proj, this.nEmb);

            // FFN norm
            this._copy(this._buf.hidden, this._buf.normed, this.nEmb);
            this._rmsnorm(this._buf.normed, gl.ffnNorm, this.nEmb);

            // FFN (dense for layers 0-1, MoE for the rest)
            if (cl.isDense) {
                this._matmul(gl.ffnGate, this._buf.normed, this._buf.gate, cl.ffnGate, this.nFF);
                this._matmul(gl.ffnUp,   this._buf.normed, this._buf.up,   cl.ffnUp,   this.nFF);
                this._siluMul(this._buf.gate, this._buf.up, this.nFF);
                this._matmul(gl.ffnDown, this._buf.gate, this._buf.down, cl.ffnDown, this.nEmb);
                this._addResidual(this._buf.hidden, this._buf.down, this.nEmb);
            } else {
                this._moEFFNGPU(l, gl, cl);  // synchronous — no GPU→CPU stall
            }
            // No per-layer yieldToBrowser() — GPU work is async; yield only between tokens
        }

        if (!skipLogits) {
            this._copy(this._buf.hidden, this._buf.normed, this.nEmb);
            this._rmsnorm(this._buf.normed, this._gpuW.outputNorm, this.nEmb);
            this._matmul(this._gpuW.output, this._buf.normed, this._buf.logits, cpu.output, this.nVocab);
        }

        // Submit all accumulated GPU commands in one shot, then read back logits
        g.endBatch();

        if (!skipLogits) {
            const logits = await g.readF32(this._buf.logits, this.nVocab);
            cpu.bufLogits.set(logits);
        }
        return skipLogits ? null : cpu.bufLogits;
    }

    // Clear GPU conv states + reset KV position counter
    _resetGPUState() {
        this._cpu.kvCache.reset();
        const enc = this._gpu.device.createCommandEncoder();
        for (let i = 0; i < this.nLayers; i++) {
            if (this._gpuConvStates[i]) enc.clearBuffer(this._gpuConvStates[i]);
        }
        this._gpu.device.queue.submit([enc.finish()]);
    }

    // ---- Generation loop ----
    async generate(tokenIds, options = {}) {
        const {
            maxSteps     = 512,
            temperature  = 0.7,
            topP         = 0.9,
            topK         = 40,
            thinkingMode = 'suppress',
            onToken,
            onPrefill,
            onFinish,
            abortSignal,
        } = options;

        // LFM2: <think>=124901, </think>=124902
        const thinkOpts = {
            mode:         thinkingMode,
            thinkId:      124901,
            endThinkId:   124902,
            thinkingDone: thinkingMode === 'suppress',
        };

        this._resetGPUState();

        // Prefill: submit one batch per token (GPU works asynchronously).
        // Sync the GPU every SYNC_EVERY tokens so onPrefill reflects actual GPU progress,
        // not just CPU submission speed.  Without sync, all progress would show instantly
        // and then freeze until the final readF32 forces a wait.
        const SYNC_EVERY = 8;
        for (let i = 0; i < tokenIds.length - 1; i++) {
            if (abortSignal?.aborted) return;
            await this.forward(tokenIds[i], i, true);
            const isLast     = (i === tokenIds.length - 2);
            const isSyncPt   = ((i + 1) % SYNC_EVERY === 0) || isLast;
            if (isSyncPt) {
                await this._gpu.device.queue.onSubmittedWorkDone();
                if (onPrefill) onPrefill(i + 1, tokenIds.length);
                await yieldToBrowser();
            }
        }

        if (abortSignal?.aborted) return;
        await this.forward(tokenIds[tokenIds.length - 1], tokenIds.length - 1, false);

        const cpu = this._cpu;

        let currentToken = sampleWithThinkControl(cpu.bufLogits, temperature, topP, topK, thinkOpts);

        const generatedTokens = [];
        for (let step = 0; step < maxSteps; step++) {
            if (currentToken === cpu.tokenizer.eosTokenId) break;
            if (cpu.tokenizer.stopTokenIds?.has(currentToken)) break;
            if (abortSignal?.aborted) return;

            if (currentToken === thinkOpts.endThinkId) thinkOpts.thinkingDone = true;

            await this.forward(currentToken, tokenIds.length + step);

            // Batch-read all MoE routing data after forward pass (one Promise.all,
            // not per-layer awaits).  Must happen before onToken so router-viz sees
            // the data for this token when it calls commitToken().
            if (this.onRouterUpdate) await this._readbackRoutingData();

            generatedTokens.push(currentToken);
            if (onToken) onToken(currentToken);

            currentToken = sampleWithThinkControl(cpu.bufLogits, temperature, topP, topK, thinkOpts);

            await yieldToBrowser();
        }
        if (onFinish) onFinish(generatedTokens);
    }

    resetKVCache() { this._resetGPUState(); }

    formatChat(messages, systemPrompt, thinkingMode = 'suppress') {
        return this._cpu.formatChat(messages, systemPrompt, thinkingMode);
    }
}
