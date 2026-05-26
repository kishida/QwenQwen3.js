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

// Q3_K 6-bit scale (j=0..15, 12-byte scales at scaleBase)
fn q3k_scale(scaleBase: u32, j: u32) -> f32 {
    let lo4 = (rb(scaleBase + (j >> 1u)) >> ((j & 1u) << 2u)) & 0xFu;
    let hi2 = (rb(scaleBase + 8u + (j >> 2u)) >> ((j & 3u) << 1u)) & 0x3u;
    return f32(lo4 | (hi2 << 4u));
}

fn dequant(bb: u32, e: u32) -> f32 {
    var result: f32 = 0.0;
    switch p.qtype {
        case 10u: { // Q2_K: d(2) dmin(2) scales[16]@4 qs[64]@20
            let d = rf16(bb); let dm = rf16(bb + 2u);
            let sub = e >> 4u;
            let sb = rb(bb + 4u + sub);
            let q = (rb(bb + 20u + (e >> 2u)) >> ((e & 3u) << 1u)) & 3u;
            result = d * f32(sb & 0xFu) * f32(q) - dm * f32((sb >> 4u) & 0xFu);
        }
        case 11u: { // Q3_K: hmask[32]@0 qs[64]@32 scales[12]@96 d(2)@108
            let d = rf16(bb + 108u);
            let scale = q3k_scale(bb + 96u, e >> 4u);
            let hb = (rb(bb + (e >> 3u)) >> (e & 7u)) & 1u;
            let low2 = (rb(bb + 32u + (e >> 2u)) >> ((e & 3u) << 1u)) & 3u;
            result = d * scale * f32(i32(low2 | (hb << 2u)) - 4);
        }
        case 12u: { // Q4_K: d(2) dmin(2) scales[12]@4 qs[128]@16
            let d = rf16(bb); let dm = rf16(bb + 2u);
            var sc: f32; var mn: f32;
            q4k_scale_min(bb + 4u, e >> 5u, &sc, &mn);
            let qb = rb(bb + 16u + (e >> 1u));
            let q = select((qb >> 4u) & 0xFu, qb & 0xFu, (e & 1u) == 0u);
            result = d * sc * f32(q) - dm * mn;
        }
        case 13u: { // Q5_K: d(2) dmin(2) scales[12]@4 qh[32]@16 qs[128]@48
            let d = rf16(bb); let dm = rf16(bb + 2u);
            var sc: f32; var mn: f32;
            q4k_scale_min(bb + 4u, e >> 5u, &sc, &mn);
            let hb = (rb(bb + 16u + (e >> 3u)) >> (e & 7u)) & 1u;
            let qb = rb(bb + 48u + (e >> 1u));
            let low4 = select((qb >> 4u) & 0xFu, qb & 0xFu, (e & 1u) == 0u);
            result = d * sc * f32(low4 | (hb << 4u)) - dm * mn;
        }
        case 14u: { // Q6_K: ql[128]@0 qh[64]@128 scales[16]@192 d(2)@208
            let d = rf16(bb + 208u);
            let scaleByte = rb(bb + 192u + (e >> 4u));
            let sc = f32(bitcast<i32>(scaleByte << 24u) >> 24u);
            let qlb = rb(bb + (e >> 1u));
            let low4 = select((qlb >> 4u) & 0xFu, qlb & 0xFu, (e & 1u) == 0u);
            let high2 = (rb(bb + 128u + (e >> 2u)) >> ((e & 3u) << 1u)) & 3u;
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
}

// a += scale * b  (axpy)
@compute @workgroup_size(256)
fn axpy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= n) { return; }
    a[i] += b[i];  // scale passed via b being pre-scaled on CPU
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

    // Submit a compute dispatch
    dispatch(pipeline, bindEntries, dispatchX, dispatchY = 1, dispatchZ = 1) {
        const bg = this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: bindEntries.map((e, i) => ({ binding: i, resource: e })),
        });
        const enc = this.device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
        pass.end();
        this.device.queue.submit([enc.finish()]);
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

        // Create pipelines (async — surfaces WGSL compile errors with line numbers)
        console.log('[GPU] Compiling shaders...');
        const [matmulQ80, matmulKQ, rmsnorm, rope, siluMul, addResidual, copy] = await Promise.all([
            eng._gpu.getOrCreatePipeline('matmul_q8_0',  WGSL_MATMUL_Q8_0),
            eng._gpu.getOrCreatePipeline('matmul_kquant', WGSL_MATMUL_KQUANT),
            eng._gpu.getOrCreatePipeline('rmsnorm',       WGSL_RMSNORM),
            eng._gpu.getOrCreatePipeline('rope',          WGSL_ROPE),
            eng._gpu.getOrCreatePipeline('silu_mul',      WGSL_ELEMENTWISE, 'silu_mul'),
            eng._gpu.getOrCreatePipeline('add_residual',  WGSL_ELEMENTWISE, 'add_residual'),
            eng._gpu.getOrCreatePipeline('copy',          WGSL_COPY),
        ]);
        eng._pipelines = { matmulQ80, matmulKQ, rmsnorm, rope, siluMul, addResidual, copy };
        console.log('[GPU] Shaders compiled OK.');

        // Upload weight tensors to GPU
        console.log('[GPU] Uploading weights...');
        await eng._uploadWeights(gguf);
        console.log('[GPU] Ready.');

        // Allocate working GPU buffers
        eng._allocGPUBuffers();

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

    // ---- GPU helper: norm weight buffer (F32, small → re-upload each call or cache) ----
    _getF32WeightBuf(meta) {
        // These are small (n_embd = 2048 floats = 8KB), upload on demand
        const slice = new Uint8Array(meta.buffer, meta.offset, meta.nbytes);
        return this._gpu.uploadBuffer(slice);
    }

    // ============================================================
    //  Forward pass (one token)
    // ============================================================
    async forward(tokenId, position, skipLogits = false) {
        const g = this._gpu;
        const cpu = this._cpu;

        // 1. Embedding lookup on CPU → upload to hiddenBuf
        const emb = embeddingLookupGeneric(cpu.tokEmbd, tokenId);
        g.device.queue.writeBuffer(this._buf.hidden, 0, emb);

        for (let l = 0; l < this.nLayers; l++) {
            const gl = this._gpuW.layers[l];
            const cl = cpu.layers[l];

            // --- Attention norm ---
            this._copy(this._buf.hidden, this._buf.normed, this.nEmb);
            const attnNormF32Buf = this._getF32WeightBuf(cl.attnNorm);
            this._rmsnorm(this._buf.normed, attnNormF32Buf, this.nEmb);
            attnNormF32Buf.destroy();

            // --- Q / K / V projections ---
            this._matmul(gl.wq, this._buf.normed, this._buf.q, cl.wq, this.nQ);
            this._matmul(gl.wk, this._buf.normed, this._buf.k, cl.wk, this.nKV);
            this._matmul(gl.wv, this._buf.normed, this._buf.v, cl.wv, this.nKV);

            // --- Per-head RMSNorm on Q and K ---
            const qNormF32Buf = this._getF32WeightBuf(cl.qNorm);
            const kNormF32Buf = this._getF32WeightBuf(cl.kNorm);
            this._rmsnorm(this._buf.q, qNormF32Buf, this.nQ, this.headDimQ);
            this._rmsnorm(this._buf.k, kNormF32Buf, this.nKV, this.headDimKV);
            qNormF32Buf.destroy();
            kNormF32Buf.destroy();

            // --- RoPE ---
            this._rope(this._buf.q, this.nHeads, this.headDimQ, position);
            this._rope(this._buf.k, this.nHeadKV, this.headDimKV, position);

            // --- Read back Q, K, V for CPU attention ---
            const [qCPU, kCPU, vCPU] = await Promise.all([
                g.readF32(this._buf.q, this.nQ),
                g.readF32(this._buf.k, this.nKV),
                g.readF32(this._buf.v, this.nKV),
            ]);

            // --- KV cache + attention core on CPU (QK softmax + weighted V, no wo projection) ---
            this.kvCache.store(l, position, kCPU, vCPU);
            const attnCoreOut = this._selfAttentionCPU(l, position, qCPU);

            // --- Upload attention output → GPU wo projection ---
            // attnCoreOut is bufAttnOut (size nQ = nHeads*headDimQ)
            g.device.queue.writeBuffer(this._buf.attnOut, 0, attnCoreOut);
            this._matmul(gl.wo, this._buf.attnOut, this._buf.proj, cl.wo, this.nEmb);

            // --- Residual: hidden += proj ---
            this._addResidual(this._buf.hidden, this._buf.proj, this.nEmb);

            // --- FFN norm ---
            this._copy(this._buf.hidden, this._buf.normed, this.nEmb);
            const ffnNormF32Buf = this._getF32WeightBuf(cl.ffnNorm);
            this._rmsnorm(this._buf.normed, ffnNormF32Buf, this.nEmb);
            ffnNormF32Buf.destroy();

            // --- FFN (dense or MoE) ---
            if (cl.isMoE) {
                await this._moEFFNGPU(l, gl, cl, position);
            } else {
                // Dense SwiGLU
                this._matmul(gl.ffnGate, this._buf.normed, this._buf.gate, cl.ffnGate, this.nFF);
                this._matmul(gl.ffnUp,   this._buf.normed, this._buf.up,   cl.ffnUp,   this.nFF);
                this._siluMul(this._buf.gate, this._buf.up, this.nFF);
                this._matmul(gl.ffnDown, this._buf.gate, this._buf.down, cl.ffnDown, this.nEmb);
                this._addResidual(this._buf.hidden, this._buf.down, this.nEmb);
            }

            if (l % 8 === 7) await yieldToBrowser();
        }

        if (!skipLogits) {
            const outNormBuf = this._getF32WeightBuf(cpu.outputNorm);
            this._copy(this._buf.hidden, this._buf.normed, this.nEmb);
            this._rmsnorm(this._buf.normed, outNormBuf, this.nEmb);
            outNormBuf.destroy();
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

            // Accumulate: moeAcc += w * down (scale on CPU by pre-scaling the readback)
            // For simplicity, read back and accumulate on CPU
            const downCPU = await g.readF32(this._buf.down, this.nEmb);
            // Write weighted version back into a temp and add
            const scaled = new Float32Array(this.nEmb);
            for (let i = 0; i < this.nEmb; i++) scaled[i] = w * downCPU[i];
            g.device.queue.writeBuffer(this._buf.down, 0, scaled);
            this._addResidual(this._buf.moeAcc, this._buf.down, this.nEmb);
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
