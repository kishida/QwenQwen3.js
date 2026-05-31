const QK_Q8_0 = 32;
const BLOCK_SIZE_Q8_0 = 34; // sizeof(block_q8_0) = ggml_half d(2B) + qs(int8_t[32],32B)

// GGUF storage: column-major (ggml internal layout).
// For shape [A, B]: ne[0]=A is fastest-varying. Each of B columns has A elements stored as Q8_0 blocks.

function _blocksPerCol(shape0) {
    return Math.ceil(shape0 / QK_Q8_0);
}

// Cache typed array views on meta object (set once, reused every call)
function _getViews(meta) {
    if (!meta._i8) {
        meta._i8  = new Int8Array(meta.buffer);
        meta._dv  = new DataView(meta.buffer); // for reading floats at arbitrary offsets
    }
    return [meta._i8, meta._dv];
}

// Read ggml_half (F16) from byte offset (little-endian), native DataView
function _readF16(dv, byteOffset) {
    return dv.getFloat16(byteOffset, true);
}

// --- matmul: weight [A, B], GGUF column-major (blocks along ne[0]=A) ---
function matmulQ80xF32(meta, input) {
    const inDim  = meta.shape[0];   // column length (fastest-varying)
    const outDim = meta.shape[1];   // number of columns (= output dimension)
    const blocksPerCol = _blocksPerCol(inDim);

    const [i8, dv] = _getViews(meta);
    const output = new Float32Array(outDim);

    for (let c = 0; c < outDim; c++) {
        let dot = 0;
        const colPtr = meta.offset + c * blocksPerCol * BLOCK_SIZE_Q8_0;

        for (let b = 0; b < blocksPerCol; b++) {
            const blockPtr = colPtr + b * BLOCK_SIZE_Q8_0;
            const d = _readF16(dv, blockPtr);
            const qsOff = blockPtr + 2;
            const elemBase = b * QK_Q8_0;

            for (let i = 0; i < QK_Q8_0 && (elemBase + i) < inDim; i++) {
                dot += input[elemBase + i] * i8[qsOff + i] * d;
            }
        }

        output[c] = dot;
    }

    return output;
}

// --- Embedding lookup: weight [n_embd, n_vocab], GGUF column-major → pick column tokenId ---
function embeddingLookupQ80(meta, tokenId) {
    const outDim = meta.shape[0];  // n_embd (column length, fastest-varying)
    const blocksPerCol = _blocksPerCol(outDim);

    const [i8, dv] = _getViews(meta);
    const output = new Float32Array(outDim);

    const colPtr = meta.offset + tokenId * blocksPerCol * BLOCK_SIZE_Q8_0;

    for (let b = 0; b < blocksPerCol; b++) {
        const blockPtr = colPtr + b * BLOCK_SIZE_Q8_0;
        const d = _readF16(dv, blockPtr);
        const qsOff = blockPtr + 2;
        const elemBase = b * QK_Q8_0;

        for (let i = 0; i < QK_Q8_0 && (elemBase + i) < outDim; i++) {
            output[elemBase + i] = i8[qsOff + i] * d;
        }
    }

    return output;
}

// --- Read 1D weight as F32 (handles both F32 and Q8_0 types) ---
function _readWeightF32(weightMeta, n) {
    if (weightMeta.type === 0) { // F32
        return new Float32Array(weightMeta.buffer, weightMeta.offset, n);
    } else if (weightMeta.type === 8) { // Q8_0
        const [i8, dv] = _getViews(weightMeta);
        const out = new Float32Array(n);
        const blocksPerRow = Math.ceil(n / QK_Q8_0);
        for (let b = 0; b < blocksPerRow; b++) {
            const blockPtr = weightMeta.offset + b * BLOCK_SIZE_Q8_0;
            const d = _readF16(dv, blockPtr);
            const qsOff = blockPtr + 2;
            const base = b * QK_Q8_0;
            for (let i = 0; i < QK_Q8_0 && (base + i) < n; i++) {
                out[base + i] = i8[qsOff + i] * d;
            }
        }
        return out;
    } else {
        throw new Error(`Unsupported weight type for norm: ${weightMeta.type}`);
    }
}

// --- RMSNorm: in-place on input, returns same array ---
function rmsnorm(input, weightMeta, eps) {
    const n = input.length;
    let sumSq = 0;
    for (let i = 0; i < n; i++) sumSq += input[i] * input[i];
    const invRms = 1.0 / Math.sqrt(sumSq / n + eps);

    const wF32 = _readWeightF32(weightMeta, n);
    for (let i = 0; i < n; i++) {
        input[i] = wF32[i] * input[i] * invRms;
    }
    return input;
}

// --- Per-head RMSNorm: tensor is [n_heads * head_dim], norm weight is [head_dim] ---
function perHeadRMSNorm(tensor, nHeads, headDim, weightMeta, eps) {
    const wF32 = _readWeightF32(weightMeta, headDim);
    for (let h = 0; h < nHeads; h++) {
        const off = h * headDim;
        let sumSq = 0;
        for (let d = 0; d < headDim; d++) sumSq += tensor[off + d] * tensor[off + d];
        const invRms = 1.0 / Math.sqrt(sumSq / headDim + eps);
        for (let d = 0; d < headDim; d++) {
            tensor[off + d] = wF32[d] * tensor[off + d] * invRms;
        }
    }
    return tensor;
}

// --- Split-half RoPE: (x[i], x[i + half]) pairs rotated ---
function applyRoPE(tensor, nHeads, headDim, position, freqBase) {
    const half = headDim >> 1;
    for (let h = 0; h < nHeads; h++) {
        const off = h * headDim;
        for (let i = 0; i < half; i++) {
            const freq = 1.0 / Math.pow(freqBase, i / half);
            const theta = position * freq;
            const cos = Math.cos(theta);
            const sin = Math.sin(theta);
            const x0 = tensor[off + i];
            const x1 = tensor[off + i + half];
            tensor[off + i]       = x0 * cos - x1 * sin;
            tensor[off + i + half] = x0 * sin + x1 * cos;
        }
    }
}

// --- Softmax in-place ---
function softmaxInPlace(arr, start, end) {
    let maxVal = arr[start];
    for (let i = start + 1; i < end; i++) {
        if (arr[i] > maxVal) maxVal = arr[i];
    }
    let sumExp = 0;
    for (let i = start; i < end; i++) {
        arr[i] = Math.exp(arr[i] - maxVal);
        sumExp += arr[i];
    }
    const invSum = 1.0 / sumExp;
    for (let i = start; i < end; i++) {
        arr[i] *= invSum;
    }
}

// --- SiLU in-place: x * sigmoid(x) ---
function siluInPlace(arr) {
    for (let i = 0; i < arr.length; i++) {
        arr[i] *= 1.0 / (1.0 + Math.exp(-arr[i]));
    }
}

// --- Element-wise multiply: a *= b, in-place on a ---
function mulInPlace(a, b) {
    for (let i = 0; i < a.length; i++) {
        a[i] *= b[i];
    }
}

// --- Add vectors: a += b, in-place on a ---
function addVec(a, b) {
    for (let i = 0; i < a.length; i++) {
        a[i] += b[i];
    }
}

// --- Top-p + top-k sampling ---
function sampleTopPTopK(logits, temperature, topP, topK) {
    const n = logits.length;

    let maxVal = -Infinity;
    for (let i = 0; i < n; i++) {
        if (logits[i] > maxVal) maxVal = logits[i];
    }
    let sumExp = 0;
    for (let i = 0; i < n; i++) {
        logits[i] = Math.exp((logits[i] - maxVal) / temperature);
        sumExp += logits[i];
    }
    const invSum = 1.0 / sumExp;
    for (let i = 0; i < n; i++) {
        logits[i] *= invSum;
    }

    const indexed = new Array(n);
    for (let i = 0; i < n; i++) indexed[i] = [logits[i], i];
    indexed.sort((a, b) => b[0] - a[0]);

    const kLimit = topK > 0 ? Math.min(topK, n) : n;
    for (let i = kLimit; i < n; i++) indexed[i][0] = 0;

    let cumProb = 0;
    for (let i = 0; i < n; i++) {
        cumProb += indexed[i][0];
        if (cumProb >= topP) {
            for (let j = i + 1; j < n; j++) indexed[j][0] = 0;
            break;
        }
    }

    const r = Math.random();
    let cum = 0;
    for (let i = 0; i < n; i++) {
        cum += indexed[i][0];
        if (r <= cum) return indexed[i][1];
    }
    return indexed[n - 1][1];
}

// --- Greedy sampling ---
function sampleGreedy(logits) {
    let maxIdx = 0;
    for (let i = 1; i < logits.length; i++) {
        if (logits[i] > logits[maxIdx]) maxIdx = i;
    }
    return maxIdx;
}

// --- Thinking-mode controlled sampling ---
// thinkOpts: { mode: 'suppress'|'shorten'|'full', thinkId, endThinkId, thinkingDone }
//   suppress: block <think> from being sampled
//   shorten:  if </think> is in top-K, force it (while thinkingDone=false)
//   full:     no control
function sampleWithThinkControl(logits, temperature, topP, topK, thinkOpts) {
    const mode = thinkOpts?.mode ?? 'suppress';

    if (mode === 'suppress') {
        logits[thinkOpts.thinkId] = -Infinity;
    } else if (mode === 'shorten' && !thinkOpts.thinkingDone) {
        // Force </think> if it ranks within top-K
        const endLogit = logits[thinkOpts.endThinkId];
        let rank = 0;
        const n = logits.length;
        for (let i = 0; i < n; i++) {
            if (logits[i] > endLogit) {
                if (++rank >= topK) break;
            }
        }
        if (rank < topK) return thinkOpts.endThinkId;
    }

    return temperature <= 0
        ? sampleGreedy(logits)
        : sampleTopPTopK(logits, temperature, topP, topK);
}

// === k-quant CPU dequantization (Q2_K through Q6_K) ===
// All k-quants: QK_K=256 elements per superblock, column-major storage

const QK_K = 256;
const BLOCK_SIZE_Q2_K = 84;   // d(2)+dmin(2)+scales[16]+qs[64]
const BLOCK_SIZE_Q3_K = 110;  // hmask[32]+qs[64]+scales[12]+d(2)
const BLOCK_SIZE_Q4_K = 144;  // d(2)+dmin(2)+scales[12]+qs[128]
const BLOCK_SIZE_Q5_K = 176;  // d(2)+dmin(2)+scales[12]+qh[32]+qs[128]
const BLOCK_SIZE_Q6_K = 210;  // ql[128]+qh[64]+scales[16]+d(2)

// Extract scale/min for Q4_K/Q5_K 8-sub-block format (12-byte scales array)
function _q4kGetScaleMin(u8, scaleBase, sub) {
    let s, m;
    if (sub < 4) {
        s = u8[scaleBase + sub] & 63;
        m = u8[scaleBase + sub + 4] & 63;
    } else {
        const ss = sub - 4;
        s = (u8[scaleBase + sub + 4] & 0xF) | ((u8[scaleBase + ss] >> 6) << 4);
        m = (u8[scaleBase + sub + 4] >> 4)  | ((u8[scaleBase + sub] >> 6) << 4);
    }
    return [s, m];
}

// Q3_K: extract 6-bit signed scale for sub-block sub (0..15) from 12-byte scales array
// GGML layout (verified against dequantize_row_q3_K + aux[] manipulation):
//   s[0..7]: one lower-4-bits per byte (sub<8 → lower nibble, sub>=8 → upper nibble of s[sub&7])
//   s[8..11]: upper-2-bits packed 4-per-byte; byte = (sub&3)+8, shift = (sub>>2)*2
function _q3kGetScale(u8, scaleBase, sub) {
    // GGML: scales[sub] is 6-bit unsigned (0..63), dequant applies (scales[sub] - 32)
    // s[0..7]: one lo4 per byte; byte = sub&7, nibble = (sub>>3)<<2
    // s[8..11]: hi2 packed 4-per-byte; byte = (sub&3)+8, shift = (sub>>2)*2
    const lo4 = (u8[scaleBase + (sub & 7)] >> ((sub >> 3) << 2)) & 0xF;
    const hi2 = (u8[scaleBase + 8 + (sub & 3)] >> ((sub >> 2) * 2)) & 0x3;
    return (lo4 | (hi2 << 4)) - 32;  // bias-32: maps 0..63 → -32..31
}

function matmulQ2KxF32(meta, input) {
    // Block layout: scales[16]@0 qs[64]@16 d(f16)@80 dmin(f16)@82
    const inDim = meta.shape[0], outDim = meta.shape[1];
    const blocksPerCol = Math.ceil(inDim / QK_K);
    const u8 = new Uint8Array(meta.buffer);
    const dv = new DataView(meta.buffer);
    const output = new Float32Array(outDim);
    for (let c = 0; c < outDim; c++) {
        const colBase = meta.offset + c * blocksPerCol * BLOCK_SIZE_Q2_K;
        let dot = 0;
        for (let b = 0; b < blocksPerCol; b++) {
            const bb = colBase + b * BLOCK_SIZE_Q2_K;
            const d    = dv.getFloat16(bb + 80, true);
            const dmin = dv.getFloat16(bb + 82, true);
            const elemBase = b * QK_K;
            const maxElem = Math.min(QK_K, inDim - elemBase);
            for (let e = 0; e < maxElem; e++) {
                const sub      = e >> 4;
                const scaleByte = u8[bb + sub];           // scales @ 0
                const scale    = scaleByte & 0xF;
                const minVal   = (scaleByte >> 4) & 0xF;
                // Q2_K qs: same interleaved layout as Q4_K — NOT a flat 2-bit array
                const qb = bb + 16 + (e>>7)*32 + ((e>>4)&1)*16 + (e&15);
                const q  = (u8[qb] >> (((e>>5)&3)*2)) & 0x3;
                dot += input[elemBase + e] * (d * scale * q - dmin * minVal);
            }
        }
        output[c] = dot;
    }
    return output;
}

function matmulQ3KxF32(meta, input) {
    // Block layout: hmask[32] qs[64] scales[12] d[2]
    const inDim = meta.shape[0], outDim = meta.shape[1];
    const blocksPerCol = Math.ceil(inDim / QK_K);
    const u8 = new Uint8Array(meta.buffer);
    const dv = new DataView(meta.buffer);
    const output = new Float32Array(outDim);
    for (let c = 0; c < outDim; c++) {
        const colBase = meta.offset + c * blocksPerCol * BLOCK_SIZE_Q3_K;
        let dot = 0;
        for (let b = 0; b < blocksPerCol; b++) {
            const bb = colBase + b * BLOCK_SIZE_Q3_K;
            const d = dv.getFloat16(bb + 108, true);
            const elemBase = b * QK_K;
            const maxElem = Math.min(QK_K, inDim - elemBase);
            for (let e = 0; e < maxElem; e++) {
                const sub = e >> 4;
                const scale = _q3kGetScale(u8, bb + 96, sub);
                // Q3_K hmask: each byte covers 8 groups; byte = l_half*16+l, bit = e>>5
                const hmaskBit = (u8[bb + ((e>>4)&1)*16 + (e&15)] >> (e>>5)) & 1;
                // Q3_K qs: same interleaved layout as Q2_K
                const qb = bb + 32 + (e>>7)*32 + ((e>>4)&1)*16 + (e&15);
                const low2 = (u8[qb] >> (((e>>5)&3)*2)) & 3;
                const q3 = low2 | (hmaskBit << 2);
                dot += input[elemBase + e] * (d * scale * (q3 - 4));
            }
        }
        output[c] = dot;
    }
    return output;
}

function matmulQ4KxF32(meta, input) {
    // Block layout: d[2] dmin[2] scales[12] qs[128]
    const inDim = meta.shape[0], outDim = meta.shape[1];
    const blocksPerCol = Math.ceil(inDim / QK_K);
    const u8 = new Uint8Array(meta.buffer);
    const dv = new DataView(meta.buffer);
    const output = new Float32Array(outDim);
    for (let c = 0; c < outDim; c++) {
        const colBase = meta.offset + c * blocksPerCol * BLOCK_SIZE_Q4_K;
        let dot = 0;
        for (let b = 0; b < blocksPerCol; b++) {
            const bb = colBase + b * BLOCK_SIZE_Q4_K;
            const d = dv.getFloat16(bb, true);
            const dmin = dv.getFloat16(bb + 2, true);
            const elemBase = b * QK_K;
            const maxElem = Math.min(QK_K, inDim - elemBase);
            for (let e = 0; e < maxElem; e++) {
                // chunk=e>>6 (4 chunks of 64), lower half (e&63)<32 → sub even, upper half → sub odd
                const sub = (e >> 6) * 2 + ((e >> 5) & 1);
                const [sc, mn] = _q4kGetScaleMin(u8, bb + 4, sub);
                // each 64-elem chunk uses 32 qs bytes; elements e and e+32 share the same byte
                const qsByte = u8[bb + 16 + (e >> 6) * 32 + (e & 31)];
                const q4 = ((e >> 5) & 1) ? (qsByte >> 4) & 0xF : qsByte & 0xF;
                dot += input[elemBase + e] * (d * sc * q4 - dmin * mn);
            }
        }
        output[c] = dot;
    }
    return output;
}

function matmulQ5KxF32(meta, input) {
    // Block layout: d[2] dmin[2] scales[12] qh[32] qs[128]
    const inDim = meta.shape[0], outDim = meta.shape[1];
    const blocksPerCol = Math.ceil(inDim / QK_K);
    const u8 = new Uint8Array(meta.buffer);
    const dv = new DataView(meta.buffer);
    const output = new Float32Array(outDim);
    for (let c = 0; c < outDim; c++) {
        const colBase = meta.offset + c * blocksPerCol * BLOCK_SIZE_Q5_K;
        let dot = 0;
        for (let b = 0; b < blocksPerCol; b++) {
            const bb = colBase + b * BLOCK_SIZE_Q5_K;
            const d = dv.getFloat16(bb, true);
            const dmin = dv.getFloat16(bb + 2, true);
            const elemBase = b * QK_K;
            const maxElem = Math.min(QK_K, inDim - elemBase);
            for (let e = 0; e < maxElem; e++) {
                const sub = (e >> 6) * 2 + ((e >> 5) & 1);
                const [sc, mn] = _q4kGetScaleMin(u8, bb + 4, sub);
                // qh: 32 bytes, qh[e&31] bit = (e>>6)*2 + ((e>>5)&1)
                const qhBit = (e >> 6) * 2 + ((e >> 5) & 1);
                const highBit = (u8[bb + 16 + (e & 31)] >> qhBit) & 1;
                const qsByte = u8[bb + 48 + (e >> 6) * 32 + (e & 31)];
                const low4 = ((e >> 5) & 1) ? (qsByte >> 4) & 0xF : qsByte & 0xF;
                const q5 = low4 | (highBit << 4);
                dot += input[elemBase + e] * (d * sc * q5 - dmin * mn);
            }
        }
        output[c] = dot;
    }
    return output;
}

function matmulQ6KxF32(meta, input) {
    // Block layout: ql[128] qh[64] scales[16] d[2]
    const inDim = meta.shape[0], outDim = meta.shape[1];
    const blocksPerCol = Math.ceil(inDim / QK_K);
    const u8 = new Uint8Array(meta.buffer);
    const dv = new DataView(meta.buffer);
    const output = new Float32Array(outDim);
    for (let c = 0; c < outDim; c++) {
        const colBase = meta.offset + c * blocksPerCol * BLOCK_SIZE_Q6_K;
        let dot = 0;
        for (let b = 0; b < blocksPerCol; b++) {
            const bb = colBase + b * BLOCK_SIZE_Q6_K;
            const d = dv.getFloat16(bb + 208, true);
            const elemBase = b * QK_K;
            const maxElem = Math.min(QK_K, inDim - elemBase);
            for (let e = 0; e < maxElem; e++) {
                // Q6_K: two 128-element chunks (e>>7), 4-interleaved within each chunk
                // ql layout: chunk*64 + (qgroup&1)*32 + l  where l=e&31, qgroup=(e>>5)&3
                const qlByte = u8[bb + (e >> 7) * 64 + ((e >> 5) & 1) * 32 + (e & 31)];
                // nibble: low for qgroups 0,1 (e&127 < 64), high for qgroups 2,3 (e&127 >= 64)
                const low4 = ((e >> 6) & 1) ? (qlByte >> 4) & 0xF : qlByte & 0xF;
                // qh: chunk*32 + l, shift = qgroup*2
                const qhByte = u8[bb + 128 + (e >> 7) * 32 + (e & 31)];
                const high2 = (qhByte >> (((e >> 5) & 3) * 2)) & 0x3;
                const q6 = low4 | (high2 << 4);
                // scale: chunk*8 + qgroup*2 + (l>>4)
                const scaleIdx = (e >> 7) * 8 + ((e >> 5) & 3) * 2 + ((e & 31) >> 4);
                const scaleByte = u8[bb + 192 + scaleIdx];
                const scale = scaleByte >= 128 ? scaleByte - 256 : scaleByte;
                dot += input[elemBase + e] * (d * scale * (q6 - 32));
            }
        }
        output[c] = dot;
    }
    return output;
}

function matmulF32xF32(meta, input) {
    const inDim = meta.shape[0], outDim = meta.shape[1];
    const dv = new DataView(meta.buffer);
    const output = new Float32Array(outDim);
    for (let c = 0; c < outDim; c++) {
        let dot = 0;
        const base = meta.offset + c * inDim * 4;
        for (let i = 0; i < inDim; i++) {
            dot += input[i] * dv.getFloat32(base + i * 4, true);
        }
        output[c] = dot;
    }
    return output;
}

// Generic matmul dispatcher (handles all tensor types)
function matmulGeneric(meta, input) {
    switch (meta.type) {
        case 0:  return matmulF32xF32(meta, input);
        case 8:  return matmulQ80xF32(meta, input);
        case 10: return matmulQ2KxF32(meta, input);
        case 11: return matmulQ3KxF32(meta, input);
        case 12: return matmulQ4KxF32(meta, input);
        case 13: return matmulQ5KxF32(meta, input);
        case 14: return matmulQ6KxF32(meta, input);
        default: throw new Error(`Unsupported tensor type for matmul: ${meta.type}`);
    }
}

// Generic embedding lookup (row = token ID, one column of the transposed matrix)
function embeddingLookupGeneric(meta, tokenId) {
    const outDim = meta.shape[0];
    switch (meta.type) {
        case 8: return embeddingLookupQ80(meta, tokenId);
        case 0: {
            const out = new Float32Array(outDim);
            const dv = new DataView(meta.buffer);
            const base = meta.offset + tokenId * outDim * 4;
            for (let i = 0; i < outDim; i++) out[i] = dv.getFloat32(base + i * 4, true);
            return out;
        }
        // For k-quants, treat as column lookup (column = tokenId)
        case 10: case 11: case 12: case 13: case 14: {
            const fakeMeta = { ...meta, shape: [outDim, meta.shape[1]] };
            const oneHot = new Float32Array(1);
            oneHot[0] = 1;
            // Simpler: decode the single column
            const input = new Float32Array(outDim);
            // Use matmulGeneric on a single column by constructing a 1-element tensor
            // Actually: just iterate through the block for column tokenId
            return _lookupKQuantColumn(meta, tokenId);
        }
        default: throw new Error(`Unsupported embedding type: ${meta.type}`);
    }
}

function _lookupKQuantColumn(meta, colIdx) {
    const outDim = meta.shape[0];
    // Build a 1D "matmul" with a 1-element output by selecting one column
    const colMeta = {
        buffer: meta.buffer,
        offset: meta.offset,
        nbytes: meta.nbytes,
        shape: [outDim, 1],  // treat as single-column matrix
        type: meta.type,
    };
    // input is a 1-element vector selecting the colIdx-th column... actually embedding lookup is
    // selecting row tokenId from the weight matrix [vocab × embd], so we need the tokenId-th column
    // since GGUF stores it as [embd, vocab] (column-major, each vocab-token is a column)
    const tmp = new Float32Array(1);
    tmp[0] = 1;
    // We need to bypass matmulGeneric and read the column directly
    switch (meta.type) {
        case 10: return _q2kColumn(meta, colIdx, outDim);
        case 11: return _q3kColumn(meta, colIdx, outDim);
        case 12: return _q4kColumn(meta, colIdx, outDim);
        case 13: return _q5kColumn(meta, colIdx, outDim);
        case 14: return _q6kColumn(meta, colIdx, outDim);
        default: return new Float32Array(outDim);
    }
}

function _q2kColumn(meta, colIdx, outDim) {
    // Block layout: scales[16]@0 qs[64]@16 d(f16)@80 dmin(f16)@82
    const blocksPerCol = Math.ceil(outDim / QK_K);
    const u8 = new Uint8Array(meta.buffer);
    const dv = new DataView(meta.buffer);
    const out = new Float32Array(outDim);
    const colBase = meta.offset + colIdx * blocksPerCol * BLOCK_SIZE_Q2_K;
    let outIdx = 0;
    for (let b = 0; b < blocksPerCol && outIdx < outDim; b++) {
        const bb = colBase + b * BLOCK_SIZE_Q2_K;
        const d    = dv.getFloat16(bb + 80, true);
        const dmin = dv.getFloat16(bb + 82, true);
        const maxElem = Math.min(QK_K, outDim - outIdx);
        for (let e = 0; e < maxElem; e++, outIdx++) {
            const sub       = e >> 4;
            const scaleByte = u8[bb + sub];                  // scales @ 0
            const qb = bb + 16 + (e>>7)*32 + ((e>>4)&1)*16 + (e&15);
            const q  = (u8[qb] >> (((e>>5)&3)*2)) & 0x3;
            out[outIdx] = d * (scaleByte & 0xF) * q - dmin * ((scaleByte >> 4) & 0xF);
        }
    }
    return out;
}

// (Similar column readers for Q3K–Q6K omitted for brevity; they follow the same pattern)
function _q3kColumn(meta, colIdx, outDim) {
    const blocksPerCol = Math.ceil(outDim / QK_K);
    const u8 = new Uint8Array(meta.buffer);
    const dv = new DataView(meta.buffer);
    const out = new Float32Array(outDim);
    const colBase = meta.offset + colIdx * blocksPerCol * BLOCK_SIZE_Q3_K;
    let outIdx = 0;
    for (let b = 0; b < blocksPerCol && outIdx < outDim; b++) {
        const bb = colBase + b * BLOCK_SIZE_Q3_K;
        const d = dv.getFloat16(bb + 108, true);
        const maxElem = Math.min(QK_K, outDim - outIdx);
        for (let e = 0; e < maxElem; e++, outIdx++) {
            const scale = _q3kGetScale(u8, bb + 96, e >> 4);
            const hmaskBit = (u8[bb + ((e>>4)&1)*16 + (e&15)] >> (e>>5)) & 1;
            const qb = bb + 32 + (e>>7)*32 + ((e>>4)&1)*16 + (e&15);
            const low2 = (u8[qb] >> (((e>>5)&3)*2)) & 3;
            out[outIdx] = d * scale * ((low2 | (hmaskBit << 2)) - 4);
        }
    }
    return out;
}
function _q4kColumn(meta, colIdx, outDim) {
    const blocksPerCol = Math.ceil(outDim / QK_K);
    const u8 = new Uint8Array(meta.buffer);
    const dv = new DataView(meta.buffer);
    const out = new Float32Array(outDim);
    const colBase = meta.offset + colIdx * blocksPerCol * BLOCK_SIZE_Q4_K;
    let outIdx = 0;
    for (let b = 0; b < blocksPerCol && outIdx < outDim; b++) {
        const bb = colBase + b * BLOCK_SIZE_Q4_K;
        const d = dv.getFloat16(bb, true), dmin = dv.getFloat16(bb + 2, true);
        const maxElem = Math.min(QK_K, outDim - outIdx);
        for (let e = 0; e < maxElem; e++, outIdx++) {
            const sub = (e >> 6) * 2 + ((e >> 5) & 1);
            const [sc, mn] = _q4kGetScaleMin(u8, bb + 4, sub);
            const qsByte = u8[bb + 16 + (e >> 6) * 32 + (e & 31)];
            const q4 = ((e >> 5) & 1) ? (qsByte >> 4) & 0xF : qsByte & 0xF;
            out[outIdx] = d * sc * q4 - dmin * mn;
        }
    }
    return out;
}
function _q5kColumn(meta, colIdx, outDim) {
    const blocksPerCol = Math.ceil(outDim / QK_K);
    const u8 = new Uint8Array(meta.buffer);
    const dv = new DataView(meta.buffer);
    const out = new Float32Array(outDim);
    const colBase = meta.offset + colIdx * blocksPerCol * BLOCK_SIZE_Q5_K;
    let outIdx = 0;
    for (let b = 0; b < blocksPerCol && outIdx < outDim; b++) {
        const bb = colBase + b * BLOCK_SIZE_Q5_K;
        const d = dv.getFloat16(bb, true), dmin = dv.getFloat16(bb + 2, true);
        const maxElem = Math.min(QK_K, outDim - outIdx);
        for (let e = 0; e < maxElem; e++, outIdx++) {
            const sub = (e >> 6) * 2 + ((e >> 5) & 1);
            const [sc, mn] = _q4kGetScaleMin(u8, bb + 4, sub);
            const qhBit = (e >> 6) * 2 + ((e >> 5) & 1);
            const highBit = (u8[bb + 16 + (e & 31)] >> qhBit) & 1;
            const qsByte = u8[bb + 48 + (e >> 6) * 32 + (e & 31)];
            const low4 = ((e >> 5) & 1) ? (qsByte >> 4) & 0xF : qsByte & 0xF;
            out[outIdx] = d * sc * (low4 | (highBit << 4)) - dmin * mn;
        }
    }
    return out;
}
function _q6kColumn(meta, colIdx, outDim) {
    const blocksPerCol = Math.ceil(outDim / QK_K);
    const u8 = new Uint8Array(meta.buffer);
    const dv = new DataView(meta.buffer);
    const out = new Float32Array(outDim);
    const colBase = meta.offset + colIdx * blocksPerCol * BLOCK_SIZE_Q6_K;
    let outIdx = 0;
    for (let b = 0; b < blocksPerCol && outIdx < outDim; b++) {
        const bb = colBase + b * BLOCK_SIZE_Q6_K;
        const d = dv.getFloat16(bb + 208, true);
        const maxElem = Math.min(QK_K, outDim - outIdx);
        for (let e = 0; e < maxElem; e++, outIdx++) {
            const qlByte = u8[bb + (e >> 7) * 64 + ((e >> 5) & 1) * 32 + (e & 31)];
            const low4 = ((e >> 6) & 1) ? (qlByte >> 4) & 0xF : qlByte & 0xF;
            const qhByte = u8[bb + 128 + (e >> 7) * 32 + (e & 31)];
            const high2 = (qhByte >> (((e >> 5) & 3) * 2)) & 0x3;
            const scaleIdx = (e >> 7) * 8 + ((e >> 5) & 3) * 2 + ((e & 31) >> 4);
            const scaleByte = u8[bb + 192 + scaleIdx];
            const scale = scaleByte >= 128 ? scaleByte - 256 : scaleByte;
            out[outIdx] = d * scale * ((low4 | (high2 << 4)) - 32);
        }
    }
    return out;
}

// --- Logits from Q8_0 weight [emb_dim, n_vocab], GGUF column-major ---
function logitsFromQ80(meta, hidden) {
    const embDim  = meta.shape[0];   // column length (fastest-varying)
    const outDim  = meta.shape[1];   // number of columns (= vocab size)
    const blocksPerCol = _blocksPerCol(embDim);

    const [i8, dv] = _getViews(meta);
    const logits = new Float32Array(outDim);

    for (let c = 0; c < outDim; c++) {
        let dot = 0;
        const colPtr = meta.offset + c * blocksPerCol * BLOCK_SIZE_Q8_0;

        for (let b = 0; b < blocksPerCol; b++) {
            const blockPtr = colPtr + b * BLOCK_SIZE_Q8_0;
            const d = _readF16(dv, blockPtr);
            const qsOff = blockPtr + 2;
            const elemBase = b * QK_Q8_0;

            for (let i = 0; i < QK_Q8_0 && (elemBase + i) < embDim; i++) {
                dot += hidden[elemBase + i] * i8[qsOff + i] * d;
            }
        }

        logits[c] = dot;
    }

    return logits;
}
