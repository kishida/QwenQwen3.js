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
