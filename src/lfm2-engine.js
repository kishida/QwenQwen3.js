// ============================================================
// Lfm2Engine — CPU inference engine for LFM2 / LFM2-MoE
//
// Architecture differences from Qwen3:
//   - Hybrid layers: 18 recurrent (short-conv) + 6 full-attention
//   - KV cache: attention layers only (6 slots, remapped)
//   - Conv state: fixed 2-step rolling buffer per recurrent layer
//   - MoE routing: sigmoid (not softmax) + expert_bias for selection
//   - Output norm tensor: "token_embd_norm.weight" (not "output_norm.weight")
//   - Expert bias tensor: "blk.N.exp_probs_b.bias"
//   - Short conv tensors: "blk.N.shortconv.{conv,in_proj,out_proj}.weight"
// ============================================================

class Lfm2Engine {
    constructor(gguf) {
        const archName = gguf.getKeyValue('general.architecture', 'lfm2moe');
        const p = archName + '.';

        this.nLayers      = Number(gguf.getKeyValue(p + 'block_count'))                         || 24;
        this.nEmb         = Number(gguf.getKeyValue(p + 'embedding_length'))                     || 2048;
        this.nHeads       = Number(gguf.getKeyValue(p + 'attention.head_count'))                 || 32;
        this.eps          = Number(gguf.getKeyValue(p + 'attention.layer_norm_rms_epsilon'))      || 1e-5;
        this.nExperts     = Number(gguf.getKeyValue(p + 'expert_count'))                         || 32;
        this.nExpertsUsed = Number(gguf.getKeyValue(p + 'expert_used_count'))                    || 4;
        this.nFFExp       = Number(gguf.getKeyValue(p + 'expert_feed_forward_length'))           || 1792;
        this.nDenseLead   = Number(gguf.getKeyValue(p + 'leading_dense_block_count'))            || 2;
        this.lCache       = Number(gguf.getKeyValue(p + 'shortconv.l_cache'))                    || 3;
        this.dConv        = this.lCache - 1;   // = 2 (rolling state depth)
        this.maxCtx       = Math.min(Number(gguf.getKeyValue(p + 'context_length')) || 4096, 4096);
        this.isMoE        = true;

        // rope_theta: LFM2 uses 5,000,000
        const ropeFreqBase = gguf.getKeyValue(p + 'rope.freq_base')
                          || gguf.getKeyValue(p + 'rope_freq_base');
        this.ropeFreqBase = Number(ropeFreqBase) || 5_000_000;

        // nHeadKV: per-layer array (conv layers = 0) or scalar
        const headKVRaw = gguf.getKeyValue(p + 'attention.head_count_kv');
        if (Array.isArray(headKVRaw)) {
            this.nHeadKV = Math.max(...headKVRaw.filter(v => v > 0));
        } else {
            this.nHeadKV = Number(headKVRaw) || 8;
        }

        this.gguf = gguf;

        // Tokenizer (BPE, shared with Qwen3)
        this.tokenizer = new BPETokenizer();
        this.tokenizer.loadFromGGUF(gguf);

        // Load tensor metadata
        this._loadWeights(gguf);

        // nVocab from embedding shape [n_embd, n_vocab]
        this.nVocab = this.tokEmbd.shape[1];

        // Derive headDim from first attention layer
        const firstAttn = this.layers.find(l => !l.isRecurrent);
        this.headDim = firstAttn ? Math.floor(firstAttn.wq.shape[1] / this.nHeads) : 64;
        this.nKV     = firstAttn ? firstAttn.wk.shape[1] : this.nHeadKV * this.headDim;

        // Dense FFN intermediate dim (layers 0,1)
        const denseLay = this.layers.find(l => l.isDense && l.ffnGate);
        this.nFF = denseLay ? denseLay.ffnGate.shape[1] : 7168;

        this._allocBuffers();

        // KV cache — only for the 6 attention layers
        this.nAttnLayers = this.layers.filter(l => !l.isRecurrent).length;
        this.kvCache = new KVCache(this.nAttnLayers, this.nHeadKV, this.maxCtx, this.headDim);

        // Conv states — one per recurrent layer
        this._initConvStates();
    }

    // ---- Weight loading ----

    _loadWeights(gguf) {
        this.tokEmbd    = gguf.getTensorMetaByName('token_embd.weight');
        // LFM2 uses "token_embd_norm" as output norm (not "output_norm")
        this.outputNorm = gguf.getTensorMetaByName('token_embd_norm.weight');
        const outputW   = gguf.getTensorMetaByName('output.weight');
        this.output     = outputW || this.tokEmbd;   // tied embeddings if absent

        this.layers = new Array(this.nLayers);
        let attnCacheIdx = 0;

        for (let i = 0; i < this.nLayers; i++) {
            const blk = `blk.${i}`;
            const convConv = gguf.getTensorMetaByName(`${blk}.shortconv.conv.weight`);
            const isRecurrent = convConv !== null;
            const isMoE       = i >= this.nDenseLead;
            const isDense     = !isMoE;

            const layer = {
                isRecurrent,
                isMoE,
                isDense,
                attnCacheIdx: isRecurrent ? -1 : attnCacheIdx,
                // Shared norms (used by both conv and attn layers)
                attnNorm: gguf.getTensorMetaByName(`${blk}.attn_norm.weight`),
                ffnNorm:  gguf.getTensorMetaByName(`${blk}.ffn_norm.weight`),
            };

            if (!isRecurrent) {
                // Full-attention layer weights
                layer.wq     = gguf.getTensorMetaByName(`${blk}.attn_q.weight`);
                layer.wk     = gguf.getTensorMetaByName(`${blk}.attn_k.weight`);
                layer.wv     = gguf.getTensorMetaByName(`${blk}.attn_v.weight`);
                layer.wo     = gguf.getTensorMetaByName(`${blk}.attn_output.weight`);
                layer.qNorm  = gguf.getTensorMetaByName(`${blk}.attn_q_norm.weight`);
                layer.kNorm  = gguf.getTensorMetaByName(`${blk}.attn_k_norm.weight`);
                attnCacheIdx++;
            } else {
                // Short-conv recurrent layer weights
                layer.shortconvConv    = convConv;
                layer.shortconvInProj  = gguf.getTensorMetaByName(`${blk}.shortconv.in_proj.weight`);
                layer.shortconvOutProj = gguf.getTensorMetaByName(`${blk}.shortconv.out_proj.weight`);

                // Pre-dequantize the tiny conv kernel into a plain Float32Array for fast access.
                // Shape [L_cache=3, n_embd=2048], column-major → element (k,d) at index d*L+k.
                layer.convKernelF32 = this._dequantConvKernel(convConv);
            }

            if (isDense) {
                layer.ffnGate = gguf.getTensorMetaByName(`${blk}.ffn_gate.weight`);
                layer.ffnUp   = gguf.getTensorMetaByName(`${blk}.ffn_up.weight`);
                layer.ffnDown = gguf.getTensorMetaByName(`${blk}.ffn_down.weight`);
            } else {
                // MoE
                layer.routerWeight  = gguf.getTensorMetaByName(`${blk}.ffn_gate_inp.weight`);
                layer.ffnGateExps   = gguf.getTensorMetaByName(`${blk}.ffn_gate_exps.weight`);
                layer.ffnUpExps     = gguf.getTensorMetaByName(`${blk}.ffn_up_exps.weight`);
                layer.ffnDownExps   = gguf.getTensorMetaByName(`${blk}.ffn_down_exps.weight`);
                // expert bias (.bias suffix — used only for top-K selection, not for weighting)
                layer.expProbsBias  = gguf.getTensorMetaByName(`${blk}.exp_probs_b.bias`);
            }

            this.layers[i] = layer;
        }
    }

    // Pre-dequantize conv kernel tensor [L_cache, n_embd] into Float32Array.
    // GGUF column-major: element (k, d) stored at flat index d * L_cache + k.
    // Keeps that layout so during inference we access kernel[d * lCache + k].
    _dequantConvKernel(meta) {
        const L = meta.shape[0];  // L_cache = 3
        const D = meta.shape[1];  // n_embd = 2048
        const n = L * D;

        if (meta.type === 0) {   // F32 — direct view
            // The raw bytes are already (d, k) column-major, matching our access pattern
            return new Float32Array(meta.buffer, meta.offset, n);
        }
        if (meta.type === 8) {   // Q8_0
            const out = new Float32Array(n);
            const dv  = new DataView(meta.buffer);
            const i8  = new Int8Array(meta.buffer);
            // Each "column" (d dimension) has L=3 elements → 1 block of 32, only 3 used
            const blocksPerCol = Math.ceil(L / 32);
            for (let d = 0; d < D; d++) {
                const colBase = meta.offset + d * blocksPerCol * 34;
                const scale   = dv.getFloat16(colBase, true);
                for (let k = 0; k < L; k++) {
                    out[d * L + k] = i8[colBase + 2 + k] * scale;
                }
            }
            return out;
        }
        if (meta.type === 1) {   // F16
            const out = new Float32Array(n);
            const dv  = new DataView(meta.buffer);
            for (let i = 0; i < n; i++) {
                out[i] = dv.getFloat16(meta.offset + i * 2, true);
            }
            return out;
        }
        throw new Error(`Unsupported conv kernel type: ${meta.type}`);
    }

    // ---- Buffer allocation ----

    _allocBuffers() {
        const E = this.nEmb;
        this.bufHidden     = new Float32Array(E);
        this.bufNormed     = new Float32Array(E);
        this.bufConvBCX    = new Float32Array(3 * E);   // in_proj output [B, C, x]
        this.bufConvBX     = new Float32Array(E);       // bx = B * x
        this.bufConvOut    = new Float32Array(E);       // conv output
        this.bufConvY      = new Float32Array(E);       // y = C * conv_out
        this.bufQ          = new Float32Array(this.nHeads   * this.headDim);
        this.bufK          = new Float32Array(this.nHeadKV  * this.headDim);
        this.bufV          = new Float32Array(this.nHeadKV  * this.headDim);
        this.bufAttnOut    = new Float32Array(this.nHeads   * this.headDim);
        this.bufAttnScores = new Float32Array(this.nHeads   * this.maxCtx);
        this.bufFFN        = new Float32Array(Math.max(this.nFF, this.nFFExp));
        this.bufFFNTmp     = new Float32Array(Math.max(this.nFF, this.nFFExp));
        this.bufMoEOut     = new Float32Array(E);
        this.bufLogits     = new Float32Array(this.nVocab);
    }

    // ---- Conv state init ----

    _initConvStates() {
        this.convStates = new Array(this.nLayers).fill(null);
        for (let i = 0; i < this.nLayers; i++) {
            if (this.layers[i].isRecurrent) {
                // [d_conv=2, n_embd=2048] as flat Float32Array, initialized to zero
                this.convStates[i] = new Float32Array(this.dConv * this.nEmb);
            }
        }
    }

    // ---- State reset (call at start of each new conversation) ----

    resetState() {
        this.kvCache.reset();
        for (const state of this.convStates) {
            if (state) state.fill(0);
        }
    }

    // Alias for compatibility with chat.html calling engine.resetKVCache()
    resetKVCache() { this.resetState(); }

    // ---- Forward pass (single token) ----

    forward(tokenId, position, skipLogits = false) {
        const emb = embeddingLookupGeneric(this.tokEmbd, tokenId);
        this.bufHidden.set(emb);

        for (let l = 0; l < this.nLayers; l++) {
            const layer = this.layers[l];

            // --- Operator sub-block (attention or short-conv) ---
            // attn_norm is used as the "operator norm" for both layer types
            this.bufNormed.set(this.bufHidden);
            rmsnorm(this.bufNormed, layer.attnNorm, this.eps);

            let opOut;
            if (layer.isRecurrent) {
                opOut = this._shortConv(l, layer);
            } else {
                opOut = this._selfAttention(l, layer, position);
            }

            // Residual
            addVec(this.bufHidden, opOut);

            // --- FFN sub-block ---
            this.bufNormed.set(this.bufHidden);
            rmsnorm(this.bufNormed, layer.ffnNorm, this.eps);

            if (layer.isDense) {
                this._denseFfn(layer);
            } else {
                this._moEFFN(l, layer);
            }
        }

        if (!skipLogits) {
            // Output norm ("token_embd_norm.weight" for LFM2)
            rmsnorm(this.bufHidden, this.outputNorm, this.eps);
            const logitsResult = matmulGeneric(this.output, this.bufHidden);
            this.bufLogits.set(logitsResult);
        }

        return skipLogits ? null : this.bufLogits;
    }

    // ---- Short-Conv recurrent layer ----
    //
    // Formula (from llama.cpp lfm2.cpp build_shortconv_block):
    //   bcx  = in_proj(normed)               [3 * n_embd]
    //   B, C, x = split(bcx, 3)              each [n_embd]
    //   bx   = B * x                          element-wise gate
    //   // prepend conv state (last dConv steps of bx), apply depthwise conv
    //   conv_out[d] = Σ_k kernel[k,d] * window[k,d]
    //   // window = [state[-2], state[-1], bx]  (kernel size 3)
    //   y    = C * conv_out                   element-wise gate
    //   out  = out_proj(y)                    [n_embd]
    _shortConv(layerIdx, layer) {
        const E = this.nEmb;
        const L = this.lCache;    // = 3
        const D = this.dConv;     // = 2

        // 1. in_proj: [n_embd] → [3 * n_embd]
        const bcxResult = matmulGeneric(layer.shortconvInProj, this.bufNormed);
        this.bufConvBCX.set(bcxResult);

        // 2. Split into B, C, x (each [n_embd])
        // B = BCX[0 .. E-1], C = BCX[E .. 2E-1], x = BCX[2E .. 3E-1]

        // 3. bx = B * x
        for (let d = 0; d < E; d++) {
            this.bufConvBX[d] = this.bufConvBCX[d] * this.bufConvBCX[2 * E + d];
        }

        // 4. Depthwise causal conv1d
        //    kernel shape [L_cache=3, n_embd=2048], column-major → kernel[k,d] at flat[d*L+k]
        //    state: [dConv=2, n_embd], flat → state[k*E + d] for lag k (0=oldest)
        const state  = this.convStates[layerIdx];
        const kernel = layer.convKernelF32;  // pre-dequantized, layout [d * L + k]

        for (let d = 0; d < E; d++) {
            // window = [state[0,d], state[1,d], bx[d]]
            this.bufConvOut[d] = kernel[d * L + 0] * state[0 * E + d]
                               + kernel[d * L + 1] * state[1 * E + d]
                               + kernel[d * L + 2] * this.bufConvBX[d];
        }

        // 5. Update conv state: shift left, append bx
        //    state[0, :] ← state[1, :]
        //    state[1, :] ← bx
        state.copyWithin(0, E, 2 * E);           // state[0] = state[1]
        state.set(this.bufConvBX, E);            // state[1] = bx

        // 6. y = C * conv_out  (C = BCX[E .. 2E-1])
        for (let d = 0; d < E; d++) {
            this.bufConvY[d] = this.bufConvBCX[E + d] * this.bufConvOut[d];
        }

        // 7. out_proj: [n_embd] → [n_embd]
        return matmulGeneric(layer.shortconvOutProj, this.bufConvY);
    }

    // ---- GQA Self-attention ----

    _selfAttention(layerIdx, layer, position) {
        const cacheIdx = layer.attnCacheIdx;

        // Q / K / V projections (bufNormed already set by caller)
        const qOut = matmulGeneric(layer.wq, this.bufNormed);
        this.bufQ.set(qOut);
        const kOut = matmulGeneric(layer.wk, this.bufNormed);
        this.bufK.set(kOut);
        const vOut = matmulGeneric(layer.wv, this.bufNormed);
        this.bufV.set(vOut);

        // Per-head RMSNorm on Q and K
        perHeadRMSNorm(this.bufQ, this.nHeads,   this.headDim, layer.qNorm, this.eps);
        perHeadRMSNorm(this.bufK, this.nHeadKV,  this.headDim, layer.kNorm, this.eps);

        // RoPE (split-half, NeoX style)
        applyRoPE(this.bufQ, this.nHeads,  this.headDim, position, this.ropeFreqBase);
        applyRoPE(this.bufK, this.nHeadKV, this.headDim, position, this.ropeFreqBase);

        // Store K/V into the per-attention-layer cache slot
        this.kvCache.store(cacheIdx, position, this.bufK, this.bufV);

        // GQA: QK^T + softmax + weighted V sum
        const seqLen = position + 1;
        const scale  = 1.0 / Math.sqrt(this.headDim);
        const scores = this.bufAttnScores;

        for (let h = 0; h < this.nHeads; h++) {
            const kvHead    = (h * this.nHeadKV / this.nHeads) | 0;
            const qOff      = h * this.headDim;
            const scoreOff  = h * seqLen;

            for (let pos = 0; pos < seqLen; pos++) {
                let dot = 0;
                const kSlice = this.kvCache.getK(cacheIdx, kvHead, pos, pos + 1);
                for (let d = 0; d < this.headDim; d++) {
                    dot += this.bufQ[qOff + d] * kSlice[d];
                }
                scores[scoreOff + pos] = dot * scale;
            }
            softmaxInPlace(scores, scoreOff, scoreOff + seqLen);
        }

        this.bufAttnOut.fill(0);
        for (let h = 0; h < this.nHeads; h++) {
            const kvHead   = (h * this.nHeadKV / this.nHeads) | 0;
            const outOff   = h * this.headDim;
            const scoreOff = h * seqLen;
            for (let pos = 0; pos < seqLen; pos++) {
                const w = scores[scoreOff + pos];
                if (w === 0) continue;
                const vSlice = this.kvCache.getV(cacheIdx, kvHead, pos, pos + 1);
                for (let d = 0; d < this.headDim; d++) {
                    this.bufAttnOut[outOff + d] += vSlice[d] * w;
                }
            }
        }

        // Output projection — residual is added by caller (forward())
        const projResult = matmulGeneric(layer.wo, this.bufAttnOut);
        return projResult;
    }

    // ---- Dense FFN (SwiGLU) ----

    _denseFfn(layer) {
        const gateOut = matmulGeneric(layer.ffnGate, this.bufNormed);
        this.bufFFN.set(gateOut);
        siluInPlace(this.bufFFN);
        const upOut = matmulGeneric(layer.ffnUp, this.bufNormed);
        this.bufFFNTmp.set(upOut);
        mulInPlace(this.bufFFN, this.bufFFNTmp);
        const downOut = matmulGeneric(layer.ffnDown, this.bufFFN);
        addVec(this.bufHidden, downOut);
    }

    // ---- MoE FFN (sigmoid routing with expert bias) ----
    //
    // Key difference from Qwen3:
    //   1. routing_weights = sigmoid(logits)       (not softmax)
    //   2. selection_scores = routing_weights + exp_probs_bias
    //      → top-K selection on selection_scores
    //   3. actual weights = routing_weights[selected]  (bias NOT applied to weights)
    //   4. normalize: weights /= (sum + 1e-6)          (norm_topk_prob = true)
    _moEFFN(layerIdx, layer) {
        const nExperts = this.nExperts;
        const topK     = this.nExpertsUsed;

        // 1. Router logits
        const logits = matmulGeneric(layer.routerWeight, this.bufNormed);

        // 2. sigmoid activation → routing weights
        const routingWeights = new Float32Array(nExperts);
        for (let e = 0; e < nExperts; e++) {
            routingWeights[e] = 1.0 / (1.0 + Math.exp(-logits[e]));
        }

        // 3. Expert bias for selection (load_balancing bias, F32 vector)
        const selectionScores = new Float32Array(routingWeights);  // copy
        if (layer.expProbsBias) {
            const bias = _readWeightF32(layer.expProbsBias, nExperts);
            for (let e = 0; e < nExperts; e++) selectionScores[e] += bias[e];
        }

        // 4. Top-K selection on biased scores
        const sorted = Array.from(selectionScores).map((v, i) => [v, i]);
        sorted.sort((a, b) => b[0] - a[0]);
        const topIndices = [];
        const topWeights = [];   // unbiased sigmoid weights
        for (let k = 0; k < topK; k++) {
            const eidx = sorted[k][1];
            topIndices.push(eidx);
            topWeights.push(routingWeights[eidx]);   // unbiased
        }

        // 5. Normalize (norm_topk_prob = true)
        let sumW = 1e-6;
        for (const w of topWeights) sumW += w;
        const invSum = 1.0 / sumW;

        // Notify visualization callbacks (same interface as Qwen3Engine)
        if (this.onRouterUpdate) {
            this.onRouterUpdate({
                layer: layerIdx,
                probs: Array.from(routingWeights),   // sigmoid probs (0..1, no sum=1 guarantee)
                selected: topIndices,
                weights: topWeights.map(w => w * invSum),
            });
        }

        // 6. Accumulate expert outputs
        this.bufMoEOut.fill(0);

        for (let k = 0; k < topK; k++) {
            const eidx   = topIndices[k];
            if (this.expertMask && this.expertMask.has(eidx)) continue;
            const weight = topWeights[k] * invSum;

            const gateSlice = this._expertSlice(layer.ffnGateExps, eidx);
            const upSlice   = this._expertSlice(layer.ffnUpExps,   eidx);
            const downSlice = this._expertSlice(layer.ffnDownExps, eidx);

            // SwiGLU: silu(gate) * up
            const gateOut = matmulGeneric(gateSlice, this.bufNormed);
            siluInPlace(gateOut);
            const upOut = matmulGeneric(upSlice, this.bufNormed);
            mulInPlace(gateOut, upOut);

            // Down projection
            const downOut = matmulGeneric(downSlice, gateOut);

            // Weighted accumulate
            for (let i = 0; i < downOut.length; i++) {
                this.bufMoEOut[i] += weight * downOut[i];
            }
        }

        addVec(this.bufHidden, this.bufMoEOut);
    }

    // Slice a single expert's weights from a 3D packed tensor [inDim, hidDim, nExperts]
    _expertSlice(meta, expertIdx) {
        const inDim  = meta.shape[0];
        const hidDim = meta.shape[1];
        const blockSize     = this._blockSizeForType(meta.type);
        const elemsPerBlock = meta.type >= 10 ? 256 : 32;
        const blocksPerCol  = Math.ceil(inDim / elemsPerBlock);
        const bytesPerCol   = blocksPerCol * blockSize;
        const expertOffset  = expertIdx * hidDim * bytesPerCol;
        return {
            buffer: meta.buffer,
            offset: meta.offset + expertOffset,
            nbytes: hidDim * bytesPerCol,
            shape:  [inDim, hidDim],
            type:   meta.type,
        };
    }

    _blockSizeForType(type) {
        switch (type) {
            case 0:  return 4;    // F32
            case 1:  return 2;    // F16
            case 8:  return 34;   // Q8_0
            case 10: return 84;   // Q2_K
            case 11: return 110;  // Q3_K
            case 12: return 144;  // Q4_K
            case 13: return 176;  // Q5_K
            case 14: return 210;  // Q6_K
            default: return 34;
        }
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

        // Reset KV cache + conv states for a fresh generation
        this.resetState();

        // Prefill: process all prompt tokens except the last (skip logits)
        // Always yield to browser — LFM2 is CPU-only and each token takes tens of ms,
        // so the UI would freeze completely without yielding on every token.
        for (let i = 0; i < tokenIds.length - 1; i++) {
            if (abortSignal?.aborted) return;
            this.forward(tokenIds[i], i, true);
            if (onPrefill) onPrefill(i + 1, tokenIds.length);
            await yieldToBrowser();
        }

        // Last prompt token → compute logits to sample first output token
        if (abortSignal?.aborted) return;
        this.forward(tokenIds[tokenIds.length - 1], tokenIds.length - 1, false);

        let currentToken = sampleWithThinkControl(this.bufLogits, temperature, topP, topK, thinkOpts);

        const generatedTokens = [];

        for (let step = 0; step < maxSteps; step++) {
            if (currentToken === this.tokenizer.eosTokenId) break;
            if (this.tokenizer.stopTokenIds?.has(currentToken)) break;

            if (currentToken === thinkOpts.endThinkId) thinkOpts.thinkingDone = true;

            if (abortSignal?.aborted) return;
            this.forward(currentToken, tokenIds.length + step);

            generatedTokens.push(currentToken);
            if (onToken) onToken(currentToken);

            currentToken = sampleWithThinkControl(this.bufLogits, temperature, topP, topK, thinkOpts);

            await yieldToBrowser();
        }

        if (onFinish) onFinish(generatedTokens);
    }

    // ---- Chat template ----

    formatChat(messages, systemPrompt, thinkingMode = 'suppress') {
        // LFM2's GGUF chat template uses macros, namespace(), filters, bracket indexing,
        // loop.index0, etc. — too complex to render dynamically.
        // The template produces standard ChatML for simple chat (no tools),
        // so we hardcode that directly.
        return this._chatML(messages, systemPrompt, thinkingMode);
    }

    // LFM2 uses ChatML format: <|im_start|>role\ncontent<|im_end|>\n
    _chatML(messages, systemPrompt, thinkingMode = 'suppress') {
        let text = '<|startoftext|>';
        if (systemPrompt) {
            text += `<|im_start|>system\n${systemPrompt}<|im_end|>\n`;
        }
        for (const msg of messages) {
            const role = msg.role || 'user';
            const content = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
            text += `<|im_start|>${role}\n${content}<|im_end|>\n`;
        }
        text += '<|im_start|>assistant\n';
        if (thinkingMode === 'suppress') {
            text += '<think>\n\n</think>\n';
        }
        // shorten / full: no prefix — LFM2 opens <think> naturally
        return text;
    }

    // Full Jinja2-like template renderer.
    // Handles the {% for %}...{% endfor %} / {% if %}...{% elif %}...{% endif %}
    // patterns used in real GGUF chat templates (LFM2 uses a for-loop, NOT {{ messages }}).
    _applyChatTemplate(template, messages, systemPrompt) {
        const allMessages = [];
        if (systemPrompt) allMessages.push({ role: 'system', content: systemPrompt });
        for (const m of messages) allMessages.push(m);

        const ctx = { messages: allMessages, add_generation_prompt: true };
        return this._j2render(template, ctx);
    }

    _j2render(src, ctx) {
        // Strip Jinja2 comments {# ... #} before anything else
        src = src.replace(/\{#[\s\S]*?#\}/g, '');
        // Normalize whitespace-trim markers ({%- -%} {{- -}})
        src = src.replace(/\{\{-\s*/g, '{{').replace(/\s*-\}\}/g, '}}');
        src = src.replace(/\{%-\s*/g, '{%').replace(/\s*-%\}/g, '%}');

        let out = '', pos = 0;
        while (pos < src.length) {
            const lb = src.indexOf('{', pos);
            if (lb === -1) { out += src.slice(pos); break; }

            if (src[lb + 1] === '%') {
                out += src.slice(pos, lb);
                const rb = src.indexOf('%}', lb + 2);
                if (rb === -1) { out += src.slice(lb); break; }
                const tag = src.slice(lb + 2, rb).trim();
                pos = rb + 2;

                if (tag.startsWith('for ')) {
                    const m = tag.match(/^for\s+(\w+)\s+in\s+([\w.]+)$/);
                    if (m) {
                        const end = this._j2findEnd(src, pos, 'for');
                        const body = src.slice(pos, end.start);
                        pos = end.end;
                        const items = this._j2val(m[2], ctx) || [];
                        for (const item of items) {
                            out += this._j2render(body, { ...ctx, [m[1]]: item });
                        }
                    }
                } else if (tag.startsWith('if ')) {
                    const { branches, end } = this._j2parseIf(src, pos, tag.slice(3).trim());
                    pos = end;
                    for (const br of branches) {
                        if (br.cond === null || this._j2cond(br.cond, ctx)) {
                            out += this._j2render(br.body, ctx);
                            break;
                        }
                    }
                }
                // endfor / endif / elif / else: consumed by parent scope, ignored here

            } else if (src[lb + 1] === '{') {
                out += src.slice(pos, lb);
                const rb = src.indexOf('}}', lb + 2);
                if (rb === -1) { out += src.slice(lb); break; }
                const expr = src.slice(lb + 2, rb).trim();
                pos = rb + 2;
                out += String(this._j2expr(expr, ctx) ?? '');
            } else {
                out += src.slice(pos, lb + 1);
                pos = lb + 1;
            }
        }
        return out;
    }

    _j2val(name, ctx) {
        // dot-path: "messages", "message.role", etc.
        const parts = name.split('.');
        let v = ctx[parts[0]];
        for (let i = 1; i < parts.length; i++) {
            if (v == null) return undefined;
            v = v[parts[i]];
        }
        return v;
    }

    _j2expr(expr, ctx) {
        expr = expr.trim();
        // String literal — match only if no unescaped inner quotes of same type
        const strLit = expr.match(/^'([^']*)'$/) || expr.match(/^"([^"]*)"$/);
        if (strLit) return strLit[1].replace(/\\n/g, '\n').replace(/\\t/g, '\t');
        // Concatenation via + or ~ (split outside of string literals)
        if (expr.includes('+') || expr.includes('~')) {
            const parts = _j2splitConcat(expr);
            if (parts.length > 1) return parts.map(p => this._j2expr(p.trim(), ctx)).join('');
        }
        // Attribute access / variable
        return this._j2val(expr, ctx) ?? '';
    }

    _j2cond(cond, ctx) {
        cond = cond.trim();

        // 'not X'
        if (cond.startsWith('not ')) return !this._j2cond(cond.slice(4).trim(), ctx);

        // 'X and Y' — split at top level (outside string literals)
        const andParts = this._j2splitLogic(cond, ' and ');
        if (andParts.length > 1) return andParts.every(p => this._j2cond(p.trim(), ctx));

        // 'X or Y'
        const orParts = this._j2splitLogic(cond, ' or ');
        if (orParts.length > 1) return orParts.some(p => this._j2cond(p.trim(), ctx));

        // 'X is not none'
        const isNotNone = cond.match(/^([\w.]+)\s+is\s+not\s+none$/i);
        if (isNotNone) return this._j2val(isNotNone[1], ctx) != null;

        // 'X is not defined'
        const isNotDef = cond.match(/^([\w.]+)\s+is\s+not\s+defined$/i);
        if (isNotDef) return this._j2val(isNotDef[1], ctx) === undefined;

        // 'X is none'
        const isNone = cond.match(/^([\w.]+)\s+is\s+none$/i);
        if (isNone) return this._j2val(isNone[1], ctx) == null;

        // 'X is defined'
        const isDef = cond.match(/^([\w.]+)\s+is\s+defined$/i);
        if (isDef) return this._j2val(isDef[1], ctx) !== undefined;

        // 'X == "Y"'
        const eq = cond.match(/^([\w.]+)\s*==\s*(['"])(.*?)\2$/);
        if (eq) return String(this._j2val(eq[1], ctx)) === eq[3];

        // 'X != "Y"'
        const ne = cond.match(/^([\w.]+)\s*!=\s*(['"])(.*?)\2$/);
        if (ne) return String(this._j2val(ne[1], ctx)) !== ne[3];

        // Truthy check
        return !!this._j2val(cond, ctx);
    }

    // Split expr by separator string, respecting string literal boundaries.
    _j2splitLogic(expr, sep) {
        const parts = []; let i = 0, start = 0;
        while (i <= expr.length - sep.length) {
            if (expr[i] === "'" || expr[i] === '"') {
                const q = expr[i++];
                while (i < expr.length && expr[i] !== q) { if (expr[i] === '\\') i++; i++; }
                i++;
            } else if (expr.startsWith(sep, i)) {
                parts.push(expr.slice(start, i));
                i += sep.length; start = i;
            } else { i++; }
        }
        parts.push(expr.slice(start));
        return parts.length > 1 ? parts : [expr];
    }

    _j2findEnd(src, start, tagName) {
        let depth = 1, pos = start;
        while (pos < src.length) {
            const lb = src.indexOf('{%', pos);
            if (lb === -1) break;
            const rb = src.indexOf('%}', lb);
            if (rb === -1) break;
            const inner = src.slice(lb + 2, rb).trim();
            if (inner === `end${tagName}`) { if (--depth === 0) return { start: lb, end: rb + 2 }; }
            else if (inner.startsWith(`${tagName} `)) depth++;
            pos = rb + 2;
        }
        return { start: src.length, end: src.length };
    }

    _j2parseIf(src, start, firstCond) {
        const branches = [];
        let cond = firstCond, bStart = start, depth = 1, pos = start;
        while (pos < src.length) {
            const lb = src.indexOf('{%', pos);
            if (lb === -1) break;
            const rb = src.indexOf('%}', lb);
            if (rb === -1) break;
            const inner = src.slice(lb + 2, rb).trim();
            if (inner.startsWith('if ') || inner === 'if') {
                depth++; pos = rb + 2;
            } else if (depth === 1 && (inner.startsWith('elif ') || inner === 'else' || inner === 'endif')) {
                branches.push({ cond, body: src.slice(bStart, lb) });
                if (inner === 'endif') return { branches, end: rb + 2 };
                cond = inner === 'else' ? null : inner.slice(5).trim();
                bStart = rb + 2; pos = rb + 2;
            } else if (inner === 'endif') {
                depth--; pos = rb + 2;
            } else {
                pos = rb + 2;
            }
        }
        branches.push({ cond, body: src.slice(bStart) });
        return { branches, end: src.length };
    }
}

// Split "a + b + c" or "a ~ b ~ c" by ' + '/' ~ ' respecting string literal boundaries.
function _j2splitConcat(expr) {
    const parts = []; let i = 0, start = 0;
    while (i < expr.length) {
        if (expr[i] === "'" || expr[i] === '"') {
            const q = expr[i++];
            while (i < expr.length && expr[i] !== q) { if (expr[i] === '\\') i++; i++; }
            i++;
        } else if ((expr[i] === '+' || expr[i] === '~') && i > 0 && expr[i-1] === ' ' && i + 1 < expr.length && expr[i+1] === ' ') {
            parts.push(expr.slice(start, i - 1)); i += 2; start = i;
        } else { i++; }
    }
    parts.push(expr.slice(start));
    return parts;
}
