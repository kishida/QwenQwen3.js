class Qwen3Engine {
    constructor(gguf) {
        const arch = gguf.getArch();
        this.nLayers = arch.blockCount;
        this.nEmb = arch.embeddingLength;
        this.nHeads = arch.headCount;
        this.nHeadKV = arch.headCountKV || arch.headCount;
        this.eps = Number(arch.layerNormRmsEps) || 1e-6;
        this.ropeFreqBase = arch.ropeFreqBase || 10000;
        this.maxCtx = Math.min(arch.contextLength || 8192, 4096);
        this.gguf = gguf;

        // Tokenizer
        this.tokenizer = new BPETokenizer();
        this.tokenizer.loadFromGGUF(gguf);

        // Load tensor metadata (no data copy)
        this._loadWeights(gguf);

        // Derive nVocab from embedding tensor shape[1] (row-major: [n_embd, n_vocab])
        const embShape = this.tokEmbd.shape;
        this.nVocab = embShape[1];

        // Pre-allocate working buffers (derives headDimQ/headDimKV from actual weight shapes)
        this._allocBuffers();

        // KV cache — uses KV head dimension
        this.kvCache = new KVCache(this.nLayers, this.nHeadKV, this.maxCtx, this.headDimKV);
    }

    _loadWeights(gguf) {
        this.tokEmbd = gguf.getTensorMetaByName("token_embd.weight");
        this.outputNorm = gguf.getTensorMetaByName("output_norm.weight");

        const outputTensor = gguf.getTensorMetaByName("output.weight");
        this.output = outputTensor || this.tokEmbd;

        // Read MoE config from GGUF metadata (arch prefix = e.g. "qwen3moe")
        const archName = gguf.getKeyValue('general.architecture', 'qwen3');
        const nExperts     = Number(gguf.getKeyValue(`${archName}.expert_count`))      || 128;
        const nExpertsUsed = Number(gguf.getKeyValue(`${archName}.expert_used_count`)) || 8;

        this.layers = new Array(this.nLayers);
        this.isMoE = false;

        for (let i = 0; i < this.nLayers; i++) {
            const p = `blk.${i}`;
            // MoE layers have ffn_gate_inp (router) instead of ffn_gate
            const routerWeight = gguf.getTensorMetaByName(`${p}.ffn_gate_inp.weight`);
            const isMoELayer = routerWeight != null;
            if (isMoELayer) this.isMoE = true;

            this.layers[i] = {
                attnNorm: gguf.getTensorMetaByName(`${p}.attn_norm.weight`),
                wq:       gguf.getTensorMetaByName(`${p}.attn_q.weight`),
                wk:       gguf.getTensorMetaByName(`${p}.attn_k.weight`),
                wv:       gguf.getTensorMetaByName(`${p}.attn_v.weight`),
                wo:       gguf.getTensorMetaByName(`${p}.attn_output.weight`),
                qNorm:    gguf.getTensorMetaByName(`${p}.attn_q_norm.weight`),
                kNorm:    gguf.getTensorMetaByName(`${p}.attn_k_norm.weight`),
                ffnNorm:  gguf.getTensorMetaByName(`${p}.ffn_norm.weight`),
                // Dense FFN weights
                ffnGate:  gguf.getTensorMetaByName(`${p}.ffn_gate.weight`),
                ffnUp:    gguf.getTensorMetaByName(`${p}.ffn_up.weight`),
                ffnDown:  gguf.getTensorMetaByName(`${p}.ffn_down.weight`),
                // MoE weights (null for dense layers)
                isMoE: isMoELayer,
                routerWeight,
                ffnGateExps: gguf.getTensorMetaByName(`${p}.ffn_gate_exps.weight`),
                ffnUpExps:   gguf.getTensorMetaByName(`${p}.ffn_up_exps.weight`),
                ffnDownExps: gguf.getTensorMetaByName(`${p}.ffn_down_exps.weight`),
                nExperts,
                nExpertsUsed,
            };
        }
    }

    _allocBuffers() {
        const wqShape = this.layers[0].wq.shape;
        const wkShape = this.layers[0].wk.shape;
        const woShape = this.layers[0].wo.shape;

        this.nQ   = wqShape[1];
        this.nKV  = wkShape[1];
        this.headDimQ = this.nQ / this.nHeads;
        this.headDimKV = this.nKV / this.nHeadKV;
        const nEmbActual = woShape[1];

        // For MoE: nFF is the expert hidden dim (from gate_exps), for dense: from ffnGate
        let nFFActual;
        const l0 = this.layers[0];
        if (l0.isMoE) {
            nFFActual = l0.ffnGateExps.shape[1];  // expert hidden dim
        } else {
            nFFActual = l0.ffnGate.shape[1];
        }
        this.nFF = nFFActual;

        this.bufHidden     = new Float32Array(nEmbActual);
        this.bufResidual   = new Float32Array(nEmbActual);
        this.bufNormed     = new Float32Array(nEmbActual);
        this.bufQ          = new Float32Array(this.nQ);
        this.bufK          = new Float32Array(this.nKV);
        this.bufV          = new Float32Array(this.nKV);
        this.bufAttnOut    = new Float32Array(this.nQ);
        this.bufFFN        = new Float32Array(nFFActual);
        this.bufFFNTmp     = new Float32Array(nFFActual);
        this.bufMoEOut     = new Float32Array(nEmbActual);
        this.bufLogits     = new Float32Array(this.nVocab);
        this.bufAttnScores = new Float32Array(this.nHeads * this.maxCtx);
    }

    // --- Single forward pass for one token ---
    // When skipLogits=true (prefill), we don't compute the expensive logits step
    forward(tokenId, position, skipLogits = false) {
        const emb = embeddingLookupGeneric(this.tokEmbd, tokenId);
        if (position === 0) {
            let sum = 0; for (let i = 0; i < Math.min(16, emb.length); i++) sum += Math.abs(emb[i]);
            console.log(`[EMB] token=${tokenId}, first16 avgAbs=${(sum/16).toFixed(4)}, max=${Math.max(...Array.from(emb.slice(0,128)).map(Math.abs)).toFixed(4)}`);
        }
        this.bufHidden.set(emb);
        this.bufResidual.set(emb);

        for (let l = 0; l < this.nLayers; l++) {
            const layer = this.layers[l];

            // Save pre-norm hidden as residual base
            this.bufNormed.set(this.bufHidden);

            if (position === 0 && l === 0) {
                let m = 0; for (let i = 0; i < this.bufNormed.length; i++) { const a = Math.abs(this.bufNormed[i]); if (a > m) m = a; }
                console.log(`[L0] pre-norm maxAbs=${m.toFixed(4)}, hasNaN=${this.bufNormed.some(v => v !== v)}`);
                
                // Check RMSNorm weight tensor type and values
                const wm = layer.attnNorm;
                const wF32 = new Float32Array(wm.buffer, wm.offset, this.nEmb);
                let wmMax = 0; for (let i = 0; i < Math.min(16, this.nEmb); i++) { const a = Math.abs(wF32[i]); if (a > wmMax) wmMax = a; }
                console.log(`[L0] attnNorm type=${wm.type}, shape=${JSON.stringify(wm.shape)}, offset=${wm.offset}, first16 maxAbs=${wmMax.toFixed(4)}, hasNaN=${wF32.some(v => v !== v)}, w[0..5]=${wF32.slice(0,6).map(v=>v.toFixed(4)).join(',')}`);
            }

            // Attention norm
            rmsnorm(this.bufNormed, layer.attnNorm, this.eps);

            if (position === 0 && l === 0) {
                let m = 0; for (let i = 0; i < this.bufNormed.length; i++) { const a = Math.abs(this.bufNormed[i]); if (a > m) m = a; }
                console.log(`[L0] post-norm maxAbs=${m.toFixed(4)}, hasNaN=${this.bufNormed.some(v => v !== v)}`);
            }

            // Q / K / V projections
            const qOut = matmulGeneric(layer.wq, this.bufNormed);
            this.bufQ.set(qOut);

            const kOut = matmulGeneric(layer.wk, this.bufNormed);
            this.bufK.set(kOut);

            const vOut = matmulGeneric(layer.wv, this.bufNormed);
            this.bufV.set(vOut);

            if (position === 0 && l === 0) {
                let qm=0, km=0, vm=0;
                for (let i = 0; i < this.bufQ.length; i++) { const a=Math.abs(this.bufQ[i]); if(a>qm) qm=a; }
                for (let i = 0; i < this.bufK.length; i++) { const a=Math.abs(this.bufK[i]); if(a>km) km=a; }
                for (let i = 0; i < this.bufV.length; i++) { const a=Math.abs(this.bufV[i]); if(a>vm) vm=a; }
                console.log(`[L0] Q maxAbs=${qm.toFixed(4)} NaN=${this.bufQ.some(v=>v!==v)}, K maxAbs=${km.toFixed(4)} NaN=${this.bufK.some(v=>v!==v)}, V maxAbs=${vm.toFixed(4)} NaN=${this.bufV.some(v=>v!==v)}`);
            }

            // Per-head RMSNorm on Q and K
            perHeadRMSNorm(this.bufQ, this.nHeads, this.headDimQ, layer.qNorm, this.eps);
            perHeadRMSNorm(this.bufK, this.nHeadKV, this.headDimKV, layer.kNorm, this.eps);

            // RoPE (split-half)
            applyRoPE(this.bufQ, this.nHeads, this.headDimQ, position, this.ropeFreqBase);
            applyRoPE(this.bufK, this.nHeadKV, this.headDimKV, position, this.ropeFreqBase);

            // Cache K/V
            this.kvCache.store(l, position, this.bufK, this.bufV);

            // Self-attention (GQA) → output projection, returns wo @ attnOut
            const attnResult = this._selfAttention(l, position, layer.wo);

            if (position === 0 && l === 0) {
                let m = 0; for (let i = 0; i < attnResult.length; i++) { const a=Math.abs(attnResult[i]); if(a>m) m=a; }
                console.log(`[L0] wo-proj maxAbs=${m.toFixed(4)}, hasNaN=${attnResult.some(v => v !== v)}`);
            }

            // Residual: hidden += attention output (after wo projection)
            addVec(this.bufHidden, attnResult);

            if (position === 0 && l < 3) {
                let m = 0; for (let i = 0; i < this.bufHidden.length; i++) { const a = Math.abs(this.bufHidden[i]); if (a > m) m = a; }
                console.log(`[L${l}] post-attn maxAbs=${m.toFixed(4)}, hasNaN=${this.bufHidden.some(v => v !== v)}`);
            }

            // FFN path — save post-attn as new residual base
            this.bufNormed.set(this.bufHidden);
            rmsnorm(this.bufNormed, layer.ffnNorm, this.eps);

            if (layer.isMoE) {
                this._moEFFN(l, layer);
            } else {
                // Dense SwiGLU: silu(x @ gate) * (x @ up) @ down
                const gateOut = matmulGeneric(layer.ffnGate, this.bufNormed);
                this.bufFFN.set(gateOut);
                siluInPlace(this.bufFFN);
                const upOut = matmulGeneric(layer.ffnUp, this.bufNormed);
                this.bufFFNTmp.set(upOut);
                mulInPlace(this.bufFFN, this.bufFFNTmp);
                const downOut = matmulGeneric(layer.ffnDown, this.bufFFN);
                addVec(this.bufHidden, downOut);
            }

            if (position === 0 && l < 3) {
                let m = 0; for (let i = 0; i < this.bufHidden.length; i++) { const a = Math.abs(this.bufHidden[i]); if (a > m) m = a; }
                console.log(`[L${l}] post-ffn maxAbs=${m.toFixed(4)}, hasNaN=${this.bufHidden.some(v => v !== v)}`);
            }
        }

        // Output norm + lm_head → logits
        if (!skipLogits) {
            rmsnorm(this.bufHidden, this.outputNorm, this.eps);
            
            // Debug: check hidden state health before logits
            let hSum = 0, hMax = 0;
            for (let i = 0; i < this.bufHidden.length; i++) {
                const a = Math.abs(this.bufHidden[i]);
                if (a > hMax) hMax = a;
            }
            console.log(`[HIDDEN] pos=${position}, maxAbs=${hMax.toFixed(4)}, hasNaN=${this.bufHidden.some(v => v !== v)}`);
            
            const logitsResult = matmulGeneric(this.output, this.bufHidden);
            this.bufLogits.set(logitsResult);
            
            // Debug: top-5 logits
            let top5 = [];
            for (let i = 0; i < Math.min(1000, this.nVocab); i++) {
                const v = this.bufLogits[i];
                if (v !== v) continue; // skip NaN
                top5.push([i, v]);
            }
            top5.sort((a, b) => b[1] - a[1]);
            console.log(`[LOGITS] pos=${position}, top5=${JSON.stringify(top5.slice(0, 5))}`);
        }

        return skipLogits ? null : this.bufLogits;
    }

    // --- Self-attention core (GQA): QK^T + softmax + weighted V sum → bufAttnOut ---
    // Does NOT apply the wo output projection. Call _selfAttention for the full pipeline.
    _selfAttentionCore(layerIdx, position) {
        const nHeads = this.nHeads;
        const nHeadKV = this.nHeadKV;
        const headDimQ = this.headDimQ;
        const headDimKV = this.headDimKV;
        const scale = 1.0 / Math.sqrt(headDimQ);
        const seqLen = position + 1;
        const scores = this.bufAttnScores;

        // QK^T for each head
        for (let h = 0; h < nHeads; h++) {
            const kvHead = (h * nHeadKV) / nHeads | 0;
            const qOff = h * headDimQ;
            const scoreOff = h * seqLen;

            for (let p = 0; p < seqLen; p++) {
                let dot = 0;
                const kSlice = this.kvCache.getK(layerIdx, kvHead, p, p + 1);
                for (let d = 0; d < headDimKV; d++) {
                    dot += this.bufQ[qOff + d] * kSlice[d];
                }
                scores[scoreOff + p] = dot * scale;
            }

            softmaxInPlace(scores, scoreOff, scoreOff + seqLen);
        }

        // Weighted sum of V → bufAttnOut
        this.bufAttnOut.fill(0);
        for (let h = 0; h < nHeads; h++) {
            const kvHead = (h * nHeadKV) / nHeads | 0;
            const outOff = h * headDimQ;
            const scoreOff = h * seqLen;

            for (let p = 0; p < seqLen; p++) {
                const w = scores[scoreOff + p];
                if (w === 0) continue;
                const vSlice = this.kvCache.getV(layerIdx, kvHead, p, p + 1);
                for (let d = 0; d < headDimKV; d++) {
                    this.bufAttnOut[outOff + d] += vSlice[d] * w;
                }
            }
        }
        return this.bufAttnOut;
    }

    // --- Self-attention with GQA (includes wo output projection) ---
    _selfAttention(layer, position, woMeta) {
        this._selfAttentionCore(layer, position);
        // Output projection: wo @ bufAttnOut → n_embd result
        const projResult = matmulGeneric(woMeta, this.bufAttnOut);
        return projResult;
    }

    // --- MoE FFN: router on CPU, expert matmuls on CPU (GPU override in Qwen3GPUEngine) ---
    _moEFFN(layerIdx, layer) {
        // Router: (2048 × 128) F32 matmul on CPU
        const routerLogits = matmulGeneric(layer.routerWeight, this.bufNormed);
        softmaxInPlace(routerLogits, 0, routerLogits.length);

        // Select top-K experts
        const topK = layer.nExpertsUsed;
        const topIndices = [], topWeights = [];
        const sorted = Array.from(routerLogits).map((v, i) => [v, i]);
        sorted.sort((a, b) => b[0] - a[0]);
        for (let k = 0; k < topK; k++) {
            topIndices.push(sorted[k][1]);
            topWeights.push(sorted[k][0]);
        }
        // Re-normalize weights
        let sumW = 0;
        for (const w of topWeights) sumW += w;
        const invSum = sumW > 0 ? 1 / sumW : 0;

        // Notify visualization callbacks
        if (this.onRouterUpdate) {
            this.onRouterUpdate({
                layer: layerIdx,
                probs: Array.from(routerLogits),
                selected: topIndices,
                weights: topWeights.map(w => w * invSum),
            });
        }

        // Accumulate expert outputs
        this.bufMoEOut.fill(0);

        for (let k = 0; k < topK; k++) {
            const expertIdx = topIndices[k];
            // Apply expert mask if set
            if (this.expertMask && this.expertMask.has(expertIdx)) continue;
            const weight = topWeights[k] * invSum;

            // Extract single expert's weights as slice metadata
            const gateSlice = this._expertSlice(layer.ffnGateExps, expertIdx);
            const upSlice   = this._expertSlice(layer.ffnUpExps, expertIdx);
            const downSlice = this._expertSlice(layer.ffnDownExps, expertIdx);

            // gate = silu(normed @ gate_exps[:,expert]) * (normed @ up_exps[:,expert])
            const gateOut = matmulGeneric(gateSlice, this.bufNormed);
            siluInPlace(gateOut);
            const upOut = matmulGeneric(upSlice, this.bufNormed);
            mulInPlace(gateOut, upOut);

            // down = gateOut @ down_exps[:,expert]
            const downOut = matmulGeneric(downSlice, gateOut);

            // Weighted accumulate
            for (let i = 0; i < downOut.length; i++) {
                this.bufMoEOut[i] += weight * downOut[i];
            }
        }

        addVec(this.bufHidden, this.bufMoEOut);
    }

    // Return metadata for a single expert's weight slice from a 3D tensor [inDim, hidDim, nExperts]
    _expertSlice(meta, expertIdx) {
        const inDim = meta.shape[0];
        const hidDim = meta.shape[1];
        const blockSize = this._blockSizeForType(meta.type);
        const elemsPerBlock = meta.type >= 10 ? 256 : 32;  // k-quants=256, q8_0=32
        const blocksPerCol = Math.ceil(inDim / elemsPerBlock);
        const bytesPerCol = blocksPerCol * blockSize;
        const expertOffset = expertIdx * hidDim * bytesPerCol;
        return {
            buffer: meta.buffer,
            offset: meta.offset + expertOffset,
            nbytes: hidDim * bytesPerCol,
            shape: [inDim, hidDim],
            type: meta.type,
        };
    }

    _blockSizeForType(type) {
        switch (type) {
            case 0: return 4;   // F32 bytes per element
            case 8: return 34;  // Q8_0
            case 10: return 84;  // Q2_K
            case 11: return 110; // Q3_K
            case 12: return 144; // Q4_K
            case 13: return 176; // Q5_K
            case 14: return 210; // Q6_K
            default: return 34;
        }
    }

    // --- Generation loop ---
    async generate(tokenIds, options = {}) {
        const {
            maxSteps     = 512,
            temperature  = 0.7,
            topP         = 0.9,
            topK         = 40,
            thinkingMode = 'suppress',
            onToken,
            onFinish,
            abortSignal,
        } = options;

        // Qwen3: <think>=151667, </think>=151668
        const thinkOpts = {
            mode:         thinkingMode,
            thinkId:      151667,
            endThinkId:   151668,
            thinkingDone: thinkingMode === 'suppress',
        };

        this.kvCache.reset();

        // Process prompt tokens (prefill) — skip logits for all but the last
        for (let i = 0; i < tokenIds.length - 1; i++) {
            if (abortSignal?.aborted) return;
            this.forward(tokenIds[i], i, true /*skipLogits*/);
            if (i % 16 === 0 && tokenIds.length > 32) {
                await yieldToBrowser();
            }
        }

        // Process last prompt token with logits to sample first generated token
        if (abortSignal?.aborted) return;
        this.forward(tokenIds[tokenIds.length - 1], tokenIds.length - 1, false /*compute logits*/);

        let currentToken = sampleWithThinkControl(this.bufLogits, temperature, topP, topK, thinkOpts);

        const generatedTokens = [];

        for (let step = 0; step < maxSteps; step++) {
            if (currentToken === this.tokenizer.eosTokenId) break;

            // 特殊終了トークン（im_end）で生成停止
            if (this.tokenizer.stopTokenIds.has(currentToken)) break;

            if (currentToken === thinkOpts.endThinkId) thinkOpts.thinkingDone = true;

            // Forward pass for this generated token at position N + step
            if (abortSignal?.aborted) return;
            this.forward(currentToken, tokenIds.length + step);

            generatedTokens.push(currentToken);
            if (onToken) onToken(currentToken);

            currentToken = sampleWithThinkControl(this.bufLogits, temperature, topP, topK, thinkOpts);

            await yieldToBrowser();
        }

        if (onFinish) onFinish(generatedTokens);
    }

    // --- Chat template formatting ---
    formatChat(messages, systemPrompt, thinkingMode = 'suppress') {
        return this._defaultQwen3Format(messages, systemPrompt, thinkingMode);
        /*
        const chatTemplate = this.gguf.getKeyValue('tokenizer.chat_template');
        if (!chatTemplate) return this._defaultQwen3Format(messages, systemPrompt);

        // Try to use the GGUF chat template (basic Jinja2-like processing)
        return this._applyChatTemplate(chatTemplate, messages, systemPrompt);
        */
    }

    _defaultQwen3Format(messages, systemPrompt, thinkingMode = 'suppress') {
        let text = '';
        if (systemPrompt) {
            text += `<|im_start|>system\n${systemPrompt}<|im_end|>\n`;
        }
        for (const msg of messages) {
            text += `<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n`;
        }
        text += '<|im_start|>assistant\n';
        if (thinkingMode === 'suppress') {
            text += '<think></think>\n';
        } else {
            // shorten / full: open thinking block, model generates inside it
            text += '<think>\n';
        }
        return text;
    }

    _applyChatTemplate(template, messages, systemPrompt) {
        // Minimal Jinja2-like template processor for common patterns
        let t = template;

        // Replace loop constructs with expanded content
        const buildMessagesText = () => {
            let result = '';
            if (systemPrompt) {
                result += `<|im_start|>system\n${systemPrompt}<|im_end|>\n`;
            }
            for (const msg of messages) {
                result += `<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n`;
            }
            result += '<|im_start|>assistant\n';
            return result;
        };

        // Handle {{ messages }} or similar placeholders
        t = t.replace(/\{\{[\s]*messages[\s]*\}\}/g, buildMessagesText());
        t = t.replace(/\{\{[\s]*message\.content[\s]*\}\}/g, (m, offset, str) => {
            return messages[messages.length - 1]?.content || '';
        });
        t = t.replace(/\{\{[\s]*message\.role[\s]*\}\}/g, (m, offset, str) => {
            return messages[messages.length - 1]?.role || '';
        });

        // Remove remaining Jinja2 constructs that we can't process
        t = t.replace(/\{%[^%]*%\}/g, '');
        t = t.replace(/\{\{[^}]*\}\}/g, '');

        return t;
    }
}

function yieldToBrowser() {
    return new Promise(r => setTimeout(r, 0));
}
