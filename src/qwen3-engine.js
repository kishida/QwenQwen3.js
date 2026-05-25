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
        
        // Derive nFF from ffn gate weight shape[1]
        this.nFF = this.layers[0].ffnGate.shape[1];

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

        this.layers = new Array(this.nLayers);
        for (let i = 0; i < this.nLayers; i++) {
            const p = `blk.${i}`;
            this.layers[i] = {
                attnNorm: gguf.getTensorMetaByName(`${p}.attn_norm.weight`),
                wq:       gguf.getTensorMetaByName(`${p}.attn_q.weight`),
                wk:       gguf.getTensorMetaByName(`${p}.attn_k.weight`),
                wv:       gguf.getTensorMetaByName(`${p}.attn_v.weight`),
                wo:       gguf.getTensorMetaByName(`${p}.attn_output.weight`),
                qNorm:    gguf.getTensorMetaByName(`${p}.attn_q_norm.weight`),
                kNorm:    gguf.getTensorMetaByName(`${p}.attn_k_norm.weight`),
                ffnNorm:  gguf.getTensorMetaByName(`${p}.ffn_norm.weight`),
                ffnGate:  gguf.getTensorMetaByName(`${p}.ffn_gate.weight`),
                ffnUp:    gguf.getTensorMetaByName(`${p}.ffn_up.weight`),
                ffnDown:  gguf.getTensorMetaByName(`${p}.ffn_down.weight`),
            };
        }
    }

    _allocBuffers() {
        // Derive actual dimensions from weight tensor shapes (row-major: shape=[in, out], shape[1]=output dim)
        const wqShape = this.layers[0].wq.shape;  // [n_embd, nQ]
        const wkShape = this.layers[0].wk.shape;  // [n_embd, nKV]
        const woShape = this.layers[0].wo.shape;  // [nQ, n_embd]
        const ffnGateShape = this.layers[0].ffnGate.shape; // [n_embd, nFF]

        this.nQ   = wqShape[1];
        this.nKV  = wkShape[1];
        this.headDimQ = this.nQ / this.nHeads;
        this.headDimKV = this.nKV / this.nHeadKV;
        const nEmbActual = woShape[1];
        const nFFActual = ffnGateShape[1];

        this.bufHidden  = new Float32Array(nEmbActual);
        this.bufResidual = new Float32Array(nEmbActual);
        this.bufNormed   = new Float32Array(nEmbActual);
        this.bufQ        = new Float32Array(this.nQ);
        this.bufK        = new Float32Array(this.nKV);
        this.bufV        = new Float32Array(this.nKV);
        this.bufAttnOut  = new Float32Array(this.nQ);
        this.bufFFN      = new Float32Array(nFFActual);
        this.bufFFNTmp   = new Float32Array(nFFActual);
        this.bufLogits   = new Float32Array(this.nVocab);

        // Attention score buffer: [nHeads, maxSeqLen] — reused each step
        this.bufAttnScores = new Float32Array(this.nHeads * this.maxCtx);
    }

    // --- Single forward pass for one token ---
    // When skipLogits=true (prefill), we don't compute the expensive logits step
    forward(tokenId, position, skipLogits = false) {
        const emb = embeddingLookupQ80(this.tokEmbd, tokenId);
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
            const qOut = matmulQ80xF32(layer.wq, this.bufNormed);
            this.bufQ.set(qOut);

            const kOut = matmulQ80xF32(layer.wk, this.bufNormed);
            this.bufK.set(kOut);

            const vOut = matmulQ80xF32(layer.wv, this.bufNormed);
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

            // SwiGLU: silu(x @ gate) * (x @ up) @ down
            const gateOut = matmulQ80xF32(layer.ffnGate, this.bufNormed);
            this.bufFFN.set(gateOut);
            siluInPlace(this.bufFFN);

            const upOut = matmulQ80xF32(layer.ffnUp, this.bufNormed);
            this.bufFFNTmp.set(upOut);
            mulInPlace(this.bufFFN, this.bufFFNTmp);

            const downOut = matmulQ80xF32(layer.ffnDown, this.bufFFN);
            addVec(this.bufHidden, downOut);

            if (position === 0 && l < 3) {
                let m = 0; for (let i = 0; i < this.bufHidden.length; i++) { const a = Math.abs(this.bufHidden[i]); if (a > m) m = a; }
                console.log(`[L${l}] post-ffn maxAbs=${m.toFixed(4)}, hasNaN=${this.bufHidden.some(v => v !== v)}`);
            }
        }

        // Output norm + lm_head → logits (skip during prefill for speed)
        if (!skipLogits) {
            rmsnorm(this.bufHidden, this.outputNorm, this.eps);
            
            // Debug: check hidden state health before logits
            let hSum = 0, hMax = 0;
            for (let i = 0; i < this.bufHidden.length; i++) {
                const a = Math.abs(this.bufHidden[i]);
                if (a > hMax) hMax = a;
            }
            console.log(`[HIDDEN] pos=${position}, maxAbs=${hMax.toFixed(4)}, hasNaN=${this.bufHidden.some(v => v !== v)}`);
            
            const logitsResult = logitsFromQ80(this.output, this.bufHidden);
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

    // --- Self-attention with GQA ---
    _selfAttention(layer, position, woMeta) {
        const nHeads = this.nHeads;
        const nHeadKV = this.nHeadKV;
        const headDimQ = this.headDimQ;
        const headDimKV = this.headDimKV;
        // Use Q head dim for scaling (standard practice)
        const scale = 1.0 / Math.sqrt(headDimQ);
        const seqLen = position + 1;
        const scores = this.bufAttnScores;

        // QK^T for each head
        for (let h = 0; h < nHeads; h++) {
            const kvHead = (h * nHeadKV) / nHeads | 0;
            const qOff = h * headDimQ;
            const scoreOff = h * seqLen;

            // Dot with cached K tokens [0..position]
            for (let p = 0; p < seqLen; p++) {
                let dot = 0;
                const kSlice = this.kvCache.getK(layer, kvHead, p, p + 1);
                for (let d = 0; d < headDimKV; d++) {
                    dot += this.bufQ[qOff + d] * kSlice[d];
                }
                scores[scoreOff + p] = dot * scale;
            }

            // Softmax over sequence dimension
            softmaxInPlace(scores, scoreOff, scoreOff + seqLen);
        }

        // Weighted sum of V → bufAttnOut[nHeads * headDimQ]
        this.bufAttnOut.fill(0);
        for (let h = 0; h < nHeads; h++) {
            const kvHead = (h * nHeadKV) / nHeads | 0;
            const outOff = h * headDimQ;
            const scoreOff = h * seqLen;

            for (let p = 0; p < seqLen; p++) {
                const w = scores[scoreOff + p];
                if (w === 0) continue;
                const vSlice = this.kvCache.getV(layer, kvHead, p, p + 1);
                for (let d = 0; d < headDimKV; d++) {
                    this.bufAttnOut[outOff + d] += vSlice[d] * w;
                }
            }
        }

        // Output projection: wo @ bufAttnOut → n_embd result
        const projResult = matmulQ80xF32(woMeta, this.bufAttnOut);
        return projResult;
    }

    // --- Generation loop ---
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

        let currentToken = temperature <= 0
            ? sampleGreedy(this.bufLogits)
            : sampleTopPTopK(this.bufLogits, temperature, topP, topK);

        const generatedTokens = [];

        for (let step = 0; step < maxSteps; step++) {
            if (currentToken === this.tokenizer.eosTokenId) break;

            // 特殊終了トークン（im_end）で生成停止
            if (this.tokenizer.stopTokenIds.has(currentToken)) break;

            // Forward pass for this generated token at position N + step
            if (abortSignal?.aborted) return;
            this.forward(currentToken, tokenIds.length + step);

            generatedTokens.push(currentToken);
            if (onToken) onToken(currentToken);

            currentToken = temperature <= 0
                ? sampleGreedy(this.bufLogits)
                : sampleTopPTopK(this.bufLogits, temperature, topP, topK);

            await yieldToBrowser();
        }

        if (onFinish) onFinish(generatedTokens);
    }

    // --- Chat template formatting ---
    formatChat(messages, systemPrompt) {
        return this._defaultQwen3Format(messages, systemPrompt);
        /*
        const chatTemplate = this.gguf.getKeyValue('tokenizer.chat_template');
        if (!chatTemplate) return this._defaultQwen3Format(messages, systemPrompt);

        // Try to use the GGUF chat template (basic Jinja2-like processing)
        return this._applyChatTemplate(chatTemplate, messages, systemPrompt);
        */
    }

    _defaultQwen3Format(messages, systemPrompt) {
        let text = '';
        if (systemPrompt) {
            text += `<|im_start|>system\n${systemPrompt}<|im_end|>\n`;
        }
        for (const msg of messages) {
            text += `<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n`;
        }
        text += '<|im_start|>assistant\n';
        text += '<think>\n\n</think>\n\n'; // if thinking on, add just '<think>\n'
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
