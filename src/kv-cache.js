class KVCache {
    constructor(nLayers, nHeadKV, maxCtx, headDim) {
        this.nLayers = nLayers;
        this.nHeadKV = nHeadKV;
        this.maxCtx = maxCtx;
        this.headDim = headDim;

        // Flat layout: [layer][head][position][dim]
        const layerSize = nHeadKV * maxCtx * headDim;
        this.kCache = new Float32Array(nLayers * layerSize);
        this.vCache = new Float32Array(nLayers * layerSize);
        this.nPast = 0;
    }

    // Store K/V for a single token at given position
    store(layer, position, kData, vData) {
        const headDim = this.headDim;
        const nHeadKV = this.nHeadKV;
        const maxCtx = this.maxCtx;
        const layerOffK = layer * nHeadKV * maxCtx * headDim;
        const layerOffV = layer * nHeadKV * maxCtx * headDim;

        for (let h = 0; h < nHeadKV; h++) {
            const posOffK = layerOffK + h * maxCtx * headDim + position * headDim;
            const posOffV = layerOffV + h * maxCtx * headDim + position * headDim;
            const srcOff = h * headDim;
            for (let d = 0; d < headDim; d++) {
                this.kCache[posOffK + d] = kData[srcOff + d];
                this.vCache[posOffV + d] = vData[srcOff + d];
            }
        }
    }

    // Get K slice for a given layer, KV head, and position range → view into flat array
    getK(layer, kvHead, start, end) {
        const base = layer * this.nHeadKV * this.maxCtx * this.headDim
                   + kvHead * this.maxCtx * this.headDim
                   + start * this.headDim;
        return new Float32Array(this.kCache.buffer, this.kCache.byteOffset + base * 4, (end - start) * this.headDim);
    }

    // Get V slice similarly
    getV(layer, kvHead, start, end) {
        const base = layer * this.nHeadKV * this.maxCtx * this.headDim
                   + kvHead * this.maxCtx * this.headDim
                   + start * this.headDim;
        return new Float32Array(this.vCache.buffer, this.vCache.byteOffset + base * 4, (end - start) * this.headDim);
    }

    reset() {
        this.nPast = 0;
    }
}
