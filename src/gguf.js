class GGUFReader {
    constructor(buffer) {
        this.buffer = buffer;
        this._view32 = new DataView(buffer);
        this._viewU8 = new Uint8Array(buffer);
        this._tensors = [];
        this._kvs = []; // cached KV positions
        this._kvCount = 0;
        this._tensorCount = 0;
        this._parseHeader();
    }

    // --- GGUF binary format constants ---
    static SIGNATURE = 'GGUF';
    static VERSION = 3;

    // KV types (uint32) — gguf.h GGUF_TYPE enum
    static KV_TYPES = {
        UINT8: 0, INT8: 1, UINT16: 2, INT16: 3,
        UINT32: 4, INT32: 5, FLOAT32: 6, BOOL: 7,
        STRING: 8, ARRAY: 9, UINT64: 10, INT64: 11, FLOAT64: 12
    };

    // Tensor types (uint32)
    static TENSOR_TYPES = {
        F32: 0, F16: 1, Q4_0: 2, Q4_1: 3,
        Q5_0: 6, Q5_1: 7, Q8_0: 8, Q8_1: 9,
        Q2_K: 10, Q3_K: 11, Q4_K: 12, Q5_K: 13,
        Q6_K: 14, IQ2_XXS: 15, IQ2_XS: 16, IQ3_XXS: 17,
        IQ1_S: 18, IQ4_NL: 19, IQ3_S: 20, IQ2_S: 21,
        IQ4_XS: 22, IQ1_M: 23, I8: 24, I16: 25, I32: 26,
        I64: 27, F64: 28, IQ1_LL: 29, BIQS: 30,
        Q4_0_4_4: 31, Q4_0_4_8: 32, Q4_0_4_4: 33
    };

    // --- Header parsing ---
    _parseHeader() {
        const sig = this._readString(0, 4);
        if (sig !== GGUFReader.SIGNATURE) {
            throw new Error(`Invalid signature: ${sig}`);
        }
        this.version = this._view32.getUint32(4, true);
        if (this.version !== GGUFReader.VERSION) {
            console.warn(`GGUF version ${this.version} may not be fully supported`);
        }
        this._tensorCount = Number(this._readU64(8));
        this._kvCount = Number(this._readU64(16));

        // Parse KV pairs first to get metadata — cache positions for O(1) access
        let offset = 24; // GGUF v3 header: magic(4) + version(4) + n_tensors(8) + n_kv(8) = 24 bytes
        for (let i = 0; i < this._kvCount; i++) {
            const keyLen = Number(this._readU64(offset));
            const keyOff = offset + 8;
            const typeOff = keyOff + keyLen;
            const valOff = typeOff + 4;

            // Read key now (needed for findKey)
            const key = this._readString(keyOff, keyLen);
            const type = this._view32.getUint32(typeOff, true);

            this._kvs.push({ key, type, valueOffset: valOff });
            offset = this._skipValue(type, valOff);
        }

        // Parse tensor metadata — data_offset is relative to tensor_data section start
        for (let i = 0; i < this._tensorCount; i++) {
            const nameLen = Number(this._readU64(offset)); offset += 8;
            const name = this._readString(offset, nameLen); offset += nameLen;
            const nDims = this._view32.getUint32(offset, true); offset += 4;
            const shape = [];
            for (let d = 0; d < nDims; d++) {
                shape.push(Number(this._readU64(offset))); offset += 8;
            }
            const type = this._view32.getUint32(offset, true); offset += 4;
            const dataOffsetRel = Number(this._readU64(offset)); offset += 8;

            // Calculate padding (GGUF tensors are padded to alignment boundary)
            const nbytes = this._tensorNbytes(shape, type);
            const paddedNbytes = Math.ceil(nbytes / 256) * 256;

            this._tensors.push({ name, shape, type, dataOffsetRel, nbytes: paddedNbytes });
        }

        // tensor_data section starts after header, padded to general.alignment boundary
        const ALIGNMENT = Number(this.getKeyValue('general.alignment')) || 32;
        const padding = (ALIGNMENT - offset % ALIGNMENT) % ALIGNMENT;
        const tensorDataStart = offset + padding;

        // Convert relative offsets to absolute file offsets
        for (let i = 0; i < this._tensors.length; i++) {
            this._tensors[i].dataOffset = tensorDataStart + this._tensors[i].dataOffsetRel;
            delete this._tensors[i].dataOffsetRel;
        }
    }

    _skipValue(type, offset) {
        switch (type) {
            case GGUFReader.KV_TYPES.UINT8:
            case GGUFReader.KV_TYPES.INT8:
            case GGUFReader.KV_TYPES.BOOL:
                return offset + 1;
            case GGUFReader.KV_TYPES.UINT16:
            case GGUFReader.KV_TYPES.INT16:
                return offset + 2;
            case GGUFReader.KV_TYPES.UINT32:
            case GGUFReader.KV_TYPES.INT32:
            case GGUFReader.KV_TYPES.FLOAT32:
                return offset + 4;
            case GGUFReader.KV_TYPES.UINT64:
            case GGUFReader.KV_TYPES.INT64:
            case GGUFReader.KV_TYPES.FLOAT64:
                return offset + 8;
            case GGUFReader.KV_TYPES.STRING: {
                const len = Number(this._readU64(offset));
                // u64 length(8) + raw bytes, no padding
                return offset + 8 + len;
            }
            case GGUFReader.KV_TYPES.ARRAY: {
                const arrType = this._view32.getUint32(offset, true);
                const arrLen = Number(this._readU64(offset + 4));
                let valOffset = offset + 12;
                for (let i = 0; i < arrLen; i++) {
                    valOffset = this._skipValue(arrType, valOffset);
                }
                return valOffset;
            }
            default:
                throw new Error(`Unknown KV type: ${type}`);
        }
    }

    // --- Low-level reads ---
    _readString(offset, len) {
        if (!this._decoder) this._decoder = new TextDecoder('utf8');
        const end = offset + len;
        if (end > this.buffer.byteLength) {
            throw new Error(`Out of bounds string read: offset=${offset}, len=${len}, buffer size=${this.buffer.byteLength}`);
        }
        const slice = new Uint8Array(this.buffer, offset, len);
        return this._decoder.decode(slice);
    }

    _readU64(offset) {
        const lo = this._view32.getUint32(offset, true);
        const hi = this._view32.getUint32(offset + 4, true);
        // Use BigInt for safety with large offsets
        return (BigInt(hi) << 32n) | BigInt(lo);
    }

    _readI64(offset) {
        const lo = this._view32.getInt32(offset, true);
        const hi = this._view32.getInt32(offset + 4, true);
        return (BigInt(hi) << 32n) | (lo & 0xFFFFFFFFn);
    }

    // --- Public API: Header info ---
    getHeader() {
        return {
            signature: GGUFReader.SIGNATURE,
            version: this.version,
            tensorCount: this._tensorCount,
            kvCount: this._kvCount,
            fileSize: this.buffer.byteLength
        };
    }

    // --- Public API: KV metadata ---
    getKVCount() { return this._kvCount; }

    _getKVAt(index) {
        return this._kvs[index];
    }

    getKVKey(index) { return this._getKVAt(index).key; }
    getKVType(index) { return this._getKVAt(index).type; }

    _readValue(type, offset) {
        switch (type) {
            case GGUFReader.KV_TYPES.UINT8: return this._viewU8[offset];
            case GGUFReader.KV_TYPES.INT8: return this._view32.getInt8(offset);
            case GGUFReader.KV_TYPES.UINT16: return this._view32.getUint16(offset, true);
            case GGUFReader.KV_TYPES.INT16: return this._view32.getInt16(offset, true);
            case GGUFReader.KV_TYPES.UINT32: return this._view32.getUint32(offset, true);
            case GGUFReader.KV_TYPES.INT32: return this._view32.getInt32(offset, true);
            case GGUFReader.KV_TYPES.FLOAT32: return this._view32.getFloat32(offset, true);
            case GGUFReader.KV_TYPES.BOOL: return this._viewU8[offset] !== 0;
            case GGUFReader.KV_TYPES.UINT64: return Number(this._readU64(offset));
            case GGUFReader.KV_TYPES.INT64: return Number(this._readI64(offset));
            case GGUFReader.KV_TYPES.FLOAT64: return this._view32.getFloat64(offset, true);
            case GGUFReader.KV_TYPES.STRING: {
                const len = Number(this._readU64(offset));
                return this._readString(offset + 8, len);
            }
            case GGUFReader.KV_TYPES.ARRAY: {
                const arrType = this._view32.getUint32(offset, true);
                const arrLen = Number(this._readU64(offset + 4));
                let valOff = offset + 12;
                const result = [];
                for (let i = 0; i < arrLen; i++) {
                    result.push(this._readValue(arrType, valOff));
                    valOff = this._skipValue(arrType, valOff);
                }
                return result;
            }
            default: throw new Error(`Unsupported KV type: ${type}`);
        }
    }

    getKVValue(index) {
        const kv = this._getKVAt(index);
        return this._readValue(kv.type, kv.valueOffset);
    }

    // --- Key search helpers ---
    findKey(keyName) {
        for (let i = 0; i < this._kvCount; i++) {
            if (this.getKVKey(i) === keyName) return i;
        }
        return -1;
    }

    getKeyValue(keyName, defaultValue) {
        const idx = this.findKey(keyName);
        if (idx === -1) return defaultValue;
        return this.getKVValue(idx);
    }

    // --- Public API: Tensor info ---
    getTensorCount() { return this._tensorCount; }
    getTensorName(index) { return this._tensors[index].name; }
    getTensorShape(index) { return this._tensors[index].shape; }
    getTensorType(index) { return this._tensors[index].type; }

    // Returns tensor metadata WITHOUT copying data
    getTensorMeta(index) {
        const t = this._tensors[index];
        if (t.tensorBuffer) {
            // Loaded via fromFile(): each tensor has its own small buffer, offset is 0
            return {
                buffer: t.tensorBuffer,
                offset: 0,
                nbytes: t.nbytes,
                shape: t.shape,
                type: t.type
            };
        }
        return {
            buffer: this.buffer,
            offset: t.dataOffset,
            nbytes: t.nbytes,
            shape: t.shape,
            type: t.type
        };
    }

    getTensorMetaByName(name) {
        for (let i = 0; i < this._tensors.length; i++) {
            if (this._tensors[i].name === name) return this.getTensorMeta(i);
        }
        return null;
    }

    // --- Architecture parameters helper ---
    getArch() {
        const arch = this.getKeyValue('general.architecture', 'unknown');
        const prefix = arch + '.';

        // head_count_kv may be a scalar or per-layer array (LFM2 stores 0 for recurrent layers)
        const headCountKVRaw = this.getKeyValue(prefix + 'attention.head_count_kv');
        const headCountKV = Array.isArray(headCountKVRaw)
            ? Math.max(...headCountKVRaw.filter(v => v > 0))
            : Number(headCountKVRaw);

        return {
            architecture: arch,
            vocabSize:           Number(this.getKeyValue(prefix + 'vocab_size')),
            contextLength:       Number(this.getKeyValue(prefix + 'context_length')),
            embeddingLength:     Number(this.getKeyValue(prefix + 'embedding_length')),
            blockCount:          Number(this.getKeyValue(prefix + 'block_count')),
            feedForwardLength:   Number(this.getKeyValue(prefix + 'feed_forward_length')),
            headCount:           Number(this.getKeyValue(prefix + 'attention.head_count')),
            headCountKV,
            layerNormRmsEps:     this.getKeyValue(prefix + 'attention.layer_norm_rms_epsilon'),
            ropeFreqBase:        Number(this.getKeyValue(prefix + 'rope.freq_base') || this.getKeyValue(prefix + 'rope_freq_base')),
            // LFM2-specific fields (undefined for other architectures)
            expertFeedForwardLength: Number(this.getKeyValue(prefix + 'expert_feed_forward_length')) || 0,
            leadingDenseBlockCount:  Number(this.getKeyValue(prefix + 'leading_dense_block_count'))  || 0,
            expertGatingFunc:        Number(this.getKeyValue(prefix + 'expert_gating_func'))          || 0,
            shortconvLCache:         Number(this.getKeyValue(prefix + 'shortconv.l_cache'))           || 0,
        };
    }

    // --- Tensor type name lookup ---
    static tensorTypeName(type) {
        for (const [name, val] of Object.entries(GGUFReader.TENSOR_TYPES)) {
            if (val === type) return name;
        }
        return `UNKNOWN(${type})`;
    }

    // --- Calculate tensor byte size from shape + type ---
    _tensorNbytes(shape, type) {
        const totalElems = shape.reduce((a, b) => a * b, 1);
        switch (type) {
            case GGUFReader.TENSOR_TYPES.F32: return totalElems * 4;
            case GGUFReader.TENSOR_TYPES.F16: return totalElems * 2;
            case GGUFReader.TENSOR_TYPES.Q8_0:
                // block_q8_0: ggml_half d(2B) + qs[32](32B) = 34 bytes per 32 elements
                return Math.ceil(totalElems / 32) * 34;
            case GGUFReader.TENSOR_TYPES.Q4_0:
                // block_q4_0: d(2) + qs[32] = 34 bytes per 32 elements
                return Math.ceil(totalElems / 32) * 34;
            case GGUFReader.TENSOR_TYPES.Q4_1:
                // block_q4_1: d(2) + dmin(2) + qs[32] = 36 bytes per 32 elements
                return Math.ceil(totalElems / 32) * 36;
            case GGUFReader.TENSOR_TYPES.Q5_0:
                // block_q5_0: d(2) + qs[32] + qh[4] = 42 bytes per 32 elements
                return Math.ceil(totalElems / 32) * 42;
            case GGUFReader.TENSOR_TYPES.Q5_1:
                // block_q5_1: d(2) + dmin(2) + qs[32] + qh[4] = 44 bytes per 32 elements
                return Math.ceil(totalElems / 32) * 44;
            case GGUFReader.TENSOR_TYPES.Q2_K:
                // d(2)+dmin(2)+scales[16]+qs[64] = 84 bytes per 256 elements
                return Math.ceil(totalElems / 256) * 84;
            case GGUFReader.TENSOR_TYPES.Q3_K:
                // hmask[32]+qs[64]+scales[12]+d(2) = 110 bytes per 256 elements
                return Math.ceil(totalElems / 256) * 110;
            case GGUFReader.TENSOR_TYPES.Q4_K:
                // d(2)+dmin(2)+scales[12]+qs[128] = 144 bytes per 256 elements
                return Math.ceil(totalElems / 256) * 144;
            case GGUFReader.TENSOR_TYPES.Q5_K:
                // d(2)+dmin(2)+scales[12]+qh[32]+qs[128] = 176 bytes per 256 elements
                return Math.ceil(totalElems / 256) * 176;
            case GGUFReader.TENSOR_TYPES.Q6_K:
                // ql[128]+qh[64]+scales[16]+d(2) = 210 bytes per 256 elements
                return Math.ceil(totalElems / 256) * 210;
            default:
                console.warn(`Unknown tensor type ${type}, assuming F32`);
                return totalElems * 4;
        }
    }

    // --- Q8_0 block constants for on-the-fly decode ---
    static QK_Q8_0 = 32;
    static BLOCK_SIZE_Q8_0 = 34; // sizeof(block_q8_0) = ggml_half d(2B) + qs[32](32B)

    // --- Factory: load from File object (avoids Chrome ~2GB ArrayBuffer limit) ---
    // Reads only the GGUF header section (always <10MB) to parse metadata,
    // then loads each tensor's data as individual slices.
    static async fromFile(file, onProgress) {
        // 64MB is a very safe upper bound for GGUF header+metadata (typically <5MB)
        const HEADER_MAX = 64 * 1024 * 1024;
        const headerSlice = file.slice(0, Math.min(HEADER_MAX, file.size));
        const headerBuf = await headerSlice.arrayBuffer();

        // Parse header: this reads KV pairs and tensor metadata (all in header section)
        // this.buffer = headerBuf, which is fine — _parseHeader() only touches header bytes
        const reader = new GGUFReader(headerBuf);
        reader._fileSize = file.size;

        // Load each tensor's data as a separate slice — each is small enough to allocate
        const n = reader._tensors.length;
        let bytesLoaded = 0;
        for (let i = 0; i < n; i++) {
            const t = reader._tensors[i];
            const end = Math.min(t.dataOffset + t.nbytes, file.size);
            const slice = file.slice(t.dataOffset, end);
            t.tensorBuffer = await slice.arrayBuffer();
            bytesLoaded += t.tensorBuffer.byteLength;
            if (onProgress) onProgress(bytesLoaded / file.size);
        }

        return reader;
    }
}
