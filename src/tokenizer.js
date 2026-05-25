// === 特殊トークン定義（必要に応じて差し替え） ===
const SPECIAL_TOKENS = [
    '<|im_start|>',   // role開始マーカー
    '<|im_end|>',     // role終了マーカー
    '<think>',      // thinking開始
    '</think>',     // thinking終了
];

// 正規表現エスケープ済みパターン（pretokenize で使用）
const SPECIAL_TOKEN_PATTERN = SPECIAL_TOKENS.map(t => t.replace(/[|\\{}()[\]^$+*?.]/g, '\\$&')).join('|');

class BPETokenizer {
    constructor() {
        buildGPT2ByteMap();
        this.tokenToId = new Map();
        this.idToToken = [];
        this.bpeRanks = new Map();
        this.cptToToken = new Map();
        this.bosTokenId = -1;
        this.eosTokenId = -1;
        this.isQwen35 = false;

        this.specialTokenIds = new Map();
        this.stopTokenIds = new Set();
    }

    loadFromGGUF(gguf) {
        const arch = gguf.getKeyValue('general.architecture', '');
        this.isQwen35 = arch === 'qwen3';

        const tokensArr = gguf.getKeyValue('tokenizer.ggml.tokens');
        if (!tokensArr || !Array.isArray(tokensArr)) {
            throw new Error('tokenizer.ggml.tokens not found in GGUF');
        }

        const mergesArr = gguf.getKeyValue('tokenizer.ggml.merges');
        if (!mergesArr || !Array.isArray(mergesArr)) {
            throw new Error('tokenizer.ggml.merges not found in GGUF');
        }

        this.bosTokenId = Number(gguf.getKeyValue('tokenizer.ggml.bos_token_id', -1));
        this.eosTokenId = Number(gguf.getKeyValue('tokenizer.ggml.eos_token_id', -1));

        for (let i = 0; i < tokensArr.length; i++) {
            const tok = tokensArr[i];
            this.tokenToId.set(tok, i);
            this.idToToken[i] = tok;
            if (tok.length === 1) {
                this.cptToToken.set(tok.codePointAt(0), i);
            }
        }

        for (let i = 0; i < mergesArr.length; i++) {
            const merge = mergesArr[i];
            let sp = merge.indexOf(' ', 1);
            if (sp === -1) sp = merge.indexOf('\x01', 1);
            if (sp === -1) continue;
            this.bpeRanks.set(merge.substring(0, sp) + '\x00' + merge.substring(sp + 1), i);
        }

        // 特殊トークンのIDをキャッシュ（vocab読み込み後）
        for (const tok of SPECIAL_TOKENS) {
            const id = this.tokenToId.get(tok);
            if (id !== undefined) {
                this.specialTokenIds.set(tok, id);
            } else {
                console.warn(`[Tokenizer] Special token not found in vocab: ${tok}`);
            }
        }

        // 終了判定用：im_end のみで生成停止
        const imEndTok = SPECIAL_TOKENS[1]; // im_end
        const id = this.specialTokenIds.get(imEndTok);
        if (id !== undefined) this.stopTokenIds.add(id);
    }

    tokenize(text, addBos = false, addEos = false) {
        const result = [];
        if (addBos && this.bosTokenId >= 0) result.push(this.bosTokenId);

        const words = pretokenize(text, this.isQwen35);
        for (const word of words) {
            if (!word || !word.length) continue;
            const directId = this.tokenToId.get(word);
            if (directId !== undefined) {
                result.push(directId);
                continue;
            }
            if (word.length > 1024) {
                for (const ch of word) {
                    result.push(...byteFallbackTokens(this.tokenToId, ch));
                }
                continue;
            }
            const toks = this.bpeTokenize(word);
            result.push(...toks);
        }

        if (addEos && this.eosTokenId >= 0) result.push(this.eosTokenId);
        return result;
    }

    bpeTokenize(word) {
        // GPT-2 bytes_to_unicode mapping:
        // Safe bytes (!~ 33-126, ¡¬ 161-172, ®ÿ 174-255) → chr(byte) directly
        // Remaining bytes (0-32, 127-160, 173) → chr(0x100 + offset)
        const encoder = new TextEncoder();
        const symStr = [];
        for (const ch of word) {
            const utf8 = encoder.encode(ch);
            for (let i = 0; i < utf8.length; i++) {
                symStr.push(_gpt2ByteToChar[utf8[i]]);
            }
        }

        const len = symStr.length;
        if (len === 0) return [];

        // Single byte → direct lookup
        if (len === 1) {
            const id = this.tokenToId.get(symStr[0]);
            if (id !== undefined) return [id];
            return [symStr[0]];
        }

        // Linked list of symbols
        const symPrev = new Int32Array(len);
        const symNext = new Int32Array(len);
        for (let i = 0; i < len; i++) {
            symPrev[i] = i - 1;
            symNext[i] = (i + 1 < len) ? i + 1 : -1;
        }

        const pq = [];
        let mergesDone = 0;
        const maxMerges = len - 1;

        for (let i = 0; i < len - 1; i++) {
            addBigram(i, i + 1, symStr, this.bpeRanks, pq);
        }

        while (pq.length > 0 && mergesDone < maxMerges) {
            const [rank, si] = heapPop(pq);
            const sj = symNext[si];

            if (sj < 0 || !symStr[si] || !symStr[sj]) continue;

            const key = symStr[si] + '\x00' + symStr[sj];
            if (this.bpeRanks.get(key) !== rank) continue;

            // Merge si ← sj
            symStr[si] += symStr[sj];
            symStr[sj] = null;

            const nar = symNext[sj];
            symNext[si] = nar;
            if (nar >= 0) symPrev[nar] = si;

            mergesDone++;

            if (symPrev[si] >= 0) addBigram(symPrev[si], si, symStr, this.bpeRanks, pq);
            if (symNext[si] >= 0) addBigram(si, symNext[si], symStr, this.bpeRanks, pq);
        }

        // Collect surviving symbols → token IDs
        const result = [];
        for (let i = 0; i < len; i++) {
            if (!symStr[i]) continue;
            const id = this.tokenToId.get(symStr[i]);
            if (id !== undefined) {
                result.push(id);
            } else {
                // Fallback: return each byte character for decode
                for (let j = 0; j < symStr[i].length; j++) {
                    result.push(symStr[i][j]);
                }
            }
        }
        return result;
    }

    decode(tokenIds) {
        const decoder = new TextDecoder('utf-8');
        let out = '';
        for (const id of tokenIds) {
            if (typeof id === 'string') {
                const cp = id.codePointAt(0);
                const b = _gpt2CharToByte.get(id);
                if (b !== undefined) {
                    out += decoder.decode(new Uint8Array([b]));
                } else {
                    out += id;
                }
            } else {
                const tok = this.idToToken[id];
                if (tok != null) {
                    out += gpt2Decode(tok);
                }
            }
        }
        return out.replace(/\u2581/g, ' ');
    }

    get vocabSize() { return this.idToToken.length; }
}

// --- GPT-2 bytes_to_unicode / unicode_to_byte (reference implementation) ---

let _gpt2ByteToChar = null;
let _gpt2CharToByte = null;

function buildGPT2ByteMap() {
    if (_gpt2ByteToChar) return _gpt2ByteToChar;
    // Safe byte ranges: !~(33-126), ¡¬(161-172), ®ÿ(174-255)
    const bs = [];
    for (let i = 0x21; i <= 0x7E; i++) bs.push(i);     // ! to ~
    for (let i = 0xA1; i <= 0xAC; i++) bs.push(i);     // ¡ to ¬
    for (let i = 0xAE; i <= 0xFF; i++) bs.push(i);     // ® to ÿ

    const cs = [];
    for (const b of bs) cs.push(b);

    let n = 0;
    for (let b = 0; b < 256; b++) {
        if (!bs.includes(b)) {
            bs.push(b);
            cs.push(0x100 + n++);
        }
    }

    _gpt2ByteToChar = new Array(256);
    _gpt2CharToByte = new Map();
    for (let i = 0; i < bs.length; i++) {
        const ch = String.fromCharCode(cs[i]);
        _gpt2ByteToChar[bs[i]] = ch;
        _gpt2CharToByte.set(ch, bs[i]);
    }
    return _gpt2ByteToChar;
}

function gpt2Decode(s) {
    const bytes = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) {
        const ch = s[i];
        const b = _gpt2CharToByte.get(ch);
        bytes[i] = b !== undefined ? b : s.codePointAt(i);
    }
    return new TextDecoder('utf-8').decode(bytes);
}

// --- BPE helpers ---

function addBigram(li, ri, symStr, bpeRanks, pq) {
    const l = symStr[li], r = symStr[ri];
    if (!l || !r) return;
    const rank = bpeRanks.get(l + '\x00' + r);
    if (rank === undefined) return;
    heapPush(pq, [rank, li]);
}

function heapPush(h, item) {
    h.push(item);
    let i = h.length - 1;
    while (i > 0) {
        const p = (i - 1) >> 1;
        if (h[p][0] <= h[i][0]) break;
        [h[i], h[p]] = [h[p], h[i]];
        i = p;
    }
}

function heapPop(h) {
    const top = h[0];
    const last = h.pop();
    if (h.length > 0) {
        h[0] = last;
        let i = 0;
        while (true) {
            let s = i, l = 2 * i + 1, r = 2 * i + 2;
            if (l < h.length && h[l][0] < h[s][0]) s = l;
            if (r < h.length && h[r][0] < h[s][0]) s = r;
            if (s === i) break;
            [h[i], h[s]] = [h[s], h[i]];
            i = s;
        }
    }
    return top;
}

function byteFallbackTokens(tokenToId, ch) {
    const utf8 = new TextEncoder().encode(ch);
    let out = '';
    for (let i = 0; i < utf8.length; i++) {
        const g2ch = _gpt2ByteToChar[utf8[i]];
        out += g2ch;
        const id = tokenToId.get(g2ch);
        if (id !== undefined) return [id];
    }
    return [out];
}

// --- Pre-tokenization（特殊トークンを保護） ---

function pretokenize(text, isQwen35) {
    const basePat = isQwen35
        ? "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
        : "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    // 特殊トークンを最優先マッチとして先頭に追加
    const pat = `(?:${SPECIAL_TOKEN_PATTERN})|${basePat}`;

    const regex = new RegExp(pat, 'dgu');
    const words = [];
    let m;
    while ((m = regex.exec(text)) !== null) {
        if (m[0]) words.push(m[0]);
    }
    return words;
}
