# Qwen3 JavaScript 推論エンジン 実装計画

## 概要

外部ライブラリを使わずに純粋なJavaScriptでQwen3 0.6B (Q8_0 GGUF) の推論エンジンを実装する。
ブラウザ環境をターゲットとし、GGUFファイルを直接読み込んで推論を行う。

---

## Qwen3 0.6B アーキテクチャ（qwen3.cpp より）

### ハイパーパラメータ
- **n_layer**: 28
- **n_embd**: 1024
- **n_head / n_head_kv**: GQA使用（GGUFから取得）
- **n_ff**: FFN隠れ次元（GGUFから取得）
- **n_vocab**: ボキャブラリサイズ（GGUFから取得）
- **RoPE**: n_rot = n_embd_head_k = n_embd / n_head

### 各レイヤーのテンソル（GGUF実測値）
| テンソル名 | 形状 | 説明 |
|---|---|---|
| `token_embd.weight` | [n_embd, n_vocab] = [1024, 151936] | トークン埋め込み (Q8_0) |
| `output_norm.weight` | [n_embd] = [1024] | 出力RMSNorm (F32) |
| `output.weight` | [n_embd, n_vocab] | lm_head（weight tying時はtoken_embd共用） |
| `blk.{i}.attn_norm.weight` | [n_embd] = [1024] | アテンション前RMSNorm (F32) |
| `blk.{i}.attn_q.weight` | [n_embd, n_head*head_dim] = [1024, 2048] | Q投影（分離）(Q8_0) |
| `blk.{i}.attn_k.weight` | [n_embd, n_head_kv*head_dim] = [1024, 1024] | K投影（分離）(Q8_0) |
| `blk.{i}.attn_v.weight` | [n_embd, n_head_kv*head_dim] = [1024, 1024] | V投影（分離）(Q8_0) |
| `blk.{i}.attn_output.weight` | [n_head*head_dim, n_embd] = [2048, 1024] | アテンション出力投影 (Q8_0) |
| `blk.{i}.attn_q_norm.weight` | [head_dim] = [128] | **QベクトルごとのRMSNorm** (F32) |
| `blk.{i}.attn_k_norm.weight` | [head_dim] = [128] | **KベクトルごとのRMSNorm** (F32) |
| `blk.{i}.ffn_norm.weight` | [n_embd] = [1024] | FFN前RMSNorm (F32) |
| `blk.{i}.ffn_gate.weight` | [n_embd, n_ff] = [1024, 3072] | SwiGLU gate投影 (Q8_0) |
| `blk.{i}.ffn_up.weight` | [n_embd, n_ff] = [1024, 3072] | SwiGLU up投影 (Q8_0) |
| `blk.{i}.ffn_down.weight` | [n_ff, n_embd] = [3072, 1024] | FFN down投影 (Q8_0) |

**重要**: QKVは結合されておらず、個別のテンソルとして保存されている。
head_dim = key_length = value_length = 128

### 順伝播グラフ（1レイヤーあたり）
```
input → attn_norm(RMSNorm)
       → QKV投影 → 分割(Q, K, V)
       → Q: q_norm(RMSNorm per-head) → RoPE
       → K: k_norm(RMSNorm per-head) → RoPE
       → V: そのまま
       → Self-Attention(Q, K, V)
       → attn_out投影
       → residual: input + attention_output = ffn_input
       → ffn_norm(RMSNorm)
       → SwiGLU: silu(X @ gate_w) * (X @ up_w) @ down_w
       → residual: ffn_input + ffn_output = output
```

### 最終出力
```
last_layer_output → output_norm(RMSNorm) → lm_head(output.weight) → logits
```

---

## ファイル構成

```
qwenqwen4/
├── src/
│   ├── gguf.js          # GGUFヘッダー・メタデータパーサー
│   ├── tokenizer.js     # BPEトークナイザー（Qwen2/Qwen3.5対応）
│   ├── tensor.js        # テンソル演算ユーティリティ
│   ├── qwen3-engine.js  # Qwen3推論エンジン本体
│   └── kv-cache.js      # K/Vキャッシュ管理
├── html/
│   ├── header-viewer.html   # GGUFヘッダー情報表示用UI
│   ├── tokenizer-test.html  # トークナイズ結果表示用UI
│   └── chat.html            # チャット推論UI
├── llama-vocab.cpp      # 参照ファイル（C++）
└── qwen3.cpp            # 参照ファイル（C++）
```

---

## 各モジュールの詳細設計

### 1. `src/gguf.js` — GGUFパーサー

GGUFバイナリフォーマットを直接パースする。

#### GGUFヘッダー構造
- Signature: "GGUF" (4 bytes)
- Version: uint32 (現在は3)
- n_tensors: uint64
- n_kv_values: uint64

#### 実装クラス
```javascript
class GGUFReader {
    constructor(arrayBuffer) { ... }
    
    // ヘッダー読み込み
    getHeader()           // { signature, version, nTensors, nKvValues }
    
    // キーバリューメタデータ
    getKVCount()          // メタデータ数
    getKVType(index)      // 型 (uint8/int8/uint16/uint32/uint64/int32/int64/float/double.bool/string/array)
    getKVKey(index)       // キー名（文字列）
    getKVValue(index)     // 値の取得（型に応じたデコード）
    
    // テンソル情報（データをコピーせずメタデータのみ保持）
    getTensorCount()      // テンソル数
    getTensorName(index)  // テンソル名
    getTensorShape(index) // 形状 [dims...]
    getTensorType(index)  // 重み型 (F32, F16, Q8_0, etc.)
    getTensorMeta(index)  // { buffer: ArrayBuffer, offset: number, nbytes: number, shape: number[], type: string }
    
    // 検索ヘルパー
    findKey(keyName)      // キー名でメタデータを検索
    getArch()             // アーキテクチャ関連パラメータをまとめて取得
}
```

#### Q8_0 ブロック構造（展開せず直接参照）
Q8_0ブロック形式: 各ブロックは [d(float16), blocks_per_channel(int8), bpe[QK_K=256]×int8] の構造。

**重要**: F32への全展開はメモリ不足を招くため行わない。GGUFのArrayBuffer上のQ8_0データをそのまま参照し、
matmulなどの演算内でブロック単位でオンザフライデコードする。
- 重みデータ: ArrayBuffer上のQ8_0ブロックをoffset/lengthで直接アクセス
- matmul: Q8_0ブロック×F32ベクトル → F32スカラー をブロックごとに計算してaccumulate
- 作業用バッファのみF32（1トークン分の隠れ状態など）

#### 取得する主要メタデータ
```
general.architecture → "qwen3"
general.name
general.parameter_count
general.quantization_version

qwen3.vocab_size
qwen3.context_length
qwen3.embedding_length
qwen3.block_count
qwen3.feed_forward_length
qwen3.attention.head_count
qwen3.attention.head_count_kv
qwen3.attention.layer_norm_rms_epsilon
qwen3.rope.freq_base

tokenizer.ggml.model → "gpt2" (BPE)
tokenizer.ggml.tokens → 文字列配列
tokenizer.ggml.scores → スコア配列
tokenizer.ggml.token_type → トークンタイプ配列
tokenizer.ggml.merges → マージルール配列
tokenizer.ggml.bos_token_id
tokenizer.ggml.eos_token_id
tokenizer.chat_template
```

---

### 2. `src/tokenizer.js` — BPEトークナイザー

llama-vocab.cpp のBPEトークナイザーをJavaScriptに移植。

#### Qwen3のプリトークナイズ regex（QWEN35 タイプ）
```
(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|
[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|
\p{N}|
 ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|
\s*[\r\n]+|
\s+(?!\S)|
\s+
```

#### 実装クラス
```javascript
class BPETokenizer {
    constructor(vocabData) {
        // vocabData: { tokens: string[], merges: string[], bosTokenId, eosTokenId }
        this.tokenToId = new Map();     // token文字列 → ID
        this.idToToken = [];            // ID → 文字列
        this.bpeRanks = new Map();      // "left\0right" → rank
        this.bosTokenId = bosTokenId;
        this.eosTokenId = eosTokenId;
    }
    
    tokenize(text, addBos = false, addEos = false): number[] {
        // 1. special tokenで分割
        // 2. pretokenize (regex split)
        // 3. 各ワードをUTF8文字に分解 → BPEマージ
        // 4. 未認識文字はバイトトークンにフォールバック
    }
    
    decode(tokenIds): string {
        // トークンID配列 → 文字列（U+2581→スペース変換、スペース整理）
    }
}
```

#### BPEマージアルゴリズム
1. ワードをUTF-8コードポイントの配列に分解
2. 隣接するペアのbpeRanksを優先キューに登録
3. 最優先（最小rank）のペアをマージ → 新たなペアを登録
4. マージが不可能になるまで繰り返し
5. 最終的なシーケンスをトークンIDに変換

---

### 3. `src/tensor.js` — テンソル演算ユーティリティ

NumPy風のアレイ操作と線形代数演算。

#### 基本操作
```javascript
// Q8_0ブロック×F32ベクトル の行列積（オンザフライデコード）
// meta: {buffer, offset, shape[rows, cols], type} × input: Float32Array[cols] → Float32Array[rows]
function matmulQ80xF32(meta, input) → Float32Array

// F32ベクトル×F32行列（小規模な演算用、norm重みなど）
function rmsnorm(input, weight_F32, eps) → Float32Array  // weightはF32（小規模）

// ベクトル演算
function addVec(a, b) → void (in-place)
function silu(x) → x * sigmoid(x)  // in-place

// ソフトマックス
function softmax(logits, temp = 1.0) → Float32Array

// サンプリング
function sampleTopPTopK(logits, topP, topK, temperature) → tokenId
```

#### Q8_0ブロック構造定数
- `QK_Q8_0 = 32`（1ブロックあたりの要素数）
- ブロックサイズ: `sizeof(block_q80) = 2 + 4 + 256 = 262 bytes`
- ブロック内: `d(float16) + blocks_per_channel(int8) + bpe[256]×uint8`

#### RMSNorm実装
```javascript
function rmsnorm(x, weight, eps) {
    const n = x.length;
    let sum = 0;
    for (let i = 0; i < n; i++) sum += x[i] * x[i];
    const rms = Math.sqrt(sum / n + eps);
    const out = new Float32Array(n);
    for (let i = 0; i < n; n++) out[i] = weight[i] * x[i] / rms;
    return out;
}
```

#### Q/K Norm（per-head RMSNorm）
Qwen3固有: QとKは [n_heads, seq_len, head_dim] の形状で、各headベクトルごとにRMSNormを適用。

---

### 4. `src/kv-cache.js` — K/Vキャッシュ管理

推論の高速化のため、過去のKとVをキャッシュする。

```javascript
class KVCache {
    constructor(nLayers, nHeadKV, nCtx, headDim) {
        // kCache[layer][head][position][dim] → 平らなFloat32Array
        this.kCache = new Float32Array(nLayers * nHeadKV * nCtx * headDim);
        this.vCache = new Float32Array(nLayers * nHeadKV * nCtx * headDim);
        this.nPast = 0;  // キャッシュ済みトークン数
    }
    
    store(layer, position, kData, vData) { ... }
    getK(layer, head, start, end) → Float32Array (view)
    getV(layer, head, start, end) → Float32Array (view)
    reset() { this.nPast = 0; }
}
```

**メモリ最適化**: 事前に確保したFloat32Arrayを直接操作。コピーなしでviewとしてアクセス。

---

### 5. `src/qwen3-engine.js` — Qwen3推論エンジン本体

GGUFから読み込んだモデルパラメータを使って順伝播を実行。

#### クラス設計
```javascript
class Qwen3Engine {
    constructor(ggufReader) {
        // GGUFからアーキテクチャ情報を取得
        this.nLayers = gguf.getArch().blockCount;       // 28
        this.nEmbbed = gguf.getArch().embeddingLength;   // 1024
        this.nHeads = gguf.getArch().attention.headCount;
        this.nHeadKV = gguf.getArch().attention.headCountKV;
        this.nFF = gguf.getArch().feedForwardLength;
        this.nVocab = gguf.getArch().vocabSize;
        this.ropeEps = gguf.getArch().layerNormRmsEpsilon;
        this.ropeFreqBase = gguf.getArch().rope.freqBase;
        
        // テンソルデータをFloat32Arrayとして読み込み（Q8_0ならデコード）
        this.loadWeights(ggufReader);
        
        // 作業用バッファを事前に確保（再割り当て回避）
        this.allocBuffers();
    }
    
    loadWeights(gguf) {
        // 各テンソルは { buffer, offset, nbytes, shape, type } のメタデータのみ保持
        // データ自体のコピーは一切行わない → ArrayBuffer上のQ8_0ブロックを直接参照
        
        this.tokEmbd = gguf.getTensorMetaByName("token_embd.weight");
        this.outputNorm = gguf.getTensorMetaByName("output_norm.weight");
        const outputTensor = gguf.getTensorMetaByName("output.weight");
        this.output = outputTensor || this.tokEmbd; // weight tying
        
        for (let i = 0; i < this.nLayers; i++) {
            const prefix = `blk.${i}`;
            this.layers[i] = {
                attnNorm: gguf.getTensorMeta(`${prefix}.attn_norm.weight`),
                wq:       gguf.getTensorMeta(`${prefix}.attn_q.weight`),     // 分離Q投影 [n_embd, n_head*head_dim]
                wk:       gguf.getTensorMeta(`${prefix}.attn_k.weight`),     // 分離K投影 [n_embd, n_head_kv*head_dim]
                wv:       gguf.getTensorMeta(`${prefix}.attn_v.weight`),     // 分離V投影 [n_embd, n_head_kv*head_dim]
                wo:       gguf.getTensorMeta(`${prefix}.attn_output.weight`),// 出力投影 [n_head*head_dim, n_embd]
                qNorm:    gguf.getTensorMeta(`${prefix}.attn_q_norm.weight`),
                kNorm:    gguf.getTensorMeta(`${prefix}.attn_k_norm.weight`),
                ffnNorm:  gguf.getTensorMeta(`${prefix}.ffn_norm.weight`),
                ffnGate:  gguf.getTensorMeta(`${prefix}.ffn_gate.weight`),
                ffnUp:    gguf.getTensorMeta(`${prefix}.ffn_up.weight`),
                ffnDown:  gguf.getTensorMeta(`${prefix}.ffn_down.weight`),
            };
        }
    }
    }
    
    // バッファ事前確保（推論中のGC・再割り当てを回避）
    allocBuffers() {
        this.bufHidden = new Float32Array(this.nEmbbed);
        this.bufResidual = new Float32Array(this.nEmbbed);
        this.bufQ = new Float32Array(this.nHeads * this.headDim);
        this.bufK = new Float32Array(this.nHeadKV * this.headDim);
        this.bufV = new Float32Array(this.nHeadKV * this.headDim);
        // ... 必要に応じて追加
    }
    
    // シングルステップ推論（1トークン分）
    forward(tokenId, position) → logits(Float32Array[nVocab]) {
        // 1. Embedding lookup
        let hidden = this.getEmbedding(tokenId);
        let residual = hidden;
        
        for (let l = 0; l < this.nLayers; l++) {
            const layer = this.layers[l];
            
            // Attention norm
            hidden = rmsnorm(hidden, layer.attnNorm, this.eps);
            
            // Separate Q/K/V projections (GGUF stores them separately)
            const Q = matmulQ80xF32(layer.wq, hidden);  // [n_head*head_dim]
            const K = matmulQ80xF32(layer.wk, hidden);  // [n_head_kv*head_dim]
            const V = matmulQ80xF32(layer.wv, hidden);  // [n_head_kv*head_dim]
            
            // Per-head RMSNorm on Q and K (Qwen3固有)
            Q = this.perHeadRMSNorm(Q, layer.qNorm, this.eps);
            K = this.perHeadRMSNorm(K, layer.kNorm, this.eps);
            
            // RoPE
            applyRoPE(Q, position, this.headDim, this.ropeFreqBase);
            applyRoPE(K, position, this.headDim, this.ropeFreqBase);
            
            // Cache K, V
            this.kvCache.store(l, position, K, V);
            
            // Self-attention (GQA) → output projection
            const attnOut = selfAttention(Q, K, V, layer.wo, this.kvCache, l, position);
            residual = hidden;
            
            // FFN norm
            hidden = rmsnorm(hidden, layer.ffnNorm, this.eps);
            
            // SwiGLU parallel FFN
            const ffnOut = swiGLU(hidden, layer.ffnGate, layer.ffnUp, layer.ffnDown);
            
            // Residual connection
            hidden = add(ffnOut, residual);
            residual = hidden;
        }
        
        // Output norm + lm_head
        hidden = rmsnorm(hidden, this.outputNorm, this.eps);
        const logits = matmulVec(this.output, hidden);
        return logits;
    }
    
    // 生成ループ
    async generate(promptTokens, maxSteps, onToken) {
        for (let pos = 0; pos < promptTokens.length; pos++) {
            const logits = this.forward(promptTokens[pos], pos);
        }
        // promptの最後のhidden状態を保持
        
        let currentToken = promptTokens[promptTokens.length - 1];
        for (let step = 0; step < maxSteps; step++) {
            const pos = promptTokens.length + step;
            const logits = this.forward(currentToken, pos);
            
            // Apply logit bias, temperature, top-p/top-k sampling
            currentToken = sample(logits, ...);
            
            if (currentToken === this.eosTokenId) break;
            
            onToken(currentToken);
            
            // yield control to browser
            await new Promise(r => setTimeout(r, 0));
        }
    }
}
```

#### RoPE（Rotary Positional Embedding）実装 — split-half
Qwen3はsplit-half方式: `(x[i], x[i + headDim/2])` をペアに回転。
隣接ペア `(x[2i], x[2i+1])` ではない。

```javascript
function applyRoPE(tensor, position, headDim, freqBase) {
    const half = headDim >> 1;
    for (let i = 0; i < half; i++) {
        const freq = 1.0 / Math.pow(freqBase, i / half);
        const theta = position * freq;
        const cos = Math.cos(theta);
        const sin = Math.sin(theta);
        const x0 = tensor[i];
        const x1 = tensor[i + half];
        tensor[i]       = x0 * cos - x1 * sin;
        tensor[i + half] = x0 * sin + x1 * cos;
    }
}
```

#### Self-Attention実装（GQA対応）
```javascript
function selfAttention(Q, KCache, VCache, Wo, kvCache, layer, position) {
    const nHeads = this.nHeads;
    const nHeadKV = this.nHeadKV;
    const headDim = this.headDim;
    const scale = 1.0 / Math.sqrt(headDim);
    
    // Q: [n_heads, head_dim]
    // K: [n_head_kv, position+1, head_dim] (cacheから)
    // V: [n_head_kv, position+1, head_dim] (cacheから)
    
    const attnWeights = new Float32Array(nHeads * (position + 1));
    
    for (let h = 0; h < nHeads; h++) {
        const kvHead = Math.floor(h / (nHeads / nHeadKV)); // GQAマッピング
        
        for (let p = 0; p <= position; p++) {
            let dot = 0;
            for (let d = 0; d < headDim; d++) {
                dot += Q[h * headDim + d] * KCache[layer][kvHead][p][d];
            }
            attnWeights[h * (position + 1) + p] = dot * scale;
        }
        
        // Softmax
        softmaxInPlace(attnWeights.subarray(h * (position + 1), (h + 1) * (position + 1)));
    }
    
    // Vとの積和 → attention出力 [n_head*head_dim]
    const attnResult = new Float32Array(nHeads * headDim);
    for (let h = 0; h < nHeads; h++) {
        const kvHead = Math.floor(h / (nHeads / nHeadKV)); // GQAマッピング
        
        for (let p = 0; p <= position; p++) {
            const w = attnWeights[h * (position + 1) + p];
            for (let d = 0; d < headDim; d++) {
                attnResult[h * headDim + d] += VCache[layer][kvHead][p][d] * w;
            }
        }
    }
    
    // Output projection: Wo @ attnResult → [n_embd]
    return matmulQ80xF32(Wo, attnResult);  // Wo: [n_head*head_dim, n_embd] (Q8_0)
            }
        }
    }
    
    return output;
}
```

---

### 6. HTML UI

#### `html/header-viewer.html` — GGUFヘッダー表示
- ファイル選択ボタンでGGUFファイルを読み込み
- GGUFReaderでパースしたメタデータをテーブル表示
- テンソル一覧（名前、形状、型）を表示
- アーキテクチャパラメータを要約表示

#### `html/tokenizer-test.html` — トークナイズ結果表示
- ファイル選択 + テキスト入力欄
- トークナイズ実行ボタン
- 各トークンのIDと文字列を並べて表示
- デコード結果も併記して整合性確認

#### `html/chat.html` — チャット推論UI
- GGUFファイル読み込み（非同期、プログレスバー付き）
- チャットメッセージ一覧
- プロンプト入力欄 + 送信ボタン
- ストリーミング出力表示（トークンごとに追加描画）
- システムプロンプト設定（chat_template使用）

---

## メモリ最適化戦略

### 0.6B Q8_0 のメモリ見積もり
| コンポーネント | サイズ |
|---|---|
| モデル重み (Q8_0) | ~600MB（ArrayBufferそのまま） |
| 作業用F32バッファ | 数MB（1トークン分の隠れ状態など） |
| **合計** | **~605MB** ← F32展開しないため大幅削減 |

### メモリ最適化戦略（Q8_0直接参照）
重みデータはGGUFのArrayBuffer上のQ8_0ブロックをそのまま参照。F32への全展開を行わない。

1. **Q8_0直接matmul**: 行列積でQ8_0ブロック×F32ベクトル → ブロックごとにスケール適用してaccumulate
2. **作業バッファのみF32**: 隠れ状態、residual、出力logitsなど推論中に必要な配列のみF32
3. **テンソルメタデータ**: 各テンソルのArrayBuffer上のoffset/size/shape/typeを保持（データ自体はコピーしない）
4. **`subarray()` view活用**: コピーなしで部分参照。`structuredClone` やスライスコピーを一切避ける

---

## 実装順序

### Phase 1: GGUFパーサー（基礎）
1. バイナリフォーマットパース（ヘッダー・KV・テンソル情報）
2. Q8_0デコーダー
3. `header-viewer.html` で動作確認

### Phase 2: トークナイザー
4. BPEトークナイザー実装（llama-vocab.cppのBPE部分を移植）
5. プリトークナイズregex（Qwen3用）
6. `tokenizer-test.html` で動作確認

### Phase 3: テンソル演算・エンジン
7. RMSNorm, matmul, RoPE, softmax, サンプリング
8. KVCache実装
9. Qwen3Engine.forward() 実装（1レイヤー→全レイヤー）
10. generate() ループ

### Phase 4: チャットUI
11. chat_template処理
12. `chat.html` でエンドツーエンド動作確認

---

## 技術的注意点

### Q8_0フォーマット詳細（直接参照用）
各ブロック: `[block_q8_0]` (34 bytes)
- `d`: float16 (2 bytes, スケール)
- `blocks_per_channel`: uint8 (1 byte)
- `bpe`: uint8[QK_K=32]

matmulでのオンザフライデコード:
```javascript
function matmulQ80xF32(meta, inputF32) {
    const { buffer, offset, shape, type } = meta;
    const [rows, cols] = shape;  // e.g. [1024, 151936] for token_embd
    const nBlocksPerRow = cols / QK_Q8_0;  // ブロック数/行
    const rowBytes = GGUFReader.BLOCK_SIZE_Q8_0 * nBlocksPerRow;
    
    const i8 = new Int8Array(buffer);
    const f16 = new Float16Array(buffer);
    const outputF32 = new Float32Array(rows);
    
    for (let r = 0; r < rows; r++) {
        let sum = 0;
        const rowPtr = offset + r * rowBytes;
        
        for (let b = 0; b < nBlocksPerRow; b++) {
            const blockPtr = rowPtr + b * GGUFReader.BLOCK_SIZE_Q8_0;
            // d(float16) はブロック先頭、bpe(int8[256]) は+5バイト目から
            const d = f16[blockPtr >> 2];  // float16として読み出し
            const bpeOff = blockPtr + 5;   // uint8配列上のオフセット
            
            for (let i = 0; i < QK_Q8_0; i++) {
                const colIdx = b * QK_Q8_0 + i;
                sum += i8[bpeOff + i] * d * inputF32[colIdx];
            }
        }
        outputF32[r] = sum;
    }
    return outputF32;
}
```

→ これにより、重みデータは常にQ8_0のままでArrayBuffer上に残り、
ピークメモリは ~600MB + 作業用数MB で抑えられる。

### Qwen3固有のアーキテクチャ特徴
1. **Q/K Norm**: QとKにper-head RMSNormを適用（標準LLaMAにはない）
2. **SwiGLU parallel FFN**: `silu(X @ W_gate) * (X @ W_up) @ W_down`
3. **ffn_norm**: FFN前にもRMSNormがある

### ブラウザ環境の制約
- Web Workerで推論をバックグラウンド実行（UIブロッキング回避）
- 大きなArrayBufferはTransferable Objectでワーカー間転送
- `setTimeout` でチャンク処理し、ブラウザフリーズを防ぐ

---

## 参照ファイルのマッピング

| C++ (llama.cpp) | JavaScript実装 |
|---|---|
| GGUFヘッダーパース | `src/gguf.js` → GGUFReader |
| ggml_quantize.h Q8_0 | `src/gguf.js` → decodeQ8_0() |
| llm_tokenizer_bpe_session | `src/tokenizer.js` → BPETokenizer.tokenize() |
| unicode_regex_split | `src/tokenizer.js` → pretokenize() |
| llama_model_qwen3::graph | `src/qwen3-engine.js` → Qwen3Engine.forward() |
| ggml_rope_ext | `src/tensor.js` → applyRoPE() |
| build_norm (RMS) | `src/tensor.js` → rmsnorm() |
| build_ffn (SILU, PAR) | `src/qwen3-engine.js` → swiGLU() |
