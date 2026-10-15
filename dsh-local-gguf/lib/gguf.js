/**
 * Minimal, dependency-free GGUF reader.
 *
 * Parses the GGUF header, the metadata key/value table, and the tensor-info
 * table of a `.gguf` file, stopping before the (potentially huge) tensor data.
 * Everything here is read-only and works on a byte-range reader, so inspecting a
 * 20 GiB model costs a few hundred kilobytes of I/O.
 *
 * Reference: ggml `gguf` spec (magic, version, tensor_count, metadata_kv_count,
 * then KV pairs, then tensor infos).
 */

import fs from 'node:fs'

/** GGUF magic, little-endian: the bytes 'G','G','U','F'. */
export const GGUF_MAGIC = 0x46554747

/** GGUF metadata value type tags, in wire order. */
const VALUE_TYPE = {
  UINT8: 0,
  INT8: 1,
  UINT16: 2,
  INT16: 3,
  UINT32: 4,
  INT32: 5,
  FLOAT32: 6,
  BOOL: 7,
  STRING: 8,
  ARRAY: 9,
  UINT64: 10,
  INT64: 11,
  FLOAT64: 12,
}

/**
 * Fixed byte widths for the non-string GGUF value types, used to skip long
 * numeric arrays without reading every element. Strings are variable-length and
 * must be walked one at a time.
 */
const FIXED_WIDTH = {
  [VALUE_TYPE.UINT8]: 1,
  [VALUE_TYPE.INT8]: 1,
  [VALUE_TYPE.UINT16]: 2,
  [VALUE_TYPE.INT16]: 2,
  [VALUE_TYPE.UINT32]: 4,
  [VALUE_TYPE.INT32]: 4,
  [VALUE_TYPE.FLOAT32]: 4,
  [VALUE_TYPE.BOOL]: 1,
  [VALUE_TYPE.UINT64]: 8,
  [VALUE_TYPE.INT64]: 8,
  [VALUE_TYPE.FLOAT64]: 8,
  [VALUE_TYPE.STRING]: 0,
  [VALUE_TYPE.ARRAY]: 0,
}

/**
 * `general.file_type` enum from llama.cpp (`llama_ftype`), used only as a
 * human-readable fallback when the tensor table cannot be read.
 */
const FILE_TYPE = {
  0: 'F32',
  1: 'F16',
  2: 'Q4_0',
  3: 'Q4_1',
  4: 'Q4_1_SOME_F16',
  5: 'Q8_0',
  6: 'Q5_0',
  7: 'Q5_1',
  8: 'Q2_K',
  9: 'Q3_K_S',
  10: 'Q3_K_M',
  11: 'Q3_K_L',
  12: 'Q4_K_S',
  13: 'Q4_K_M',
  14: 'Q5_K_S',
  15: 'Q5_K_M',
  16: 'Q6_K',
  17: 'IQ2_XXS',
  18: 'IQ2_XS',
  19: 'Q2_K_S',
  20: 'IQ3_XS',
  21: 'IQ3_XXS',
  22: 'IQ1_S',
  23: 'IQ4_NL',
  24: 'IQ3_S',
  25: 'IQ3_M',
  26: 'IQ2_S',
  27: 'IQ2_M',
  28: 'IQ4_XS',
  29: 'IQ1_M',
  30: 'BF16',
  31: 'Q4_0_4_4',
  32: 'Q4_0_4_8',
  33: 'Q4_0_8_8',
  34: 'TQ1_0',
  35: 'TQ2_0',
  36: 'MXFP4_MOE',
}

/**
 * ggml tensor types: element count per block and bytes per block. Used to turn a
 * tensor's dimensions into exact on-disk bytes, which is what makes the memory
 * estimate authoritative rather than a guess from the file size.
 */
const GGML_TYPE = {
  0: { name: 'F32', block: 1, bytes: 4 },
  1: { name: 'F16', block: 1, bytes: 2 },
  2: { name: 'Q4_0', block: 32, bytes: 18 },
  3: { name: 'Q4_1', block: 32, bytes: 20 },
  6: { name: 'Q5_0', block: 32, bytes: 22 },
  7: { name: 'Q5_1', block: 32, bytes: 24 },
  8: { name: 'Q8_0', block: 32, bytes: 34 },
  9: { name: 'Q8_1', block: 32, bytes: 40 },
  10: { name: 'Q2_K', block: 256, bytes: 84 },
  11: { name: 'Q3_K', block: 256, bytes: 110 },
  12: { name: 'Q4_K', block: 256, bytes: 144 },
  13: { name: 'Q5_K', block: 256, bytes: 176 },
  14: { name: 'Q6_K', block: 256, bytes: 210 },
  15: { name: 'Q8_K', block: 256, bytes: 292 },
  16: { name: 'IQ2_XXS', block: 256, bytes: 66 },
  17: { name: 'IQ2_XS', block: 256, bytes: 74 },
  18: { name: 'IQ3_XXS', block: 256, bytes: 98 },
  19: { name: 'IQ1_S', block: 256, bytes: 50 },
  20: { name: 'IQ4_NL', block: 32, bytes: 18 },
  21: { name: 'IQ3_S', block: 256, bytes: 110 },
  22: { name: 'IQ2_S', block: 256, bytes: 82 },
  23: { name: 'IQ4_XS', block: 256, bytes: 136 },
  24: { name: 'I8', block: 1, bytes: 1 },
  25: { name: 'I16', block: 1, bytes: 2 },
  26: { name: 'I32', block: 1, bytes: 4 },
  27: { name: 'I64', block: 1, bytes: 8 },
  28: { name: 'F64', block: 1, bytes: 8 },
  29: { name: 'IQ1_M', block: 256, bytes: 56 },
  30: { name: 'BF16', block: 1, bytes: 2 },
  34: { name: 'TQ1_0', block: 256, bytes: 54 },
  35: { name: 'TQ2_0', block: 256, bytes: 66 },
  39: { name: 'MXFP4', block: 32, bytes: 17 },
}

/** Human-readable name of a ggml tensor type id. */
export function tensorTypeName(id) {
  return GGML_TYPE[id]?.name ?? `TYPE_${id}`
}

/**
 * A cursor over a file that pulls 1 MiB windows on demand, so metadata parsing
 * never loads tensor data into memory.
 */
class ByteReader {
  constructor(fd, fileSize) {
    this.fd = fd
    this.fileSize = fileSize
    this.base = 0
    this.buf = Buffer.alloc(0)
  }

  /** Ensure `length` bytes starting at absolute `offset` are buffered. */
  #window(offset, length) {
    if (offset >= this.base && offset + length <= this.base + this.buf.length) return
    const size = Math.max(length, 1 << 20)
    const start = Math.max(0, offset)
    const want = Math.min(size, Math.max(0, this.fileSize - start))
    const buf = Buffer.alloc(want)
    if (want > 0) fs.readSync(this.fd, buf, 0, want, start)
    this.base = start
    this.buf = buf
  }

  /** Read `length` bytes at absolute `offset` as a subarray of the window. */
  bytes(offset, length) {
    this.#window(offset, length)
    const rel = offset - this.base
    if (rel < 0 || rel + length > this.buf.length) {
      throw new Error(`GGUF read past end of file at offset ${offset} (+${length})`)
    }
    return this.buf.subarray(rel, rel + length)
  }

  u32(offset) {
    return this.bytes(offset, 4).readUInt32LE(0)
  }

  u64(offset) {
    return Number(this.bytes(offset, 8).readBigUInt64LE(0))
  }
}

/** Read one GGUF string (uint64 length + raw bytes) and advance the cursor. */
function readString(reader, cursor) {
  const length = reader.u64(cursor.offset)
  cursor.offset += 8
  const value = reader.bytes(cursor.offset, length).toString('utf8')
  cursor.offset += length
  return value
}

/**
 * Read one typed GGUF value and advance the cursor.
 *
 * Arrays are returned as `{ arrayType, length, values }` truncated to a bounded
 * preview: `tokenizer.ggml.tokens` holds >100k strings and nothing downstream
 * needs them, but its length is the vocabulary size.
 */
function readValue(reader, cursor, arrayBudget = { remaining: 4096 }) {
  const type = reader.u32(cursor.offset)
  cursor.offset += 4
  return readTypedValue(reader, cursor, type, arrayBudget)
}

/**
 * Read one value whose type tag has already been consumed.
 *
 * Array elements use this directly: GGUF writes an array's element type once and
 * then the elements with no per-element tag.
 */
function readTypedValue(reader, cursor, type, arrayBudget) {
  switch (type) {
    case VALUE_TYPE.UINT8:
      return reader.bytes(cursor.offset++, 1).readUInt8(0)
    case VALUE_TYPE.INT8:
      return reader.bytes(cursor.offset++, 1).readInt8(0)
    case VALUE_TYPE.UINT16: {
      const v = reader.bytes(cursor.offset, 2).readUInt16LE(0)
      cursor.offset += 2
      return v
    }
    case VALUE_TYPE.INT16: {
      const v = reader.bytes(cursor.offset, 2).readInt16LE(0)
      cursor.offset += 2
      return v
    }
    case VALUE_TYPE.UINT32: {
      const v = reader.u32(cursor.offset)
      cursor.offset += 4
      return v
    }
    case VALUE_TYPE.INT32: {
      const v = reader.bytes(cursor.offset, 4).readInt32LE(0)
      cursor.offset += 4
      return v
    }
    case VALUE_TYPE.FLOAT32: {
      const v = reader.bytes(cursor.offset, 4).readFloatLE(0)
      cursor.offset += 4
      return v
    }
    case VALUE_TYPE.BOOL: {
      const v = reader.bytes(cursor.offset++, 1).readUInt8(0) !== 0
      return v
    }
    case VALUE_TYPE.STRING:
      return readString(reader, cursor)
    case VALUE_TYPE.UINT64: {
      const v = reader.u64(cursor.offset)
      cursor.offset += 8
      return v
    }
    case VALUE_TYPE.INT64: {
      const v = Number(reader.bytes(cursor.offset, 8).readBigInt64LE(0))
      cursor.offset += 8
      return v
    }
    case VALUE_TYPE.FLOAT64: {
      const v = reader.bytes(cursor.offset, 8).readDoubleLE(0)
      cursor.offset += 8
      return v
    }
    case VALUE_TYPE.ARRAY: {
      const arrayType = reader.u32(cursor.offset)
      const length = reader.u64(cursor.offset + 4)
      cursor.offset += 12
      const values = []
      const fixed = FIXED_WIDTH[arrayType]
      if (fixed) {
        // Fixed-width elements: keep a bounded preview, then skip the tail by
        // size alone. `tokenizer.ggml.token_type` alone is >128k int32s.
        const take = Math.min(length, arrayBudget.remaining)
        for (let i = 0; i < take; i++) {
          values.push(readTypedValue(reader, cursor, arrayType, arrayBudget))
        }
        cursor.offset += (length - take) * fixed
        arrayBudget.remaining -= take
      } else {
        // Variable-width (strings) or nested arrays: every element must be walked
        // to advance the cursor, even once the preview budget is exhausted.
        for (let i = 0; i < length; i++) {
          const keep = arrayBudget.remaining > 0
          const value = readTypedValue(reader, cursor, arrayType, arrayBudget)
          if (keep) {
            values.push(value)
            arrayBudget.remaining--
          }
        }
      }
      return { arrayType, length, values, truncated: values.length < length }
    }
    default:
      throw new Error(`unsupported GGUF value type ${type} at offset ${cursor.offset - 4}`)
  }
}

/** Decode a metadata array of numbers into a plain array. */
function numbers(value) {
  if (Array.isArray(value)) return value
  if (value && Array.isArray(value.values)) return value.values
  return []
}

/**
 * Read the GGUF header, metadata, and tensor table of one file.
 *
 * @param {string} filePath absolute path to a `.gguf` file.
 * @returns {Promise<object>} parsed summary; see `summarizeGguf`.
 */
export async function readGguf(filePath) {
  const stat = await fs.promises.stat(filePath)
  const fd = fs.openSync(filePath, 'r')
  try {
    const reader = new ByteReader(fd, stat.size)
    const magic = reader.u32(0)
    if (magic !== GGUF_MAGIC) {
      throw new Error('not a GGUF file (bad magic)')
    }
    const version = reader.u32(4)
    if (version < 2 || version > 3) {
      throw new Error(`unsupported GGUF version ${version}`)
    }
    const tensorCount = reader.u64(8)
    const metadataCount = reader.u64(16)

    const cursor = { offset: 24 }
    const metadata = {}
    for (let i = 0; i < metadataCount; i++) {
      const key = readString(reader, cursor)
      metadata[key] = readValue(reader, cursor)
    }

    // Tensor table: name, n_dims, dims, type, offset. Summing exact per-tensor
    // bytes gives the true weight size for any quantization mix.
    const tensors = []
    let tensorBytes = 0
    let unknownTensors = 0
    for (let i = 0; i < tensorCount; i++) {
      const name = readString(reader, cursor)
      const nDims = reader.u32(cursor.offset)
      cursor.offset += 4
      const dims = []
      for (let d = 0; d < nDims; d++) {
        dims.push(reader.u64(cursor.offset))
        cursor.offset += 8
      }
      const typeId = reader.u32(cursor.offset)
      cursor.offset += 4
      cursor.offset += 8 // tensor data offset, unused here
      const info = GGML_TYPE[typeId]
      let bytes = 0
      if (info) {
        const elements = dims.reduce((a, b) => a * b, 1)
        bytes = Math.ceil(elements / info.block) * info.bytes
      } else {
        unknownTensors++
      }
      tensorBytes += bytes
      tensors.push({ name, dims, type: info?.name ?? `TYPE_${typeId}`, bytes })
    }

    return summarizeGguf({
      filePath,
      fileSize: stat.size,
      mtimeMs: stat.mtimeMs,
      version,
      tensorCount,
      metadata,
      tensors,
      tensorBytes,
      unknownTensors,
    })
  } finally {
    fs.closeSync(fd)
  }
}

/**
 * Fold raw GGUF metadata into the architecture facts the estimator and the
 * recommendation engine need.
 *
 * Metadata keys are namespaced by architecture (for example `qwen3.block_count`),
 * so every lookup is done against the architecture named by
 * `general.architecture` with the historical `llama.*` prefix as a fallback.
 */
function summarizeGguf(raw) {
  const { metadata, tensors } = raw
  const arch = String(metadata['general.architecture'] ?? 'unknown')

  const pick = (suffix) => {
    const direct = metadata[`${arch}.${suffix}`]
    if (direct !== undefined) return direct
    const legacy = metadata[`llama.${suffix}`]
    if (legacy !== undefined) return legacy
    return undefined
  }
  const num = (suffix) => {
    const v = pick(suffix)
    return typeof v === 'number' ? v : undefined
  }

  const blockCount = num('block_count')
  const embeddingLength = num('embedding_length')
  const headCount = num('attention.head_count')
  // Grouped-query attention stores fewer KV heads than query heads; the KV cache
  // is sized by the KV head count, which is the single biggest lever on VRAM.
  const headCountKv = num('attention.head_count_kv') ?? headCount
  const keyLength = num('attention.key_length') ?? (embeddingLength && headCount ? embeddingLength / headCount : undefined)
  const valueLength = num('attention.value_length') ?? keyLength

  const vocabSize = Array.isArray(metadata['tokenizer.ggml.tokens'])
    ? metadata['tokenizer.ggml.tokens'].length
    : metadata['tokenizer.ggml.tokens']?.length

  const fileTypeId = metadata['general.file_type']
  const quantFromTensors = dominantQuant(tensors)
  const splitCount = num('split.count')

  // Split the tensor table into per-transformer-block bytes and everything else.
  // Offloading N layers to the GPU costs the first N block entries plus the KV
  // cache, which turns "how many layers fit" into arithmetic on real tensor
  // sizes rather than a guess scaled from the file size.
  const layerBytes = []
  let nonLayerBytes = 0
  for (const tensor of tensors) {
    const match = /^blk\.(\d+)\./.exec(tensor.name)
    if (match) {
      const index = Number(match[1])
      layerBytes[index] = (layerBytes[index] ?? 0) + tensor.bytes
    } else {
      nonLayerBytes += tensor.bytes
    }
  }
  for (let i = 0; i < layerBytes.length; i++) layerBytes[i] ??= 0

  return {
    path: raw.filePath,
    fileSize: raw.fileSize,
    mtimeMs: raw.mtimeMs,
    ggufVersion: raw.version,
    tensorCount: raw.tensorCount,
    layerBytes,
    nonLayerBytes,
    kind: classifyKind(arch, raw.filePath, tensors),
    arch,
    name: String(metadata['general.name'] ?? ''),
    sizeLabel: String(metadata['general.size_label'] ?? ''),
    parameterCount: typeof metadata['general.parameter_count'] === 'number' ? metadata['general.parameter_count'] : undefined,
    quantization: quantFromTensors ?? (typeof fileTypeId === 'number' ? FILE_TYPE[fileTypeId] : undefined) ?? 'unknown',
    fileTypeName: typeof fileTypeId === 'number' ? FILE_TYPE[fileTypeId] : undefined,
    blockCount,
    contextLength: num('context_length'),
    embeddingLength,
    headCount,
    headCountKv,
    keyLength,
    valueLength,
    expertCount: num('expert_count'),
    expertUsedCount: num('expert_used_count'),
    feedForwardLength: num('feed_forward_length'),
    ropeFreqBase: num('rope.freq_base'),
    ropeScalingType: metadata[`${arch}.rope.scaling.type`],
    slidingWindow: num('attention.sliding_window'),
    slidingWindowPattern: num('attention.sliding_window_pattern'),
    vocabSize: typeof vocabSize === 'number' ? vocabSize : undefined,
    splitCount: typeof splitCount === 'number' ? splitCount : undefined,
    splitTensorCount: num('split.tensors.count'),
    hasChatTemplate: typeof metadata['tokenizer.chat_template'] === 'string' && metadata['tokenizer.chat_template'].length > 0,
    chatTemplate: typeof metadata['tokenizer.chat_template'] === 'string' ? metadata['tokenizer.chat_template'] : undefined,
    tensorBytes: raw.tensorBytes,
    unknownTensors: raw.unknownTensors,
    // Exact stored weight bytes, the term that dominates every memory estimate.
    weightsBytes: raw.tensorBytes > 0 ? raw.tensorBytes : raw.fileSize,
    mmproj: tensors.some((t) => t.name.startsWith('mm.') || t.name.includes('mmproj')),
    // Key/value head dimension, needed for the KV-cache formula.
    headDim: keyLength,
  }
}

/**
 * Classify a GGUF file by what it can actually be used for.
 *
 * A model directory commonly mixes three kinds of file, and only the first can
 * serve a chat request:
 *  - `llm`       — a text (or multimodal) language model.
 *  - `mmproj`    — a vision/audio projector that must be paired with an `llm`
 *                  file via `--mmproj`; on its own it has no vocabulary.
 *  - `embedding` — an encoder that produces vectors, not completions.
 */
function classifyKind(arch, filePath, tensors) {
  const base = String(filePath).replace(/\\/g, '/').split('/').pop()?.toLowerCase() ?? ''
  if (base.startsWith('mmproj') || base.includes('mmproj-')) return 'mmproj'
  // CLIP-family architectures are the projector half of a multimodal model.
  if (arch === 'clip' || arch === 'siglip' || arch === 'qwen2vl_merger') return 'mmproj'
  if (tensors.some((t) => t.name.startsWith('mm.'))) return 'mmproj'
  if (arch === 'nomic-bert' || arch === 'bert' || arch.endsWith('-bert')) return 'embedding'
  if (base.includes('embed')) return 'embedding'
  return 'llm'
}

/** Pick the quantization name with the largest byte share in the tensor table. */
function dominantQuant(tensors) {
  if (!tensors.length) return undefined
  const byType = new Map()
  for (const t of tensors) {
    // Skip the 1-D norms and biases: they stay high precision in every quant and
    // would otherwise dominate the vote for small models.
    if (t.dims.length < 2) continue
    byType.set(t.type, (byType.get(t.type) ?? 0) + t.bytes)
  }
  if (!byType.size) {
    for (const t of tensors) byType.set(t.type, (byType.get(t.type) ?? 0) + t.bytes)
  }
  let best
  let bestBytes = -1
  for (const [type, bytes] of byType) {
    if (bytes > bestBytes) {
      best = type
      bestBytes = bytes
    }
  }
  return best
}
