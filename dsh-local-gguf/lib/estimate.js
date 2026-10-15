/**
 * Memory estimation and configuration recommendation for GGUF models.
 *
 * The estimator mirrors how a llama.cpp-family runtime actually spends memory:
 *
 *   total = weights + KV cache + compute buffer + runtime overhead
 *
 * Each term is computed from real GGUF tensor sizes and attention geometry rather
 * than from the file size, so partial GPU offload is arithmetic:
 * offloading `L` of `N` blocks moves the first `L` block tensors, and optionally
 * their KV cache, to the GPU.
 *
 * All results are estimates. Once a model is served by a llama.cpp backend the
 * plugin reports the backend's own measured numbers alongside them; see
 * `lib/backend.js`.
 */

import { formatBytes } from './hardware.js'

/** Bytes per stored element for every KV-cache quantization ggml accepts. */
const KV_ELEMENT_BYTES = {
  f32: 4,
  f16: 2,
  bf16: 2,
  q8_0: 34 / 32,
  q5_1: 24 / 32,
  q5_0: 22 / 32,
  q4_1: 20 / 32,
  q4_0: 18 / 32,
  iq4_nl: 18 / 32,
}

/** KV-cache types offered by the configuration surface, cheapest last. */
export const KV_CACHE_TYPES = ['f16', 'q8_0', 'q4_0']

/** Context lengths offered as presets by the configuration surface. */
export const CONTEXT_PRESETS = [4096, 8192, 16384, 32768, 65536, 131072, 262144]

/** Bytes per element for one KV-cache quantization name. */
export function kvElementBytes(type) {
  return KV_ELEMENT_BYTES[String(type).toLowerCase()] ?? KV_ELEMENT_BYTES.f16
}

/**
 * Bytes the KV cache needs for one token across every layer.
 *
 * `n_kv_head * head_dim` is the stored width of one K or V vector per layer; the
 * head count is the *KV* head count, which grouped-query models keep far below
 * their query head count. That single term explains most of the difference
 * between two models of similar weight size.
 */
export function kvBytesPerToken(gguf, { kvTypeK = 'f16', kvTypeV = 'f16' } = {}) {
  const layers = gguf.blockCount ?? 0
  const heads = gguf.headCountKv ?? gguf.headCount ?? 0
  const keyLength = gguf.keyLength ?? 0
  const valueLength = gguf.valueLength ?? keyLength
  if (!layers || !heads) return 0
  return layers * heads * (keyLength * kvElementBytes(kvTypeK) + valueLength * kvElementBytes(kvTypeV))
}

/**
 * Bytes the KV cache needs for a full context.
 *
 * Models with sliding-window attention keep a bounded cache on the windowed
 * layers. GGUF exposes the window size and, for alternating patterns, a stride;
 * the windowed share of layers is charged the window length instead of the full
 * context, which is what makes long-context MoE models like gpt-oss fit.
 */
export function kvCacheBytes(gguf, { contextLength, kvTypeK = 'f16', kvTypeV = 'f16' } = {}) {
  const perToken = kvBytesPerToken(gguf, { kvTypeK, kvTypeV })
  if (!perToken || !contextLength) return 0
  const layers = gguf.blockCount ?? 0
  const window = gguf.slidingWindow
  if (!Number.isFinite(window) || window <= 0 || window >= contextLength) {
    return perToken * contextLength
  }
  const stride = Number.isFinite(gguf.slidingWindowPattern) && gguf.slidingWindowPattern > 0 ? gguf.slidingWindowPattern : 2
  const windowedShare = stride > 1 ? (stride - 1) / stride : 0.5
  const windowedLayers = Math.round(layers * windowedShare)
  const fullLayers = layers - windowedLayers
  // A windowed layer still stores the window plus the tokens in flight; a small
  // constant covers the batch that is always resident.
  const windowedTokens = window + 512
  return perToken * ((fullLayers * contextLength + windowedLayers * Math.min(windowedTokens, contextLength)) / layers)
}

/**
 * Bytes of graph/compute scratch space a llama.cpp backend allocates.
 *
 * This scales with the logical batch and the embedding width rather than with the
 * context. The constant is calibrated against llama.cpp's reported
 * `compute buffer size` for the default 2048/512 batch on wide models; it is the
 * least precise term in the estimate and is labelled as such.
 */
export function computeBufferBytes(gguf, { batchSize = 2048, ubatchSize = 512, flashAttention = true } = {}) {
  const embedding = gguf.embeddingLength ?? 4096
  const vocab = Math.min(gguf.vocabSize ?? 32000, 262144)
  // Logits for the logical batch dominate on wide-vocabulary models.
  const logits = batchSize * vocab * 4
  const activations = batchSize * embedding * 4 * 4
  const physical = ubatchSize * embedding * 4 * 4
  const base = logits + activations + physical
  // Flash attention avoids materializing the full attention matrix, which is the
  // single largest scratch allocation otherwise.
  return Math.round(base * (flashAttention ? 0.5 : 1))
}

/** Fixed per-process overhead: runtime tables, mmap bookkeeping, CUDA context. */
export function runtimeOverheadBytes(gpuLayers) {
  // A live CUDA context costs a few hundred MiB before any weights are touched.
  return gpuLayers > 0 ? 512 * 1024 ** 2 : 128 * 1024 ** 2
}

/**
 * Split weight bytes between GPU and RAM for a partial offload.
 *
 * @param {object} gguf parsed GGUF summary.
 * @param {number} gpuLayers number of leading blocks placed in VRAM.
 */
export function weightSplit(gguf, gpuLayers) {
  const layers = gguf.blockCount ?? 0
  const layerBytes = gguf.layerBytes ?? []
  const offloaded = Math.max(0, Math.min(gpuLayers, layers))
  let gpuBytes = 0
  for (let i = 0; i < offloaded; i++) gpuBytes += layerBytes[i] ?? 0
  const totalWeights = gguf.weightsBytes ?? 0
  // Non-block tensors (token embeddings, output projection, norms) stay on the
  // CPU unless every layer is offloaded, which is where llama.cpp puts them too.
  const withNonLayer = offloaded >= layers && layers > 0
  if (withNonLayer) gpuBytes += gguf.nonLayerBytes ?? 0
  return { gpuBytes, cpuBytes: Math.max(0, totalWeights - gpuBytes), layers, offloaded }
}

/**
 * Estimate memory for one exact configuration.
 *
 * @param {object} gguf parsed GGUF summary from `readGguf`.
 * @param {object} config load configuration; see `defaultLoadConfig`.
 * @returns {object} a breakdown that sums to `totalBytes`, plus the GPU/RAM split.
 */
export function estimateMemory(gguf, config = {}) {
  const {
    contextLength = gguf.contextLength ?? 8192,
    kvTypeK = 'f16',
    kvTypeV = 'f16',
    gpuLayers = gguf.blockCount ?? 0,
    offloadKvToGpu = gpuLayers > 0,
    batchSize = 2048,
    ubatchSize = 512,
    flashAttention = true,
    mmap = true,
  } = config

  const layers = gguf.blockCount ?? 0
  const offloaded = Math.max(0, Math.min(gpuLayers, layers))
  const weights = weightSplit(gguf, offloaded)
  const kvTotal = kvCacheBytes(gguf, { contextLength, kvTypeK, kvTypeV })
  // The KV cache follows its layers: with a partial offload only the offloaded
  // share lives in VRAM, unless the user keeps the cache on the CPU entirely.
  const layerShare = layers > 0 ? offloaded / layers : 0
  const kvOnGpu = offloadKvToGpu && offloaded > 0
  const kvGpu = kvOnGpu ? kvTotal * layerShare : 0
  const kvCpu = kvTotal - kvGpu

  const compute = computeBufferBytes(gguf, { batchSize, ubatchSize, flashAttention })
  const computeTotal = compute + runtimeOverheadBytes(offloaded)
  const computeGpu = offloaded > 0 ? computeTotal : 0
  const computeCpu = computeTotal - computeGpu

  const gpuBytes = weights.gpuBytes + kvGpu + computeGpu
  const cpuBytes = weights.cpuBytes + kvCpu + computeCpu
  return {
    weightsBytes: weights.gpuBytes + weights.cpuBytes,
    weightsGpuBytes: weights.gpuBytes,
    weightsCpuBytes: weights.cpuBytes,
    kvCacheBytes: kvTotal,
    kvGpuBytes: kvGpu,
    kvCpuBytes: kvCpu,
    computeBytes: computeTotal,
    gpuBytes,
    cpuBytes,
    totalBytes: gpuBytes + cpuBytes,
    mmap,
    config: { contextLength, kvTypeK, kvTypeV, gpuLayers: offloaded, offloadKvToGpu, batchSize, ubatchSize, flashAttention },
  }
}

/**
 * Largest block count that keeps the GPU side inside a VRAM budget.
 *
 * Evaluated per layer rather than by ratio because block sizes differ (the first
 * and last blocks of MoE models can be several times the average), and because a
 * ratio would happily offload a block that does not fit.
 */
export function maxGpuLayers(gguf, vramBudgetBytes, config = {}) {
  const layers = gguf.blockCount ?? 0
  if (!layers) return 0
  for (let candidate = layers; candidate >= 0; candidate--) {
    const estimate = estimateMemory(gguf, { ...config, gpuLayers: candidate })
    if (estimate.gpuBytes <= vramBudgetBytes) return candidate
  }
  return 0
}

/** The load configuration the surfaces start from before any tuning. */
export function defaultLoadConfig(gguf) {
  const native = gguf.contextLength ?? 8192
  // llama.cpp's own defaults; both are good for prompt throughput on desktop GPUs.
  return {
    contextLength: Math.min(native, 8192),
    kvTypeK: 'f16',
    kvTypeV: 'f16',
    gpuLayers: gguf.blockCount ?? 0,
    offloadKvToGpu: true,
    batchSize: 2048,
    ubatchSize: 512,
    flashAttention: true,
    mmap: true,
  }
}

/** Round a context length down to a conventional preset, keeping the native cap. */
function snapContext(value, native) {
  const capped = Math.min(value, native)
  let best = CONTEXT_PRESETS[0]
  for (const preset of CONTEXT_PRESETS) {
    if (preset <= capped) best = preset
  }
  return Math.min(best, native)
}

/**
 * Recommend a complete load configuration for one model on one machine, and
 * explain the tradeoffs the numbers encode.
 *
 * The strategy is the one a person would follow by hand:
 *  1. Reserve a slice of VRAM for the desktop compositor and other apps.
 *  2. Offload as many blocks as fit, keeping the KV cache on the GPU while it
 *     still fits, because that is what makes long contexts fast.
 *  3. Spend whatever VRAM remains on context length.
 *  4. If the model cannot fit in VRAM at all, fall back to CPU execution with
 *     thread and batch settings sized to the machine.
 *
 * @returns {{config: object, estimate: object, rationale: string[], warnings: string[],
 *   rationaleNotes: Array<{key: string, params: object}>,
 *   warningNotes: Array<{key: string, params: object}>}}
 */
export function recommendConfig(gguf, hardware, options = {}) {
  const rationale = []
  const warnings = []
  /** Machine-readable companions to the prose, for localized surfaces. */
  const rationaleNotes = []
  const warningNotes = []
  const native = gguf.contextLength ?? 8192
  const layers = gguf.blockCount ?? 0

  const totalVram = hardware?.totalVramBytes ?? 0
  const cpu = hardware?.cpu ?? { logicalCores: 8, physicalCores: 4, recommendedThreads: 4 }
  const ramTotal = hardware?.memory?.totalBytes ?? 16 * 1024 ** 3

  // Leave room for the desktop compositor, the browser and the harness itself.
  const reserveRatio = options.vramReserveRatio ?? 0.08
  const reserveBytes = options.vramReserveBytes ?? Math.max(768 * 1024 ** 2, Math.round(totalVram * reserveRatio))
  const vramBudget = Math.max(0, totalVram - reserveBytes)

  const base = {
    ...defaultLoadConfig(gguf),
    contextLength: Math.min(native, 8192),
  }

  /**
   * Record one explanatory line.
   *
   * Each line is emitted twice: as English prose for the agent tools, which read
   * it in a transcript, and as a machine key with its parameters so the Models-page
   * panel can render the same statement in the user's own language. Keeping one
   * source avoids the two drifting apart.
   */
  const say = (bucket, key, params, text) => {
    if (bucket === 'rationale') {
      rationale.push(text)
      rationaleNotes.push({ key, params })
    } else {
      warnings.push(text)
      warningNotes.push({ key, params })
    }
  }

  let config = { ...base }

  if (totalVram === 0 || vramBudget <= 0) {
    // CPU-only host: everything stays in RAM, so threads and batch size are the
    // only levers, and a smaller batch keeps interactive latency acceptable.
    config = {
      ...base,
      gpuLayers: 0,
      offloadKvToGpu: false,
      flashAttention: false,
      batchSize: 512,
      ubatchSize: 256,
      contextLength: snapContext(Math.min(native, 8192), native),
    }
    say('rationale', 'noGpu', {}, 'No GPU was detected, so the model runs on the CPU with all layers in system memory.')
    say('rationale', 'cpuThreads', { threads: cpu.recommendedThreads }, `Using ${cpu.recommendedThreads} generation threads, matching the physical core count.`)
    if (gguf.weightsBytes > ramTotal * 0.8) {
      say('warning', 'weightsNearRam', { weights: formatBytes(gguf.weightsBytes), ram: formatBytes(ramTotal) },
        `Weights (${formatBytes(gguf.weightsBytes)}) are close to total system memory (${formatBytes(ramTotal)}); expect swapping.`)
    }
  } else {
    const gpu = hardware.gpus.find((g) => g.vendor === 'nvidia') ?? hardware.gpus[0]
    say('rationale', 'vramReserve', { reserve: formatBytes(reserveBytes), total: formatBytes(totalVram) },
      `Reserving ${formatBytes(reserveBytes)} of ${formatBytes(totalVram)} VRAM for the desktop and other applications.`)

    // Step 1: layers with the KV cache on the GPU.
    let offloadKv = true
    let gpuLayers = maxGpuLayers(gguf, vramBudget, { ...config, offloadKvToGpu: true, contextLength: config.contextLength })
    if (gpuLayers === 0) {
      // Weights alone do not fit; try again letting the cache stay in RAM, which
      // still wins because weights dominate the per-token cost.
      offloadKv = false
      gpuLayers = maxGpuLayers(gguf, vramBudget, { ...config, offloadKvToGpu: false, contextLength: config.contextLength })
    }
    config = { ...config, gpuLayers, offloadKvToGpu: offloadKv }

    if (gpuLayers >= layers && layers > 0) {
      say('rationale', 'allLayersFit', { layers }, `All ${layers} layers fit in VRAM, so the whole model is GPU-resident.`)
    } else if (gpuLayers > 0) {
      say('rationale', 'partialOffload', { offloaded: gpuLayers, layers }, `Offloading ${gpuLayers} of ${layers} layers to the GPU; the rest run on the CPU.`)
      say('warning', 'partialOffloadSlow', {}, 'Partial offload splits every token across PCIe; expect lower throughput than a fully GPU-resident model.')
    } else {
      say('rationale', 'noFit', {}, 'The weights do not fit in VRAM even with the cache in RAM, so the model runs on the CPU.')
      say('warning', 'smallerQuant', {}, 'Consider a smaller quantization (for example Q4_K_M) to fit this model in VRAM.')
    }

    // Step 2: spend remaining VRAM on context, growing only while it fits.
    let context = snapContext(config.contextLength, native)
    for (const preset of CONTEXT_PRESETS) {
      if (preset <= context) continue
      if (preset > native) break
      const candidate = estimateMemory(gguf, { ...config, contextLength: preset })
      if (candidate.gpuBytes <= vramBudget) context = preset
      else break
    }
    if (context > config.contextLength) {
      say('rationale', 'raisedContext', { context: context.toLocaleString() }, `Raised the context to ${context.toLocaleString()} tokens, which still fits the VRAM budget.`)
    }
    config = { ...config, contextLength: context }

    // Step 3: quantize the KV cache only if it buys a materially longer context.
    const full = estimateMemory(gguf, config)
    if (full.gpuBytes > vramBudget && kvElementBytes(config.kvTypeK) > kvElementBytes('q8_0')) {
      const quantized = estimateMemory(gguf, { ...config, kvTypeK: 'q8_0', kvTypeV: 'q8_0' })
      if (quantized.gpuBytes <= vramBudget) {
        config = { ...config, kvTypeK: 'q8_0', kvTypeV: 'q8_0' }
        say('rationale', 'quantizedKv', {}, 'Quantized the KV cache to q8_0 to bring the estimate inside the VRAM budget.')
        say('warning', 'kvQuality', {}, 'q8_0 KV cache slightly reduces long-context quality; f16 is preferable when memory allows.')
      }
    }

    // Flash attention is what makes a quantized cache worthwhile and is supported
    // on every modern NVIDIA, AMD and Intel backend.
    config = { ...config, flashAttention: true }
    if (gpu?.vramBytesEstimated) {
      say('warning', 'estimatedVram', { gpu: gpu.name },
        `VRAM for "${gpu.name}" was read from a 32-bit registry value and may understate cards above 4 GB.`)
    }
  }

  // Batch defaults: the logical batch drives prompt throughput, the physical
  // batch drives peak scratch memory.
  if (config.gpuLayers === 0) {
    config.batchSize = Math.min(config.batchSize, 1024)
    config.ubatchSize = Math.min(config.ubatchSize, 256)
  }

  const estimate = estimateMemory(gguf, config)
  const ramBudget = ramTotal * 0.9
  if (estimate.cpuBytes > ramBudget) {
    say('warning', 'ramOverflow', { used: formatBytes(estimate.cpuBytes), ram: formatBytes(ramTotal) },
      `Estimated system memory use (${formatBytes(estimate.cpuBytes)}) exceeds 90% of installed RAM (${formatBytes(ramTotal)}).`)
  }
  if (gguf.expertCount && gguf.expertUsedCount && config.gpuLayers > 0 && config.gpuLayers < layers) {
    say('rationale', 'moeAdvice', { used: gguf.expertUsedCount, total: gguf.expertCount },
      `This is a mixture-of-experts model (${gguf.expertUsedCount} of ${gguf.expertCount} experts per token); keeping the expert weights on the CPU with --n-cpu-moe usually beats offloading whole layers.`)
  }

  return { config, estimate, rationale, warnings, rationaleNotes, warningNotes }
}

/** Describe a load configuration in llama.cpp flag terms, for the CLI and the UI. */
export function toLlamaServerArgs(gguf, config, { modelPath, alias, port, host = '127.0.0.1' } = {}) {
  const args = ['--model', modelPath]
  if (alias) args.push('--alias', alias)
  args.push('--host', String(host), '--port', String(port))
  args.push('-c', String(config.contextLength))
  args.push('-ngl', String(config.gpuLayers))
  if (config.gpuLayers === 0) args.push('--device', 'none')
  args.push('-t', String(config.threads ?? 0))
  args.push('-b', String(config.batchSize))
  args.push('-ub', String(config.ubatchSize))
  args.push('-fa', config.flashAttention ? 'on' : 'off')
  if (config.kvTypeK && config.kvTypeK !== 'f16') args.push('-ctk', config.kvTypeK)
  if (config.kvTypeV && config.kvTypeV !== 'f16') args.push('-ctv', config.kvTypeV)
  if (config.offloadKvToGpu === false) args.push('--no-kv-offload')
  if (config.mmap === false) args.push('--no-mmap')
  if (config.mlock) args.push('--mlock')
  if (Number.isInteger(config.seed) && config.seed >= 0) args.push('--seed', String(config.seed))
  if (Number.isInteger(config.cpuMoeLayers) && config.cpuMoeLayers > 0) args.push('--n-cpu-moe', String(config.cpuMoeLayers))
  if (config.unifiedKv === false) args.push('--no-kv-unified')
  if (Number.isInteger(config.parallel) && config.parallel > 0) args.push('-np', String(config.parallel))
  if (Number.isFinite(config.ropeFreqBase)) args.push('--rope-freq-base', String(config.ropeFreqBase))
  if (Number.isFinite(config.ropeFreqScale)) args.push('--rope-freq-scale', String(config.ropeFreqScale))
  if (Number.isInteger(config.contextCheckpoints) && config.contextCheckpoints > 0) args.push('--ctx-checkpoints', String(config.contextCheckpoints))
  args.push('--jinja')
  // llama.cpp logs its real per-device memory accounting at trace level only, and
  // that log is what turns the estimate into a measured figure. Level 4 adds the
  // allocation lines while keeping startup near 200 lines instead of the ~8000 full
  // verbosity produces, and it costs about 34 lines for a whole generation.
  if (config.verboseLogs !== false) args.push('-lv', '4')
  if (config.extraArgs) args.push(...config.extraArgs)
  return args
}
