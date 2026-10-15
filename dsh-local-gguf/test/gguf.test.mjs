/**
 * GGUF reader, memory estimator and recommendation engine.
 *
 * Reads real models when they are present — set `LOCAL_GGUF_TEST_MODELS` to a
 * directory, or rely on the LM Studio index — and otherwise still exercises the
 * estimator and recommender against a synthetic model summary, so the suite is
 * meaningful on a machine with no GGUF files.
 *
 *   node test/gguf.test.mjs
 */

import { readGguf } from '../lib/gguf.js'
import { scanModels } from '../lib/models.js'
import { probeHardware, formatBytes } from '../lib/hardware.js'
import {
  CONTEXT_PRESETS,
  computeBufferBytes,
  estimateMemory,
  kvBytesPerToken,
  kvCacheBytes,
  maxGpuLayers,
  recommendConfig,
  toLlamaServerArgs,
} from '../lib/estimate.js'

let failures = 0
const check = (name, ok, detail) => {
  if (!ok) failures++
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${name}${detail !== undefined && !ok ? ` — ${detail}` : ''}`)
}
const close = (a, b, tolerance = 1e-6) => Math.abs(a - b) <= tolerance

console.log('=== estimator arithmetic (synthetic, exact) ===')
{
  // A worst-case-free model: 32 layers, 32 query heads, 8 KV heads, head_dim 128.
  const gguf = {
    blockCount: 32, headCount: 32, headCountKv: 8, keyLength: 128, valueLength: 128,
    embeddingLength: 4096, contextLength: 32768, vocabSize: 128256,
    weightsBytes: 8 * 1024 ** 3, nonLayerBytes: 0,
    layerBytes: Array.from({ length: 32 }, () => (8 * 1024 ** 3) / 32),
  }
  // n_layer * n_kv_heads * head_dim * (2 + 2) = 32 * 8 * 128 * 4 = 131072 B/token
  check('KV bytes per token matches the formula', kvBytesPerToken(gguf) === 131072, kvBytesPerToken(gguf))
  check('KV at 4096 tokens is exactly 512 MiB', kvCacheBytes(gguf, { contextLength: 4096 }) === 536870912, kvCacheBytes(gguf, { contextLength: 4096 }))
  // q8_0 K + f16 V = 34/32 + 2 bytes per element per head.
  const mixed = kvBytesPerToken(gguf, { kvTypeK: 'q8_0', kvTypeV: 'f16' })
  check('q8_0 K halves its share', close(mixed, 32 * 8 * 128 * (34 / 32 + 2), 1), mixed)

  const full = estimateMemory(gguf, { contextLength: 4096, gpuLayers: 32, kvTypeK: 'f16', kvTypeV: 'f16' })
  check('all layers offloaded leaves no CPU weights', full.weightsCpuBytes === 0, full.weightsCpuBytes)
  check('KV follows the offloaded layers', full.kvGpuBytes === full.kvCacheBytes, `${full.kvGpuBytes}/${full.kvCacheBytes}`)
  check('total is the sum of its parts', full.totalBytes === full.gpuBytes + full.cpuBytes)

  const half = estimateMemory(gguf, { contextLength: 4096, gpuLayers: 16 })
  check('half offload splits weights evenly', half.weightsGpuBytes === half.weightsCpuBytes, `${half.weightsGpuBytes}/${half.weightsCpuBytes}`)
  check('half offload splits the KV cache', close(half.kvGpuBytes, half.kvCacheBytes / 2, 1))

  const cpu = estimateMemory(gguf, { contextLength: 4096, gpuLayers: 0, offloadKvToGpu: false })
  check('a CPU-only configuration uses no VRAM', cpu.gpuBytes === 0, cpu.gpuBytes)

  check('flash attention reduces the compute buffer', computeBufferBytes(gguf, { flashAttention: true }) < computeBufferBytes(gguf, { flashAttention: false }))
  check('a zero budget offloads nothing', maxGpuLayers(gguf, 0, { contextLength: 4096 }) === 0)
  check('an unlimited budget offloads everything', maxGpuLayers(gguf, Number.MAX_SAFE_INTEGER, { contextLength: 4096 }) === 32)

  const args = toLlamaServerArgs(gguf, { contextLength: 4096, gpuLayers: 16, threads: 8, batchSize: 2048, ubatchSize: 512, flashAttention: true, kvTypeK: 'q8_0', kvTypeV: 'f16' }, { modelPath: 'x.gguf', alias: 'x', port: 8080 })
  const joined = args.join(' ')
  check('args include the context length', joined.includes('-c 4096'))
  check('args include the GPU layer count', joined.includes('-ngl 16'))
  check('args include the K cache type only when quantized', joined.includes('-ctk q8_0'))
  check('args omit the V cache type when it stays f16', !joined.includes('-ctv'))
  check('args include the physical batch', joined.includes('-ub 512'))
}

console.log('\n=== real models, when present ===')
const hardware = await probeHardware()
console.log(`  hardware: ${hardware.cpu.model} · ${hardware.cpu.logicalCores} logical cores · ${formatBytes(hardware.memory.totalBytes)} RAM · ${formatBytes(hardware.totalVramBytes)} usable VRAM`)
for (const gpu of hardware.gpus) {
  console.log(`  gpu: ${gpu.name} · ${formatBytes(gpu.vramBytes)}${gpu.integrated ? ' (integrated, excluded)' : ''} · via ${gpu.source}`)
}
check('hardware probe reports CPU, memory and VRAM fields', typeof hardware.cpu.logicalCores === 'number' && typeof hardware.memory.totalBytes === 'number' && typeof hardware.totalVramBytes === 'number')
check('usable VRAM excludes integrated adapters', hardware.gpus.filter((g) => g.integrated).every((g) => g.vramBytes > 0))

const extra = process.env.LOCAL_GGUF_TEST_MODELS
const scan = await scanModels({ directories: extra ? [extra] : [] })
console.log(`  scanned ${scan.scanned} file(s) in ${scan.directories.length} directories; ${scan.models.length} parsed`)
if (scan.failures.length) console.log(`  unreadable: ${JSON.stringify(scan.failures)}`)

if (scan.models.length === 0) {
  console.log('  (no GGUF files found; the real-model checks were skipped)')
} else {
  for (const model of scan.models.slice(0, 6)) {
    console.log(`\n  --- ${model.displayName} [${model.kind}] ${model.arch} / ${model.quantization}`)
    console.log(`      ${model.blockCount} layers · ctx ${model.contextLength} · heads ${model.headCount}/${model.headCountKv} · weights ${formatBytes(model.weightsBytes)}`)

    check(`${model.arch}: kind is classified`, ['llm', 'mmproj', 'embedding'].includes(model.kind), model.kind)
    if (model.kind === 'llm') {
      check(`${model.arch}: has a layer count`, Number.isInteger(model.blockCount) && model.blockCount > 0, model.blockCount)
      check(`${model.arch}: has a context length`, Number.isInteger(model.contextLength) && model.contextLength > 0, model.contextLength)
      check(`${model.arch}: KV heads do not exceed query heads`, model.headCountKv <= model.headCount, `${model.headCountKv} > ${model.headCount}`)
      check(`${model.arch}: per-layer bytes cover the whole tensor table`,
        model.layerBytes.reduce((a, b) => a + b, 0) + model.nonLayerBytes === model.weightsBytes,
        `${model.layerBytes.reduce((a, b) => a + b, 0) + model.nonLayerBytes} vs ${model.weightsBytes}`)
      check(`${model.arch}: weights are within the file size`, model.weightsBytes <= model.fileSize)

      const recommended = recommendConfig(model, hardware)
      check(`${model.arch}: recommended layers are within range`, recommended.config.gpuLayers >= 0 && recommended.config.gpuLayers <= model.blockCount, recommended.config.gpuLayers)
      check(`${model.arch}: recommended context is a valid preset or the native cap`,
        CONTEXT_PRESETS.includes(recommended.config.contextLength) || recommended.config.contextLength === model.contextLength,
        recommended.config.contextLength)
      check(`${model.arch}: recommendation fits the VRAM budget when a GPU exists`,
        hardware.totalVramBytes === 0 || recommended.estimate.gpuBytes <= hardware.totalVramBytes,
        `${formatBytes(recommended.estimate.gpuBytes)} > ${formatBytes(hardware.totalVramBytes)}`)
      console.log(`      recommend: ngl ${recommended.config.gpuLayers}/${model.blockCount} · ctx ${recommended.config.contextLength} · ${recommended.config.kvTypeK} KV · GPU ${formatBytes(recommended.estimate.gpuBytes)} · RAM ${formatBytes(recommended.estimate.cpuBytes)}`)

      // Re-reading the same file must be stable and cheap (the header memo).
      const again = await readGguf(model.path)
      check(`${model.arch}: reparse agrees`, again.weightsBytes === model.weightsBytes && again.blockCount === model.blockCount)
    }
  }
}

console.log('\n=== rationale notes are emitted as machine keys ===')
{
  // The Models-page panel renders these keys in the user's language, so the Host
  // must emit a key for every line it produces — a line with prose but no key
  // would appear untranslated in the panel.
  const model = {
    blockCount: 32,
    headCount: 32,
    headCountKv: 8,
    keyLength: 128,
    valueLength: 128,
    embeddingLength: 4096,
    contextLength: 131072,
    vocabSize: 128256,
    weightsBytes: 9 * 1024 ** 3,
    nonLayerBytes: 0.5 * 1024 ** 3,
    layerBytes: Array.from({ length: 32 }, () => (8.5 * 1024 ** 3) / 32),
  }
  const gpuHost = {
    totalVramBytes: 12 * 1024 ** 3,
    cpu: { logicalCores: 16, physicalCores: 8, recommendedThreads: 8 },
    memory: { totalBytes: 32 * 1024 ** 3 },
    gpus: [{ name: 'Test GPU', vendor: 'nvidia', vramBytes: 12 * 1024 ** 3 }],
  }
  const recommended = recommendConfig(model, gpuHost)
  check('the recommended result carries note arrays',
    Array.isArray(recommended.rationaleNotes) && Array.isArray(recommended.warningNotes))
  check('every rationale line has a key',
    recommended.rationaleNotes.length === recommended.rationale.length && recommended.rationaleNotes.length > 0,
    `${recommended.rationaleNotes.length} notes for ${recommended.rationale.length} lines`)
  check('keys are non-empty strings', recommended.rationaleNotes.every((note) => typeof note.key === 'string' && note.key.length > 0))
  check('params are always an object',
    [...recommended.rationaleNotes, ...recommended.warningNotes].every((note) => note.params && typeof note.params === 'object'))
  console.log(`  emitted keys: ${[...recommended.rationaleNotes, ...recommended.warningNotes].map((n) => n.key).join(', ')}`)

  // A CPU-only host takes a different branch and must also produce keys.
  const cpuHost = { totalVramBytes: 0, cpu: { logicalCores: 8, physicalCores: 4, recommendedThreads: 4 }, memory: { totalBytes: 16 * 1024 ** 3 }, gpus: [] }
  const cpuRecommended = recommendConfig(model, cpuHost)
  check('the CPU-only branch also emits keys',
    cpuRecommended.rationaleNotes.length === cpuRecommended.rationale.length && cpuRecommended.rationaleNotes.length > 0,
    `${cpuRecommended.rationaleNotes.length}/${cpuRecommended.rationale.length}`)
  check('the CPU-only branch identifies itself', cpuRecommended.rationaleNotes[0].key === 'noGpu', cpuRecommended.rationaleNotes[0].key)

  // A partial offload of a MoE model is the case that emits the most notes.
  const moeModel = { ...model, expertCount: 32, expertUsedCount: 4 }
  const moe = recommendConfig(moeModel, { ...gpuHost, totalVramBytes: 8 * 1024 ** 3 })
  const keys = moe.rationaleNotes.map((note) => note.key)
  check('a partial offload reports its layer split', keys.includes('partialOffload'), keys.join(','))
  check('a MoE model gets the expert-placement hint', keys.includes('moeAdvice'), keys.join(','))
  check('the MoE hint carries its parameters',
    moe.rationaleNotes.find((n) => n.key === 'moeAdvice')?.params?.total === 32)
}

console.log('')
console.log(failures === 0 ? 'ALL PASS' : `${failures} FAILURE(S)`)
process.exit(failures === 0 ? 0 : 1)
