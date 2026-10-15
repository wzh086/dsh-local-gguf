/**
 * llama-server log parsing and the GPU/host split.
 *
 * The fixture lines are copied verbatim from a real llama.cpp 0.6.0-dev startup
 * (LM Studio's CUDA 12 runtime, 2.57.0) loading Llama-3.1-SuperNova-Lite-14B Q5_K
 * with `-ngl 55 -c 8192` on a 12 GB RTX 5070 Ti. The expected totals are what
 * llama.cpp actually allocated, so this doubles as the accuracy record for the
 * estimator.
 *
 *   node test/backend.test.mjs
 */

import { parseLlamaServerLog, BackendManager } from '../lib/backend.js'
import { estimateMemory, kvCacheBytes, weightSplit } from '../lib/estimate.js'

let failures = 0
const check = (name, ok, detail) => {
  if (!ok) failures++
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${name}${detail !== undefined && !ok ? ` — ${detail}` : ''}`)
}
const near = (actual, expected, tolerance) => Math.abs(actual - expected) <= tolerance

const MIB = 1024 * 1024
const GIB = 1024 ** 3

console.log('=== llama.cpp startup log, captured verbatim ===')
// The sizing pass reports zeros first; the real allocation follows. Both are in
// the fixture so the "keep the largest value" rule is actually exercised.
const LOG = `
0.01.137.094 I load_tensors: offloaded 55/57 layers to GPU
0.01.137.106 I load_tensors:        CUDA0 model buffer size =     0.00 MiB
0.01.137.107 I load_tensors:    CUDA_Host model buffer size =     0.00 MiB
0.01.139.646 I llama_context: n_ctx                 = 8192
0.01.140.342 I llama_context:  CUDA_Host  output buffer size =     1.96 MiB
0.01.140.517 I llama_kv_cache:        CPU KV buffer size =     0.00 MiB
0.01.140.519 I llama_kv_cache:      CUDA0 KV buffer size =     0.00 MiB
0.01.806.418 I sched_reserve:      CUDA0 compute buffer size =   158.50 MiB
0.01.806.420 I sched_reserve:  CUDA_Host compute buffer size =    40.02 MiB
0.03.384.162 I load_tensors: offloaded 55/57 layers to GPU
0.03.384.167 I load_tensors:   CPU_Mapped model buffer size =   646.44 MiB
0.03.384.168 I load_tensors:        CUDA0 model buffer size =  8341.87 MiB
0.06.319.681 I llama_context: n_ctx                 = 8192
0.06.320.198 I llama_context:  CUDA_Host  output buffer size =     1.96 MiB
0.06.320.424 I llama_kv_cache:        CPU KV buffer size =    64.00 MiB
0.06.336.063 I llama_kv_cache:      CUDA0 KV buffer size =  1728.00 MiB
0.06.737.872 I sched_reserve:      CUDA0 compute buffer size =   158.50 MiB
0.06.737.873 I sched_reserve:  CUDA_Host compute buffer size =    40.02 MiB
0.09.225.312 I slot   load_model: id  0 | task -1 | new slot, n_ctx = 8192
`

const measured = parseLlamaServerLog(LOG)
check('weights: CPU_Mapped captured', measured.weightsByDevice.CPU_Mapped === Math.round(646.44 * MIB), measured.weightsByDevice.CPU_Mapped)
check('weights: CUDA0 kept the real value, not the zeros', measured.weightsByDevice.CUDA0 === Math.round(8341.87 * MIB), measured.weightsByDevice.CUDA0)
check('KV: CPU captured', measured.kvByDevice.CPU === 64 * MIB, measured.kvByDevice.CPU)
check('KV: CUDA0 kept the real value', measured.kvByDevice.CUDA0 === 1728 * MIB, measured.kvByDevice.CUDA0)
check('compute: CUDA0 captured', measured.computeByDevice.CUDA0 === Math.round(158.5 * MIB), measured.computeByDevice.CUDA0)
check('compute: CUDA_Host captured', measured.computeByDevice.CUDA_Host === Math.round(40.02 * MIB), measured.computeByDevice.CUDA_Host)
check('offloaded layer ratio captured', measured.offloadedLayers === '55/57', measured.offloadedLayers)
check('context size captured', measured.contextSize === 8192, measured.contextSize)
check('the output buffer line is not mistaken for a KV buffer', !('CUDA_Host' in measured.kvByDevice), JSON.stringify(measured.kvByDevice))

console.log('\n=== accumulation across chunk boundaries ===')
// A split mid-line must not lose or double-count a value.
const split = parseLlamaServerLog('0.03.384.168 I load_tensors:        CUDA0 model buffer')
parseLlamaServerLog(' size =  8341.87 MiB\n', split)
check('a value split across chunks is not counted', Object.keys(split.weightsByDevice).length === 0, JSON.stringify(split.weightsByDevice))
const whole = parseLlamaServerLog('0.03.384.168 I load_tensors:        CUDA0 model buffer size =  8341.87 MiB\n')
check('the same line parsed whole is counted', whole.weightsByDevice.CUDA0 === Math.round(8341.87 * MIB))
// Larger values win, so a later smaller repeat cannot shrink the total.
parseLlamaServerLog('0.09.000.000 I load_tensors:        CUDA0 model buffer size =   100.00 MiB\n', whole)
check('a smaller later reading does not shrink the total', whole.weightsByDevice.CUDA0 === Math.round(8341.87 * MIB))

console.log('\n=== list() and describe() must agree on the measured shape ===')
{
  // Regression: `list()` once returned the raw per-device accumulator while
  // `describe()` returned the summarized view, so the same running backend read as
  // real numbers through `start` and as "unavailable" through `status`.
  // Each line is rounded to whole bytes as it is parsed, so the expectation is the
  // sum of the rounded terms rather than the rounded sum.
  const fixtureGpuBytes = Math.round(8341.87 * MIB) + Math.round(1728.0 * MIB) + Math.round(158.5 * MIB)
  const manager = new BackendManager({ log: () => {} })
  const entry = {
    modelPath: 'M:\\probe.gguf',
    child: { exitCode: null, killed: false },
    pid: 1234,
    port: 8080,
    baseUrl: 'http://127.0.0.1:8080/v1',
    backendId: 'probe',
    startedAt: 0,
    args: [],
    measuredAvailable: true,
    logTail: [],
    measured: parseLlamaServerLog(LOG),
    exit: undefined,
  }
  manager.processes.set(entry.modelPath, entry)

  const viaList = manager.list()[0].measured
  const viaDescribe = manager.describe(entry).measured
  for (const [label, measured] of [['list()', viaList], ['describe()', viaDescribe]]) {
    check(`${label} exposes totalBytes`, typeof measured.totalBytes === 'number', JSON.stringify(measured))
    check(`${label} derives the GPU total`, measured.gpuBytes === fixtureGpuBytes, `${measured.gpuBytes} vs ${fixtureGpuBytes}`)
    check(`${label} carries the device tables`, 'CUDA0' in measured.weightsByDevice)
    check(`${label} carries the offload ratio`, measured.offloadedLayers === '55/57', measured.offloadedLayers)
  }
  check('both views agree exactly', JSON.stringify(viaList) === JSON.stringify(viaDescribe))
  manager.processes.clear()
}

console.log('\n=== older llama.cpp reporting style ===')
const legacy = parseLlamaServerLog('llama_model_loader: - model size =  8341.87 MiB\nllama_kv_cache: KV self size =  1728.00 MiB\n')
check('legacy model size captured', legacy.weightsTotal === Math.round(8341.87 * MIB), legacy.weightsTotal)
check('legacy KV size captured', legacy.kvTotal === 1728 * MIB, legacy.kvTotal)

console.log('\n=== estimator accuracy against the measured values ===')
// The model the fixture was captured from, reduced to the fields the estimator
// reads. Layer bytes are distributed as the real file does (embeddings and output
// stay outside the blocks).
const layers = 56
const weightsBytes = 8.78 * GIB
const nonLayerBytes = 0.755 * GIB
const perLayer = (weightsBytes - nonLayerBytes) / layers
const model = {
  blockCount: layers,
  headCount: 32,
  headCountKv: 8,
  keyLength: 128,
  valueLength: 128,
  embeddingLength: 4096,
  contextLength: 131072,
  vocabSize: 128256,
  weightsBytes,
  nonLayerBytes,
  layerBytes: Array.from({ length: layers }, () => perLayer),
}
const config = { contextLength: 8192, gpuLayers: 55, offloadKvToGpu: true, kvTypeK: 'f16', kvTypeV: 'f16', batchSize: 2048, ubatchSize: 512, flashAttention: true }
const estimate = estimateMemory(model, config)

// Measured: 8341.87 + 1728.00 + 158.50 = 10228.37 MiB device; 646.44 + 64.00 + 40.02 host.
const measuredGpu = (8341.87 + 1728.0 + 158.5) * MIB
const measuredCpu = (646.44 + 64.0 + 40.02) * MIB
const gpuError = Math.abs(estimate.gpuBytes - measuredGpu) / measuredGpu
const cpuError = Math.abs(estimate.cpuBytes - measuredCpu) / measuredCpu

console.log(`  measured GPU ${(measuredGpu / GIB).toFixed(2)} GB   estimate ${(estimate.gpuBytes / GIB).toFixed(2)} GB   error ${(gpuError * 100).toFixed(1)}%`)
console.log(`  measured RAM ${(measuredCpu / GIB).toFixed(2)} GB   estimate ${(estimate.cpuBytes / GIB).toFixed(2)} GB   error ${(cpuError * 100).toFixed(1)}%`)

check('GPU estimate is within 10% of measured', gpuError < 0.10, `${(gpuError * 100).toFixed(1)}%`)
check('RAM estimate is within 60% of measured', cpuError < 0.60, `${(cpuError * 100).toFixed(1)}%`)
check('GPU estimate errs high (never suggests a load that will not fit)', estimate.gpuBytes >= measuredGpu, `${estimate.gpuBytes} < ${measuredGpu}`)

// The KV term is the one a user tunes, so it is checked on its own.
const measuredKv = (1728.0 + 64.0) * MIB
const kvError = Math.abs(kvCacheBytes(model, config) - measuredKv) / measuredKv
console.log(`  measured KV  ${(measuredKv / GIB).toFixed(2)} GB   estimate ${(kvCacheBytes(model, config) / GIB).toFixed(2)} GB   error ${(kvError * 100).toFixed(1)}%`)
check('KV estimate is within 5% of measured', kvError < 0.05, `${(kvError * 100).toFixed(1)}%`)

// The offload split is what decides whether a configuration fits at all.
const offload = weightSplit(model, 55)
const measuredGpuWeights = 8341.87 * MIB
const weightError = Math.abs(offload.gpuBytes - measuredGpuWeights) / measuredGpuWeights
console.log(`  measured GPU weights ${(measuredGpuWeights / GIB).toFixed(2)} GB   estimate ${(offload.gpuBytes / GIB).toFixed(2)} GB   error ${(weightError * 100).toFixed(1)}%`)
check('offloaded weight estimate is within 10% of measured', weightError < 0.10, `${(weightError * 100).toFixed(1)}%`)

console.log('')
console.log(failures === 0 ? 'ALL PASS' : `${failures} FAILURE(S)`)
process.exit(failures === 0 ? 0 : 1)
