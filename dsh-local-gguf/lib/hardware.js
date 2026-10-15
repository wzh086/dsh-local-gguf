/**
 * Host hardware probe used by the memory estimator and the recommendation engine.
 *
 * Everything here is best-effort: a sandboxed or locked-down host can deny any
 * one source, so each fact is discovered through an ordered list of fallbacks and
 * reported as `undefined` rather than throwing. The engine degrades to
 * conservative assumptions when a fact is missing.
 *
 * Sources, in order of preference:
 *  - CPU/RAM: Node's own `os` module (always available).
 *  - GPU: `nvidia-smi` (NVIDIA), then the Windows display-class registry, then
 *    a performance-counter read of the GPU adapter memory.
 */

import os from 'node:os'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'

const run = promisify(execFile)

/** Run a command and return trimmed stdout, or undefined on any failure. */
async function tryExec(file, args, timeout = 4000) {
  try {
    const { stdout } = await run(file, args, { timeout, windowsHide: true, maxBuffer: 4 << 20 })
    return stdout.trim()
  } catch {
    return undefined
  }
}

/**
 * Read total VRAM of every NVIDIA GPU.
 *
 * `nvidia-smi` is preferred because it reports the real, driver-visible VRAM
 * rather than the 32-bit-truncated value some Windows APIs return for large
 * cards.
 */
async function nvidiaGpus() {
  const out = await tryExec('nvidia-smi', [
    '--query-gpu=index,name,memory.total,memory.free,compute_cap',
    '--format=csv,noheader,nounits',
  ])
  if (!out) return undefined
  const gpus = []
  for (const line of out.split(/\r?\n/)) {
    const parts = line.split(',').map((s) => s.trim())
    if (parts.length < 5) continue
    const totalMiB = Number(parts[2])
    const freeMiB = Number(parts[3])
    if (!Number.isFinite(totalMiB)) continue
    gpus.push({
      index: Number(parts[0]),
      name: parts[1],
      vendor: 'nvidia',
      vramBytes: Math.round(totalMiB * 1024 * 1024),
      vramFreeBytes: Number.isFinite(freeMiB) ? Math.round(freeMiB * 1024 * 1024) : undefined,
      computeCapability: parts[4],
      source: 'nvidia-smi',
    })
  }
  return gpus.length ? gpus : undefined
}

/**
 * Read GPU adapters from the Windows display class registry.
 *
 * This covers AMD and Intel adapters, for which no portable CLI exists. The
 * registry stores VRAM as a QWORD under `HardwareInformation.qwMemorySize`; the
 * older 32-bit `MemorySize` is used only as a fallback because it wraps above
 * 4 GiB, and it is reported as `vramBytesEstimated` so callers can say so.
 */
async function windowsRegistryGpus() {
  if (process.platform !== 'win32') return undefined
  const script = `
$base = 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}'
$out = @()
Get-ChildItem $base -ErrorAction SilentlyContinue | ForEach-Object {
  $p = Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue
  if ($p.DriverDesc) {
    $qw = $null
    if ($p.'HardwareInformation.qwMemorySize') { $qw = [uint64]$p.'HardwareInformation.qwMemorySize' }
    $ms = $null
    if ($p.'HardwareInformation.MemorySize') { $ms = [uint64]$p.'HardwareInformation.MemorySize' }
    $out += [pscustomobject]@{ name = $p.DriverDesc; qw = $qw; ms = $ms }
  }
}
$out | ConvertTo-Json -Compress
`
  const out = await tryExec('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', script], 8000)
  if (!out) return undefined
  let parsed
  try {
    parsed = JSON.parse(out)
  } catch {
    return undefined
  }
  const rows = Array.isArray(parsed) ? parsed : [parsed]
  const gpus = []
  for (const row of rows) {
    if (!row?.name) continue
    let bytes
    let estimated = false
    if (typeof row.qw === 'number' && row.qw > 0) bytes = row.qw
    else if (typeof row.ms === 'number' && row.ms > 0) {
      bytes = row.ms
      estimated = true
    }
    if (bytes === undefined) continue
    gpus.push({ name: String(row.name), vendor: vendorOf(String(row.name)), vramBytes: bytes, vramBytesEstimated: estimated, source: 'registry' })
  }
  return gpus.length ? gpus : undefined
}

/** Guess the GPU vendor from its marketing name. */
function vendorOf(name) {
  const n = name.toLowerCase()
  if (n.includes('nvidia') || n.includes('geforce') || n.includes('quadro') || n.includes('rtx') || n.includes('tesla')) return 'nvidia'
  if (n.includes('amd') || n.includes('radeon') || n.includes('ati ')) return 'amd'
  if (n.includes('intel') || n.includes('arc ') || n.includes('iris') || n.includes('uhd graphics')) return 'intel'
  return 'unknown'
}

/**
 * Logical processor count, physical-core estimate, and a usable thread
 * recommendation.
 *
 * The physical count is estimated at `logical / 2` on x86/x64 with SMT, which is
 * what `llama.cpp` targets by default; `os.cpus()` exposes only logical
 * processors on Windows. The estimate is flagged so the UI can say `~`.
 */
function cpuFacts() {
  const cpus = os.cpus()
  const logical = cpus.length || 1
  const model = cpus[0]?.model?.trim() ?? 'unknown'
  const arch = process.arch
  const smtLikely = (arch === 'x64' || arch === 'ia32') && logical > 4
  const physical = smtLikely ? Math.max(1, Math.round(logical / 2)) : logical
  return {
    model,
    arch,
    logicalCores: logical,
    physicalCores: physical,
    physicalCoresEstimated: smtLikely,
    // llama.cpp's own default is the physical core count; recommending more
    // threads than physical cores usually costs throughput.
    recommendedThreads: physical,
  }
}

/**
 * Total and available system memory.
 *
 * `os.totalmem()` reports the memory usable by the OS, which is the right basis
 * for a model-fitting decision on Windows.
 */
function memoryFacts() {
  return {
    totalBytes: os.totalmem(),
    freeBytes: os.freemem(),
  }
}

/**
 * Read the GPU survey LM Studio itself recorded.
 *
 * LM Studio probes the machine when it installs an inference runtime and stores
 * the result under `.lmstudio/.internal/internal-engine-index.json`. That is a
 * plain file read, so it succeeds even where spawning `nvidia-smi` is denied, and
 * it is the most accurate source for the VRAM a llama.cpp backend will actually
 * see.
 *
 * Two traps in that data are handled here, because both would otherwise produce a
 * recommendation that cannot load:
 *  - `totalMemoryCapacityBytes` counts memory shared with the host under Vulkan
 *    (27.5 GB for a 12 GB card), so the *dedicated* figure is preferred.
 *  - Integrated adapters report system RAM as their dedicated memory and are
 *    therefore flagged rather than counted as usable VRAM.
 * The same physical adapter is reported once per runtime, so entries are merged
 * by device UUID with the CUDA reading preferred.
 */
async function lmStudioGpuSurvey() {
  const { lmStudioHome } = await import('./lmstudio.js')
  const file = `${lmStudioHome()}/.internal/internal-engine-index.json`
  let parsed
  try {
    parsed = JSON.parse(await (await import('node:fs')).promises.readFile(file, 'utf8'))
  } catch {
    return undefined
  }
  const engines = Array.isArray(parsed?.json) ? parsed.json : []
  const best = new Map()
  for (const engine of engines) {
    const info = engine?.hardwareSurveyResult?.gpuSurveyResult?.gpuInfo
    if (!Array.isArray(info)) continue
    for (const gpu of info) {
      const name = gpu?.name
      if (typeof name !== 'string') continue
      const dedicated = num(gpu?.dedicatedMemoryCapacityBytes)
      const total = num(gpu?.totalMemoryCapacityBytes)
      const bytes = dedicated ?? total
      if (!bytes) continue
      const integrated = String(gpu?.integrationType ?? '').toLowerCase() === 'integrated'
      const viaCuda = String(gpu?.detectionPlatform ?? '').toUpperCase() === 'CUDA'
      const key = gpu?.otherInfo?.deviceUUID ?? name
      const candidate = {
        name,
        vendor: vendorOf(name),
        vramBytes: bytes,
        vramSharedBytes: total !== undefined && dedicated !== undefined && total > dedicated ? total : undefined,
        integrated,
        computeCapability: gpu?.otherInfo?.computeCapability ?? gpu?.otherInfo?.cudaComputeCapability ?? undefined,
        driverVersion: gpu?.otherInfo?.driverVersion ?? gpu?.otherInfo?.driverInfo ?? undefined,
        source: 'lmstudio-survey',
        detectionPlatform: gpu?.detectionPlatform,
      }
      const previous = best.get(key)
      if (!previous || betterGpuReading(candidate, previous)) best.set(key, candidate)
    }
  }
  return best.size ? [...best.values()] : undefined
}

/** Prefer a CUDA reading, then dedicated memory that is not shared. */
function betterGpuReading(candidate, previous) {
  const candCuda = String(candidate.detectionPlatform ?? '').toUpperCase() === 'CUDA'
  const prevCuda = String(previous.detectionPlatform ?? '').toUpperCase() === 'CUDA'
  if (candCuda !== prevCuda) return candCuda
  const candExact = candidate.vramSharedBytes === undefined
  const prevExact = previous.vramSharedBytes === undefined
  if (candExact !== prevExact) return candExact
  return candidate.vramBytes > previous.vramBytes
}

/** Coerce a survey field to a positive finite number. */
function num(value) {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : undefined
}

/**
 * Probe the whole machine once.
 *
 * @returns {Promise<object>} a detached, JSON-serializable hardware report.
 */
export async function probeHardware() {
  const cpu = cpuFacts()
  const memory = memoryFacts()
  // Ordered by accuracy and by how likely the source is to be permitted:
  // nvidia-smi reports live free memory, LM Studio's survey is a plain file read.
  let gpus = (await nvidiaGpus()) ?? (await lmStudioGpuSurvey()) ?? (await windowsRegistryGpus()) ?? []
  // A machine with no discrete GPU still runs GGUF models on the CPU, so an
  // empty list is a valid result rather than an error. Integrated adapters stay
  // in the report for display, but their memory is system RAM and is never
  // counted as a VRAM budget.
  const discrete = gpus.filter((gpu) => !gpu.integrated)
  const totalVram = discrete.reduce((sum, gpu) => sum + (gpu.vramBytes ?? 0), 0)
  return {
    platform: process.platform,
    arch: process.arch,
    cpu,
    memory,
    gpus,
    totalVramBytes: totalVram,
    hasCuda: discrete.some((g) => g.vendor === 'nvidia'),
    probedAt: Date.now(),
  }
}

/**
 * Human-readable byte size, using binary units (GiB) because every model size in
 * this domain is reported that way.
 *
 * @param {number|undefined} bytes
 * @param {number} digits
 */
export function formatBytes(bytes, digits = 2) {
  if (bytes === undefined || bytes === null || !Number.isFinite(bytes)) return 'unknown'
  const gib = bytes / 1024 ** 3
  if (gib >= 1) return `${gib.toFixed(digits)} GB`
  const mib = bytes / 1024 ** 2
  if (mib >= 1) return `${mib.toFixed(0)} MB`
  return `${Math.round(bytes / 1024)} KB`
}
