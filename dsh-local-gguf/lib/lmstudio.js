/**
 * LM Studio discovery and lifecycle.
 *
 * LM Studio is the preferred local backend because it ships a CUDA/Vulkan/CPU
 * build of `llama.cpp` and already knows where the user's GGUF files live. Three
 * things are discovered here:
 *
 *  1. **Installation** — the app, its `lms` CLI, and the downloaded llama.cpp
 *     runtimes under `%USERPROFILE%\.lmstudio\extensions\backends`.
 *  2. **Model directories** — read from LM Studio's own model index so a model
 *     the user already downloaded is found without re-configuring anything.
 *  3. **Server state** — whether the OpenAI-compatible endpoint answers, and
 *     which models it advertises.
 *
 * Nothing here mutates LM Studio state except the explicitly requested
 * `startServer` / `loadModel` operations.
 */

import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { spawn } from 'node:child_process'

/** Default port of LM Studio's OpenAI-compatible server. */
export const DEFAULT_LMSTUDIO_PORT = 1234

/** LM Studio's per-user data directory. */
export function lmStudioHome(env = process.env) {
  const override = env.LMSTUDIO_HOME
  if (override) return override
  return path.join(os.homedir(), '.lmstudio')
}

/** Read a JSON file, returning undefined instead of throwing. */
async function readJson(file) {
  try {
    return JSON.parse(await fs.promises.readFile(file, 'utf8'))
  } catch {
    return undefined
  }
}

/**
 * Rank a llama.cpp backend directory by preference for this machine.
 *
 * A CUDA build is worth far more than a CPU build when an NVIDIA card is
 * present, and `llama.cpp` version ordering is numeric rather than lexical
 * (`2.9.0` is older than `2.10.0`), so the parsed version numbers drive the sort
 * instead of the directory name.
 */
function backendRank(name, hasCuda, hasVulkan) {
  const version = /-(\d+\.\d+\.\d+)$/.exec(name)
  const numeric = version ? version[1].split('.').map(Number) : [0, 0, 0]
  const comparable = numeric[0] * 10000 + numeric[1] * 100 + numeric[2]
  if (name.includes('nvidia-cuda12')) return hasCuda ? 3000000 + comparable : -1
  if (name.includes('nvidia-cuda-')) return hasCuda ? 2000000 + comparable : -1
  if (name.includes('vulkan')) return hasVulkan ? 1000000 + comparable : -1
  if (name.includes('avx2') || name.includes('avx')) return 500000 + comparable
  return 100000 + comparable
}

/**
 * Find the llama.cpp runtimes LM Studio has downloaded.
 *
 * Each runtime directory carries a small `llama-server.exe` that loads its heavy
 * sibling DLLs, plus a CUDA/Vulkan runtime that lives in a shared `vendor`
 * directory and therefore has to be placed on `PATH` before launch.
 */
export async function findLlamaCppBackends(hardware, env = process.env) {
  const backendsRoot = path.join(lmStudioHome(env), 'extensions', 'backends')
  let entries
  try {
    entries = await fs.promises.readdir(backendsRoot, { withFileTypes: true })
  } catch {
    return []
  }
  const hasCuda = hardware?.gpus?.some((g) => g.vendor === 'nvidia') ?? false
  const hasVulkan = hardware?.gpus?.some((g) => g.vendor === 'amd' || g.vendor === 'intel') ?? false
  const vendorRoot = path.join(backendsRoot, 'vendor')

  const found = []
  for (const entry of entries) {
    if (!entry.isDirectory() || entry.name === 'vendor') continue
    const dir = path.join(backendsRoot, entry.name)
    const serverExe = path.join(dir, 'llama-server.exe')
    if (!fs.existsSync(serverExe)) continue
    let vendorDir
    if (entry.name.includes('nvidia-cuda12')) vendorDir = path.join(vendorRoot, 'win-llama-cuda12-vendor-v2')
    else if (entry.name.includes('nvidia-cuda')) vendorDir = path.join(vendorRoot, 'win-llama-cuda-vendor-v2')
    else if (entry.name.includes('vulkan')) vendorDir = path.join(vendorRoot, 'win-llama-vulkan-vendor-v2')
    if (vendorDir && !fs.existsSync(vendorDir)) vendorDir = undefined
    const manifest = await readJson(path.join(dir, 'backend-manifest.json'))
    found.push({
      id: entry.name,
      dir,
      serverExe,
      vendorDir,
      version: manifest?.version,
      engine: manifest?.engine ?? 'llama.cpp',
      acceleration: entry.name.includes('cuda') ? 'cuda' : entry.name.includes('vulkan') ? 'vulkan' : 'cpu',
      rank: backendRank(entry.name, hasCuda, hasVulkan),
    })
  }
  return found.sort((a, b) => b.rank - a.rank)
}

/**
 * Read the model directories LM Studio has indexed.
 *
 * The index stores an absolute path per model; the directory that holds the
 * publisher folders is two levels up from any model file.
 */
export async function lmStudioModelDirs(env = process.env) {
  const dirs = new Set()
  const home = lmStudioHome(env)
  const index = await readJson(path.join(home, '.internal', 'model-index-cache.json'))
  const models = Array.isArray(index?.models) ? index.models : []
  for (const model of models) {
    const absolute = model?.containingDirAbsolutePath
    if (typeof absolute !== 'string' || !absolute) continue
    const publisherDir = absolute.replace(/\//g, path.sep)
    const root = path.dirname(path.dirname(publisherDir))
    if (root && root !== '.' && !dirs.has(root)) dirs.add(root)
  }
  const settings = await readJson(path.join(home, 'settings.json'))
  if (typeof settings?.downloadsFolder === 'string' && settings.downloadsFolder) {
    dirs.add(settings.downloadsFolder)
  }
  return [...dirs]
}

/** Read LM Studio's GGUF metadata cache, keyed by absolute model path. */
export async function lmStudioMetadataCache(env = process.env) {
  const cache = await readJson(path.join(lmStudioHome(env), '.internal', 'gguf-metadata-cache.json'))
  const rows = cache?.json?.map
  if (!Array.isArray(rows)) return new Map()
  const out = new Map()
  for (const row of rows) {
    if (!Array.isArray(row) || row.length < 2) continue
    out.set(String(row[0]).replace(/\//g, path.sep), row[1])
  }
  return out
}

/**
 * Locate an LM Studio installation.
 *
 * Candidate roots are gathered from the recorded install location, the standard
 * per-user and machine-wide install directories, and an explicit
 * `installDir` from configuration. LM Studio records where it was installed in
 * `app-install-location.json`, which is what makes a non-default location work
 * without hardcoding anyone's path here.
 */
export async function discoverLmStudio(hardware, env = process.env, configured = {}) {
  const home = lmStudioHome(env)
  const candidates = []
  if (configured.installDir) candidates.push(configured.installDir)

  const recorded = await readJson(path.join(home, '.internal', 'app-install-location.json'))
  if (typeof recorded?.path === 'string') candidates.push(recorded.path)
  if (typeof recorded?.installLocation === 'string') candidates.push(recorded.installLocation)
  // `recorded.path` points at the executable; its directory is the install root.
  if (typeof recorded?.path === 'string' && recorded.path.toLowerCase().endsWith('.exe')) {
    candidates.push(path.dirname(recorded.path))
  }

  const localAppData = env.LOCALAPPDATA ?? path.join(os.homedir(), 'AppData', 'Local')
  const programFiles = env.ProgramFiles ?? 'C:\\Program Files'
  candidates.push(
    path.join(localAppData, 'Programs', 'LM Studio'),
    path.join(localAppData, 'LM-Studio'),
    path.join(programFiles, 'LM Studio'),
  )

  let installDir
  for (const candidate of candidates) {
    if (!candidate) continue
    if (fs.existsSync(path.join(candidate, 'LM Studio.exe'))) {
      installDir = candidate
      break
    }
  }

  const lmsCandidates = [path.join(home, 'bin', 'lms.exe'), path.join(home, 'bin', 'lms')]
  if (installDir) lmsCandidates.push(path.join(installDir, 'resources', 'app', '.webpack', 'lms.exe'))
  let lmsPath
  for (const candidate of lmsCandidates) {
    if (fs.existsSync(candidate)) {
      lmsPath = candidate
      break
    }
  }

  const backends = await findLlamaCppBackends(hardware, env)
  const modelDirs = await lmStudioModelDirs(env)
  return {
    installed: Boolean(installDir),
    installDir,
    appExe: installDir ? path.join(installDir, 'LM Studio.exe') : undefined,
    lmsPath,
    home,
    modelDirs,
    backends,
    bestBackend: backends[0],
  }
}

/**
 * Normalize a base URL to the OpenAI-compatible root, so a user may type
 * `http://127.0.0.1:1234` or `http://127.0.0.1:1234/v1` and mean the same thing.
 */
export function normalizeBaseUrl(baseUrl, defaultPort = DEFAULT_LMSTUDIO_PORT) {
  let url = String(baseUrl ?? '').trim()
  if (!url) url = `http://127.0.0.1:${defaultPort}/v1`
  if (!/^https?:\/\//i.test(url)) url = `http://${url}`
  url = url.replace(/\/+$/, '')
  if (!/\/v\d+$/.test(url) && !/\/api\//.test(url)) url += '/v1'
  return url
}

/** Join a base URL with an endpoint path, tolerating a base ending in `/v1`. */
function endpoint(baseUrl, suffix) {
  const root = baseUrl.replace(/\/v1$/, '')
  return `${root}${suffix}`
}

/**
 * Fetch JSON with a bounded timeout.
 *
 * A local server that is starting up refuses connections for a while, so callers
 * treat any failure as "not ready" rather than as an error to surface.
 */
async function fetchJson(url, { apiKey, signal, timeoutMs = 4000, method = 'GET', body } = {}) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  const signals = signal ? [signal, controller.signal] : [controller.signal]
  try {
    const response = await fetch(url, {
      method,
      signal: AbortSignal.any(signals),
      headers: {
        accept: 'application/json',
        ...(body ? { 'content-type': 'application/json' } : {}),
        ...(apiKey ? { authorization: `Bearer ${apiKey}` } : {}),
      },
      ...(body ? { body: JSON.stringify(body) } : {}),
    })
    if (!response.ok) {
      const text = await response.text().catch(() => '')
      throw new Error(`HTTP ${response.status} ${response.statusText}${text ? `: ${text.slice(0, 200)}` : ''}`)
    }
    return await response.json()
  } finally {
    clearTimeout(timer)
  }
}

/**
 * Ask a local server which models it serves, preferring LM Studio's richer
 * native endpoint and falling back to the OpenAI one every compatible server
 * implements.
 *
 * @returns {Promise<{reachable: boolean, models: object[], native: boolean, error?: string}>}
 */
export async function listServerModels(baseUrl, { apiKey, signal, timeoutMs } = {}) {
  const base = normalizeBaseUrl(baseUrl)
  try {
    const native = await fetchJson(endpoint(base, '/api/v0/models'), { apiKey, signal, timeoutMs })
    const rows = Array.isArray(native?.data) ? native.data : []
    if (rows.length) {
      return {
        reachable: true,
        native: true,
        models: rows.map((row) => ({
          id: String(row.id),
          name: String(row.id),
          description: row.arch ? `${row.arch}${row.quantization ? ` · ${row.quantization}` : ''}` : undefined,
          contextWindow: firstNumber(row.max_context_length, row.loaded_context_length),
          loadedContextWindow: firstNumber(row.loaded_context_length),
          state: typeof row.state === 'string' ? row.state : undefined,
          publisher: row.publisher,
          arch: row.arch,
          quantization: row.quantization,
          compatibilityType: row.compatibility_type,
          type: row.type,
          source: 'lmstudio-native',
        })),
      }
    }
  } catch (error) {
    // Fall through to the OpenAI-compatible probe; the native route is optional.
    if (signal?.aborted) throw error
  }
  try {
    const openai = await fetchJson(`${base}/models`, { apiKey, signal, timeoutMs })
    const rows = Array.isArray(openai?.data) ? openai.data : []
    return {
      reachable: true,
      native: false,
      models: rows.map((row) => ({
        id: String(row.id),
        name: String(row.id),
        contextWindow: firstNumber(row.max_context_length, row.context_length),
        source: 'openai-compatible',
      })),
    }
  } catch (error) {
    return { reachable: false, native: false, models: [], error: error instanceof Error ? error.message : String(error) }
  }
}

function firstNumber(...values) {
  for (const value of values) {
    if (typeof value === 'number' && Number.isFinite(value) && value > 0) return value
  }
  return undefined
}

/**
 * Ask LM Studio to load a model through the HTTP API (LM Studio 0.4.0+).
 *
 * This is the only way to load with real load parameters: the `lms` CLI exposes
 * just `--context-length` and a `--gpu` ratio, while `POST /api/v1/models/load`
 * accepts `context_length`, `eval_batch_size`, `flash_attention`, `num_experts`
 * and `offload_kv_cache_to_gpu`.
 *
 * @returns {Promise<{loaded: true, via: 'http', response: object}>} on success.
 * @throws when the endpoint is absent, so the caller can fall back to `lms`.
 */
export async function lmStudioHttpLoad(baseUrl, { modelKey, identifier, config = {}, timeoutMs = 300000, apiKey, signal } = {}) {
  const base = normalizeBaseUrl(baseUrl)
  const body = {
    model_key: modelKey,
    ...(identifier ? { identifier } : {}),
    ...(Number.isFinite(config.contextLength) ? { context_length: config.contextLength } : {}),
    ...(Number.isFinite(config.batchSize) ? { eval_batch_size: config.batchSize } : {}),
    ...(config.flashAttention !== undefined ? { flash_attention: config.flashAttention } : {}),
    ...(Number.isFinite(config.expertCount) ? { num_experts: config.expertCount } : {}),
    ...(config.offloadKvToGpu !== undefined ? { offload_kv_cache_to_gpu: config.offloadKvToGpu } : {}),
    echo_load_config: true,
  }
  const response = await fetchJson(endpoint(base, '/api/v1/models/load'), {
    apiKey,
    signal,
    timeoutMs,
    method: 'POST',
    body,
  })
  return { loaded: true, via: 'http', response }
}

/**
 * Ask LM Studio to unload a resident model through the HTTP API (0.4.0+).
 *
 * @param {string} instanceId the `instance_id` from `GET /api/v1/models`.
 */
export async function lmStudioHttpUnload(baseUrl, instanceId, { apiKey, signal, timeoutMs = 30000 } = {}) {
  const base = normalizeBaseUrl(baseUrl)
  return fetchJson(endpoint(base, '/api/v1/models/unload'), {
    apiKey,
    signal,
    timeoutMs,
    method: 'POST',
    body: { instance_id: instanceId },
  })
}

/**
 * List the models LM Studio currently has resident, using the v1 API when the
 * build provides it.
 *
 * `lms ps` is the fallback because it needs the daemon but works on every
 * version that ships the CLI.
 *
 * @returns {Promise<{loaded: object[], via: string}>}
 */
export async function lmStudioLoadedInstances(baseUrl, { apiKey, signal, lmsPath, timeoutMs = 8000 } = {}) {
  try {
    const v1 = await fetchJson(endpoint(normalizeBaseUrl(baseUrl), '/api/v1/models'), { apiKey, signal, timeoutMs })
    const rows = Array.isArray(v1?.data) ? v1.data : Array.isArray(v1?.models) ? v1.models : []
    const loaded = []
    for (const row of rows) {
      for (const instance of row?.loaded_instances ?? []) {
        loaded.push({
          modelKey: row.key ?? row.id,
          instanceId: instance?.id,
          contextLength: instance?.config?.context_length,
          config: instance?.config,
        })
      }
    }
    return { loaded, via: 'http-v1' }
  } catch (error) {
    if (signal?.aborted) throw error
  }
  if (lmsPath && fs.existsSync(lmsPath)) {
    const result = await lmsPs(lmsPath)
    return { loaded: parseLmsPs(result.stdout), via: 'lms-ps' }
  }
  return { loaded: [], via: 'none' }
}

/** Parse `lms ps --json`, tolerating both a bare array and a wrapped object. */
function parseLmsPs(stdout) {
  try {
    const parsed = JSON.parse(stdout)
    const rows = Array.isArray(parsed) ? parsed : Array.isArray(parsed?.models) ? parsed.models : []
    return rows.map((row) => ({
      modelKey: row.modelKey ?? row.identifier ?? row.path ?? row.model,
      instanceId: row.identifier,
      contextLength: row.contextLength ?? row.maxContextLength,
    }))
  } catch {
    // `lms ps` without `--json` prints a table; the caller only needs a count.
    return stdout
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith('IDENTIFIER') && !line.startsWith('-'))
      .map((line) => ({ modelKey: line.split(/\s{2,}/)[0] }))
  }
}

/**
 * Load a model with explicit parameters.
 *
 * Prefers the HTTP API because only it can express a full load configuration;
 * `lms load` is the fallback for LM Studio 0.3.x, where only the context length
 * and a GPU-memory ratio are settable and the rest is ignored with a warning.
 *
 * @returns {Promise<{loaded: boolean, via: string, warnings: string[], detail?: string}>}
 */
export async function loadModel(baseUrl, { lmsPath, modelKey, identifier, config = {}, apiKey, signal } = {}) {
  const warnings = []
  try {
    const result = await lmStudioHttpLoad(baseUrl, { modelKey, identifier, config, apiKey, signal })
    return { loaded: true, via: 'http', warnings, detail: result.response }
  } catch (error) {
    if (signal?.aborted) throw error
    warnings.push(
      `the HTTP load endpoint is unavailable (${error instanceof Error ? error.message : String(error)}); falling back to the \`lms\` CLI`,
    )
  }
  if (!lmsPath || !fs.existsSync(lmsPath)) {
    throw new Error(`cannot load "${modelKey}": neither the HTTP load endpoint nor the \`lms\` CLI is available`)
  }
  // Only these two are settable through the CLI; anything else is reported as
  // ignored rather than silently dropped.
  const cliConfig = {}
  if (Number.isFinite(config.contextLength)) cliConfig.contextLength = config.contextLength
  if (Number.isFinite(config.gpuLayers) && Number.isFinite(config.totalLayers) && config.totalLayers > 0) {
    cliConfig.gpuRatio = Math.max(0, Math.min(1, config.gpuLayers / config.totalLayers))
  }
  for (const key of ['batchSize', 'flashAttention', 'kvTypeK', 'threads']) {
    if (config[key] !== undefined) warnings.push(`\`lms load\` cannot set ${key}; it was not applied`)
  }
  const result = await lmsLoad(lmsPath, { modelKey, identifier, config: cliConfig })
  return { loaded: result.code === 0, via: 'lms-cli', warnings, detail: result.stdout || result.stderr }
}

/**
 * Ask LM Studio to load a model, optionally with explicit load parameters.
 *
 * `lms` talks to the running LM Studio daemon over a local socket, so this is the
 * only supported way to control loading on 0.3.x. The call is bounded because
 * `lms` blocks while it waits for the daemon.
 */
export async function lmsLoad(lmsPath, { modelKey, identifier, config, timeoutMs = 120000, env = process.env } = {}) {
  if (!lmsPath || !fs.existsSync(lmsPath)) throw new Error('the LM Studio `lms` CLI was not found')
  const args = ['load', modelKey]
  if (identifier) args.push('--identifier', identifier)
  if (config) {
    if (Number.isFinite(config.contextLength)) args.push('--context-length', String(config.contextLength))
    // `--gpu` is a memory-offload ratio or off/max, never a layer count.
    if (typeof config.gpuRatio === 'number') args.push('--gpu', String(config.gpuRatio.toFixed(3)))
  }
  args.push('--yes')
  return runCommand(lmsPath, args, { timeoutMs, env })
}

/** Ask LM Studio which models are currently resident. */
export async function lmsPs(lmsPath, { timeoutMs = 30000, env = process.env } = {}) {
  if (!lmsPath || !fs.existsSync(lmsPath)) throw new Error('the LM Studio `lms` CLI was not found')
  return runCommand(lmsPath, ['ps', '--json'], { timeoutMs, env })
}

/** Start LM Studio's OpenAI-compatible server, optionally on a chosen port. */
export async function lmsServerStart(lmsPath, { port, timeoutMs = 60000, env = process.env } = {}) {
  if (!lmsPath || !fs.existsSync(lmsPath)) throw new Error('the LM Studio `lms` CLI was not found')
  const args = ['server', 'start']
  if (Number.isFinite(port) && port > 0) args.push('--port', String(port))
  return runCommand(lmsPath, args, { timeoutMs, env })
}

/** Report LM Studio's server status. */
export async function lmsServerStatus(lmsPath, { timeoutMs = 30000, env = process.env } = {}) {
  if (!lmsPath || !fs.existsSync(lmsPath)) throw new Error('the LM Studio `lms` CLI was not found')
  return runCommand(lmsPath, ['server', 'status'], { timeoutMs, env })
}

/**
 * Launch the LM Studio application detached.
 *
 * The app must be running before `lms` can talk to it, so this is the first step
 * of the cold-start path.
 */
export function launchLmStudioApp(appExe) {
  if (!appExe || !fs.existsSync(appExe)) throw new Error('the LM Studio application was not found')
  const child = spawn(appExe, [], { detached: true, stdio: 'ignore', windowsHide: false })
  child.unref()
  return { pid: child.pid }
}

/**
 * Run a short-lived process and capture its output.
 *
 * Exported because the panel's download action needs the same bounded,
 * output-capturing invocation the `lms` wrappers use.
 */
export function runCommand(file, args, { timeoutMs = 60000, env = process.env } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(file, args, { windowsHide: true, env })
    let stdout = ''
    let stderr = ''
    const timer = setTimeout(() => {
      child.kill()
      reject(new Error(`${path.basename(file)} ${args[0]} timed out after ${timeoutMs} ms`))
    }, timeoutMs)
    child.stdout?.on('data', (chunk) => {
      stdout += chunk
    })
    child.stderr?.on('data', (chunk) => {
      stderr += chunk
    })
    child.on('error', (error) => {
      clearTimeout(timer)
      reject(error)
    })
    child.on('close', (code) => {
      clearTimeout(timer)
      resolve({ code, stdout: stdout.trim(), stderr: stderr.trim() })
    })
  })
}

/**
 * Wait until a local server answers, polling the models endpoint.
 *
 * A cold LM Studio start can take a while because the daemon loads the inference
 * runtime before it binds the port, so the deadline is generous and the caller
 * controls it.
 */
export async function waitForServer(baseUrl, { timeoutMs = 60000, intervalMs = 750, apiKey, signal } = {}) {
  const deadline = Date.now() + timeoutMs
  let lastError
  while (Date.now() < deadline) {
    if (signal?.aborted) throw signal.reason ?? new Error('aborted')
    const probe = await listServerModels(baseUrl, { apiKey, signal, timeoutMs: Math.min(4000, timeoutMs) })
    if (probe.reachable) return probe
    lastError = probe.error
    await new Promise((resolve) => setTimeout(resolve, intervalMs))
  }
  throw new Error(`no server answered at ${normalizeBaseUrl(baseUrl)} within ${timeoutMs} ms${lastError ? ` (${lastError})` : ''}`)
}
