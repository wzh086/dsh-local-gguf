/**
 * Managed `llama-server` backends for direct GGUF execution.
 *
 * This is the path that does not need LM Studio to be running: the plugin starts
 * a llama.cpp server itself on one model file, using the CUDA/Vulkan/CPU runtime
 * LM Studio already downloaded (or any `llama-server` on the machine), and talks
 * to it over the same OpenAI-compatible API.
 *
 * Two things make this more than a `spawn` call:
 *  - A CUDA runtime loads its `cudart`/`cublas` DLLs from LM Studio's shared
 *    `vendor` directory, so that directory has to be prepended to `PATH` or the
 *    child dies with `STATUS_DLL_NOT_FOUND` (0xC0000135).
 *  - llama.cpp prints its real memory accounting to stderr on startup, which is
 *    a far better number than any estimate, so the log is parsed as it arrives.
 */

import fs from 'node:fs'
import net from 'node:net'
import path from 'node:path'
import { spawn } from 'node:child_process'
import { toLlamaServerArgs } from './estimate.js'

/**
 * Startup log patterns for llama.cpp's own memory accounting.
 *
 * These lines are logged at *trace* verbosity, which is why a managed backend is
 * started with `-lv 4`: at the default level 3 llama.cpp prints nothing about its
 * allocations. Measured against llama.cpp 0.6.0-dev, they read:
 *
 *   load_tensors:        CUDA0 model buffer size =  8341.87 MiB
 *   llama_kv_cache:        CPU KV buffer size =    64.00 MiB
 *   sched_reserve:      CUDA0 compute buffer size =   158.50 MiB
 *
 * The device name is the token immediately before the phrase, so one pattern per
 * category captures both the device and the size.
 */
const LOG_PATTERNS = {
  weights: /(\S+)\s+model buffer size\s*=\s*([\d.]+)\s*MiB/i,
  kv: /(\S+)\s+KV buffer size\s*=\s*([\d.]+)\s*MiB/i,
  compute: /(\S+)\s+compute buffer size\s*=\s*([\d.]+)\s*MiB/i,
  // Older builds report one total instead of per-device buffers.
  weightsTotal: /model size\s*=\s*([\d.]+)\s*MiB/i,
  kvTotal: /KV self size\s*=\s*([\d.]+)\s*MiB/i,
  offloaded: /offloaded\s+(\d+)\/(\d+)\s+layers to GPU/i,
  contextSize: /n_ctx\s*=\s*(\d+)/i,
}

/** Devices whose buffers live in host memory rather than device memory. */
function isHostDevice(device) {
  const name = String(device).toUpperCase()
  return name === 'CPU' || name.startsWith('CPU_') || name.startsWith('CUDA_HOST') || name === 'HOST' || name === 'DISK'
}

/** Fold one category's per-device sizes into device and host totals. */
function splitByDevice(byDevice) {
  let gpu = 0
  let host = 0
  for (const [device, bytes] of Object.entries(byDevice)) {
    if (isHostDevice(device)) host += bytes
    else gpu += bytes
  }
  return { gpu, host }
}

/**
 * Derive the GPU/host split a caller can compare against the estimate.
 *
 * llama.cpp logs its buffers more than once — a sizing pass that may report zeros,
 * then the real allocation — so each device keeps the largest value seen, which is
 * stable regardless of how the passes interleave or how much of the log arrives.
 */
function summarizeMeasured(measured) {
  const weights = splitByDevice(measured.weightsByDevice ?? {})
  const kv = splitByDevice(measured.kvByDevice ?? {})
  const compute = splitByDevice(measured.computeByDevice ?? {})
  const gpuBytes = weights.gpu + kv.gpu + compute.gpu
  const cpuBytes = weights.host + kv.host + compute.host
  return {
    ...measured,
    weightsGpuBytes: weights.gpu,
    weightsCpuBytes: weights.host,
    kvGpuBytes: kv.gpu,
    kvCpuBytes: kv.host,
    computeGpuBytes: compute.gpu,
    computeCpuBytes: compute.host,
    gpuBytes,
    cpuBytes,
    totalBytes: gpuBytes + cpuBytes,
  }
}

/**
 * Fold a chunk of llama-server output into an accumulator.
 *
 * Exported so the parsing can be tested against real captured output rather than
 * only exercised through a live child process. Each device keeps the largest value
 * seen for its category, because llama.cpp logs a sizing pass (often reporting
 * zeros) before the real allocation.
 *
 * @param {string} text one or more log lines.
 * @param {object} [into] accumulator to mutate; a fresh one is created if omitted.
 * @returns {object} the same accumulator.
 */
export function parseLlamaServerLog(text, into) {
  const measured = into ?? { weightsByDevice: {}, kvByDevice: {}, computeByDevice: {} }
  measured.weightsByDevice ??= {}
  measured.kvByDevice ??= {}
  measured.computeByDevice ??= {}

  for (const line of String(text).split(/\r?\n/)) {
    if (!line.trim()) continue
    for (const [category, field] of [
      ['weights', 'weightsByDevice'],
      ['kv', 'kvByDevice'],
      ['compute', 'computeByDevice'],
    ]) {
      const match = LOG_PATTERNS[category].exec(line)
      if (!match) continue
      const bytes = Math.round(Number(match[2]) * 1024 * 1024)
      if (!Number.isFinite(bytes)) continue
      const table = measured[field]
      table[match[1]] = Math.max(table[match[1]] ?? 0, bytes)
    }
    for (const [field, pattern] of [
      ['weightsTotal', LOG_PATTERNS.weightsTotal],
      ['kvTotal', LOG_PATTERNS.kvTotal],
    ]) {
      const match = pattern.exec(line)
      if (!match) continue
      const bytes = Math.round(Number(match[1]) * 1024 * 1024)
      if (Number.isFinite(bytes)) measured[field] = Math.max(measured[field] ?? 0, bytes)
    }
    const offloaded = LOG_PATTERNS.offloaded.exec(line)
    if (offloaded) measured.offloadedLayers = `${offloaded[1]}/${offloaded[2]}`
    const context = LOG_PATTERNS.contextSize.exec(line)
    if (context) measured.contextSize = Number(context[1])
  }
  return measured
}

/** Find a free TCP port at or above `start`. */
export async function findFreePort(start = 8080, attempts = 40) {
  for (let port = start; port < start + attempts; port++) {
    if (await isPortFree(port)) return port
  }
  throw new Error(`no free port found in ${start}-${start + attempts - 1}`)
}

function isPortFree(port) {
  return new Promise((resolve) => {
    const server = net.createServer()
    server.once('error', () => resolve(false))
    server.once('listening', () => server.close(() => resolve(true)))
    server.listen(port, '127.0.0.1')
  })
}

/** Is anything listening on this port? */
export function isPortOpen(port, host = '127.0.0.1') {
  return new Promise((resolve) => {
    const socket = net.connect({ port, host })
    const done = (result) => {
      socket.destroy()
      resolve(result)
    }
    socket.setTimeout(500)
    socket.once('connect', () => done(true))
    socket.once('timeout', () => done(false))
    socket.once('error', () => done(false))
  })
}

/**
 * Owns every `llama-server` process this plugin starts.
 *
 * A live backend is keyed by absolute model path, so asking for the same model
 * twice reuses the running server instead of starting a second copy that would
 * compete for VRAM.
 */
export class BackendManager {
  /**
   * @param {object} options
   * @param {(level: string, message: string) => void} options.log
   */
  constructor({ log = () => {} } = {}) {
    this.log = log
    /** @type {Map<string, object>} keyed by absolute model path */
    this.processes = new Map()
  }

  /**
   * Describe every process this manager owns.
   *
   * `measured` goes through the same summarizer `describe()` uses: returning the
   * raw per-device accumulator here would make the very same backend report real
   * numbers through `start` and "unavailable" through `status`, because only the
   * summarized shape carries `totalBytes`.
   */
  list() {
    return [...this.processes.values()].map((entry) => ({
      modelPath: entry.modelPath,
      pid: entry.pid,
      port: entry.port,
      baseUrl: entry.baseUrl,
      backendId: entry.backendId,
      startedAt: entry.startedAt,
      running: entry.child.exitCode === null && !entry.child.killed,
      measured: summarizeMeasured(entry.measured),
      measuredAvailable: entry.measuredAvailable,
      args: entry.args,
    }))
  }

  /**
   * Start (or reuse) a backend for one model.
   *
   * @param {object} options
   * @param {string} options.modelPath absolute `.gguf` path.
   * @param {object} options.backend a backend descriptor from `findLlamaCppBackends`.
   * @param {object} options.config load configuration (see `lib/estimate.js`).
   * @param {string} [options.alias] model id the OpenAI API should advertise.
   * @param {number} [options.port] fixed port; a free one is chosen otherwise.
   * @param {string[]} [options.extraArgs] extra llama-server flags.
   * @param {number} [options.startupTimeoutMs]
   */
  async ensure({ modelPath, backend, config, alias, port, extraArgs, startupTimeoutMs = 180000, signal }) {
    const key = path.resolve(modelPath)
    const existing = this.processes.get(key)
    if (existing && existing.child.exitCode === null && !existing.child.killed) {
      const healthy = await this.#healthy(existing.baseUrl, 2000)
      if (healthy) return { ...this.describe(existing), reused: true }
      // The process is alive but not serving; a half-dead server would otherwise
      // block every later request, so it is replaced.
      this.stop(key)
    }

    if (!fs.existsSync(modelPath)) throw new Error(`model file not found: ${modelPath}`)
    if (!backend?.serverExe || !fs.existsSync(backend.serverExe)) {
      throw new Error('no llama-server runtime is available; install LM Studio or configure a llama-server path')
    }

    const chosenPort = port ?? (await findFreePort(8080))
    const args = toLlamaServerArgs(null, { ...config, extraArgs }, {
      modelPath,
      alias: alias ?? path.basename(modelPath, '.gguf'),
      port: chosenPort,
    })

    // A CUDA/Vulkan runtime resolves its sibling DLLs from LM Studio's shared
    // vendor directory, which is not on PATH by default.
    const env = { ...process.env }
    if (backend.vendorDir) {
      env.PATH = `${backend.vendorDir}${path.delimiter}${backend.dir}${path.delimiter}${env.PATH ?? ''}`
    } else {
      env.PATH = `${backend.dir}${path.delimiter}${env.PATH ?? ''}`
    }

    this.log('info', `starting ${backend.id}: ${backend.serverExe} ${args.join(' ')}`)
    const spawnOptions = {
      cwd: backend.dir,
      env,
      windowsHide: true,
    }
    // Piped stdio is what makes the real memory accounting readable, but a
    // process sandbox can refuse to create the pipes (spawn EPERM). Capturing the
    // log is a nicety, so a refused pipe degrades to an inherited-nothing child
    // rather than failing the load.
    let child
    let pipesAvailable = true
    try {
      child = spawn(backend.serverExe, args, { ...spawnOptions, stdio: ['ignore', 'pipe', 'pipe'] })
    } catch (error) {
      if (error?.code !== 'EPERM') throw error
      pipesAvailable = false
      this.log('warn', 'the sandbox refused piped stdio; starting llama-server without log capture')
      child = spawn(backend.serverExe, args, { ...spawnOptions, stdio: 'ignore' })
    }

    const entry = {
      modelPath: key,
      child,
      pid: child.pid,
      port: chosenPort,
      baseUrl: `http://127.0.0.1:${chosenPort}/v1`,
      backendId: backend.id,
      startedAt: Date.now(),
      args,
      measured: {},
      measuredAvailable: pipesAvailable,
      logTail: [],
      exit: undefined,
    }
    this.processes.set(key, entry)

    // llama.cpp reports its real allocation on stderr at trace level; collecting
    // it turns the estimate into a measured figure for the model actually loaded.
    const consume = (chunk) => {
      const text = String(chunk)
      for (const line of text.split(/\r?\n/)) {
        if (!line.trim()) continue
        entry.logTail.push(line)
        if (entry.logTail.length > 200) entry.logTail.shift()
      }
      parseLlamaServerLog(text, entry.measured)
    }
    child.stdout?.on('data', consume)
    child.stderr?.on('data', consume)

    // A child whose pipes were refused after creation reports the failure here.
    child.on('error', (error) => {
      if (error?.code === 'EPERM') {
        entry.measuredAvailable = false
        entry.logTail.push(`stdio refused: ${error.message}`)
        return
      }
      entry.logTail.push(`spawn failed: ${error.message}`)
    })

    const exited = new Promise((resolve) => {
      child.once('exit', (code, exitSignal) => {
        entry.exit = { code, signal: exitSignal }
        entry.logTail.push(`process exited: code=${code} signal=${exitSignal}`)
        resolve(entry.exit)
      })
    })
    entry.exited = exited

    try {
      await this.#awaitReady(entry, { startupTimeoutMs, signal, exitPromise: exited })
    } catch (error) {
      this.stop(key)
      const tail = entry.logTail.slice(-12).join('\n')
      throw new Error(`${error instanceof Error ? error.message : String(error)}${tail ? `\n${tail}` : ''}`)
    }
    return { ...this.describe(entry), reused: false }
  }

  /** Poll `/health` until the server answers or the deadline passes. */
  async #awaitReady(entry, { startupTimeoutMs, signal, exitPromise }) {
    const deadline = Date.now() + startupTimeoutMs
    while (Date.now() < deadline) {
      if (signal?.aborted) throw signal.reason ?? new Error('aborted')
      const raced = await Promise.race([
        this.#healthy(entry.baseUrl, 1500).then((ok) => (ok ? 'ready' : 'pending')),
        exitPromise.then(() => 'exited'),
        new Promise((resolve) => setTimeout(() => resolve('pending'), 1500)),
      ])
      if (raced === 'ready') return
      if (raced === 'exited') {
        throw new Error(`llama-server exited during startup (code=${entry.exit?.code})`)
      }
    }
    throw new Error(`llama-server did not become ready within ${startupTimeoutMs} ms`)
  }

  /** Ask llama-server whether it is serving yet. */
  async #healthy(baseUrl, timeoutMs) {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), timeoutMs)
    try {
      // `/health` returns 503 with {"status":"loading model"} while it loads.
      const response = await fetch(`${baseUrl.replace(/\/v1$/, '')}/health`, { signal: controller.signal })
      if (!response.ok) return false
      const body = await response.json().catch(() => undefined)
      return body === undefined || body.status === undefined || body.status === 'ok'
    } catch {
      return false
    } finally {
      clearTimeout(timer)
    }
  }

  /** A detached, JSON-serializable view of one entry. */
  describe(entry) {
    return {
      modelPath: entry.modelPath,
      pid: entry.pid,
      port: entry.port,
      baseUrl: entry.baseUrl,
      backendId: entry.backendId,
      startedAt: entry.startedAt,
      args: entry.args,
      measured: summarizeMeasured(entry.measured),
      measuredAvailable: entry.measuredAvailable,
      running: entry.child.exitCode === null && !entry.child.killed,
      logTail: entry.logTail.slice(-25),
    }
  }

  /** Stop the backend serving one model. */
  stop(modelPath) {
    const key = path.resolve(modelPath)
    const entry = this.processes.get(key)
    if (!entry) return false
    this.#terminate(entry)
    this.processes.delete(key)
    return true
  }

  /** Stop every backend this plugin started. Registered as a plugin effect. */
  stopAll() {
    for (const [key, entry] of this.processes) {
      this.#terminate(entry)
      this.processes.delete(key)
    }
  }

  /**
   * Terminate one child process tree.
   *
   * `taskkill /T` is used on Windows because llama-server does not exit on its
   * own while a request is streaming, and killing only the parent would leave the
   * GPU allocation held.
   */
  #terminate(entry) {
    const { child } = entry
    if (child.exitCode !== null || child.killed) return
    try {
      if (process.platform === 'win32' && child.pid) {
        spawn('taskkill', ['/pid', String(child.pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' }).on('error', () => {})
      } else {
        child.kill('SIGTERM')
        setTimeout(() => {
          if (child.exitCode === null) child.kill('SIGKILL')
        }, 5000).unref?.()
      }
    } catch {
      // A process that is already gone needs no further handling.
    }
  }
}
