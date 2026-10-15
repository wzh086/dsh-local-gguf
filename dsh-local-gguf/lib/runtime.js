/**
 * Plugin runtime: route resolution, the model catalog, and backend lifecycles.
 *
 * The adapter asks this object two questions — "which models exist?" and "how do I
 * reach this one?" — and everything that needs the filesystem, the hardware probe
 * or a child process lives here so the adapter stays a pure protocol translation.
 *
 * Two route kinds are supported:
 *  - `lmstudio`   talks to LM Studio's own server, starting it if necessary.
 *  - `llamacpp`   starts a `llama-server` on a specific GGUF file, so no LM Studio
 *                 process has to be running at all.
 * A third, `openai`, is the escape hatch for any other OpenAI-compatible local
 * server (Ollama, vLLM, text-generation-webui).
 */

import path from 'node:path'
import { LocalGgufAdapter } from './adapter.js'
import { BackendManager } from './backend.js'
import { probeHardware, formatBytes } from './hardware.js'
import { defaultLoadConfig, estimateMemory, recommendConfig } from './estimate.js'
import { clearHeaderCache, matchModel, modelByPath, scanModels } from './models.js'
import {
  DEFAULT_LMSTUDIO_PORT,
  discoverLmStudio,
  launchLmStudioApp,
  listServerModels,
  lmStudioLoadedInstances,
  lmStudioHttpUnload,
  loadModel,
  lmsServerStart,
  lmsServerStatus,
  normalizeBaseUrl,
  waitForServer,
} from './lmstudio.js'

/** Route ids owned by this plugin and served out of the box. */
export const DEFAULT_ROUTES = ['lmstudio', 'local-gguf']

/** How long a cached hardware probe and catalog stay fresh. */
const HARDWARE_TTL_MS = 60_000
const CATALOG_TTL_MS = 15_000

export class LocalRuntime {
  /**
   * @param {object} options
   * @param {() => object} options.readConfig returns the live plugin config.
   * @param {(level: string, message: string) => void} options.log
   * @param {(command: string, args: string[], options?: object) => Promise<object>} [options.runCommand]
   */
  constructor({ readConfig, log = () => {} }) {
    this.readConfig = readConfig
    this.log = log
    this.backends = new BackendManager({ log })
    const adapter = new LocalGgufAdapter(this)
    this.adapter = adapter
    /** Memoized probes, invalidated by their own TTL. */
    this.hardwareCache = undefined
    this.catalogCache = undefined
    this.lmStudioCache = undefined
    /** Resolved model metadata, so `prepareCall` can answer synchronously. */
    this.modelInfoCache = new Map()
    /** Tokens/second reported by the last completed request, per route. */
    this.lastThroughput = new Map()
  }

  /**
   * The current config as a plain, defensively normalized object.
   *
   * This plugin declares no Config schema, so the value arrives exactly as the
   * patch wrote it — possibly with missing, mistyped or extra fields. Every
   * reader below therefore treats it as untrusted input.
   */
  config() {
    const raw = this.readConfig()
    const source = raw && typeof raw === 'object' ? raw : {}
    const providers =
      source.providers && typeof source.providers === 'object' && !Array.isArray(source.providers)
        ? source.providers
        : {}
    return {
      ...source,
      providers,
      modelDirs: Array.isArray(source.modelDirs) ? source.modelDirs.filter((d) => typeof d === 'string') : [],
      includeLmStudioDirs: source.includeLmStudioDirs !== false,
      vramReserveRatio:
        typeof source.vramReserveRatio === 'number' && source.vramReserveRatio >= 0 && source.vramReserveRatio <= 0.5
          ? source.vramReserveRatio
          : 0.08,
    }
  }

  /**
   * The effective descriptor for one route.
   *
   * A route named in config wins; otherwise the two built-in routes are described
   * by their defaults, so the plugin is useful before anything is configured.
   */
  route(provider) {
    if (!provider || !DEFAULT_ROUTES.includes(provider)) return undefined
    const configuredRaw = this.config().providers[provider]
    const configured = configuredRaw && typeof configuredRaw === 'object' ? configuredRaw : {}
    if (provider === 'lmstudio') {
      return {
        ...configured,
        id: 'lmstudio',
        kind: 'lmstudio',
        displayName: configured.displayName ?? 'LM Studio (local)',
        baseURL: normalizeBaseUrl(configured.baseURL ?? `http://127.0.0.1:${DEFAULT_LMSTUDIO_PORT}/v1`),
        apiKey: configured.apiKey,
        modelDirs: Array.isArray(configured.modelDirs) ? configured.modelDirs : [],
        autoStart: configured.autoStart !== false,
      }
    }
    return {
      ...configured,
      id: 'local-gguf',
      kind: configured.kind ?? 'llamacpp',
      displayName: configured.displayName ?? 'Local GGUF (direct llama.cpp)',
      baseURL: normalizeBaseUrl(configured.baseURL ?? 'http://127.0.0.1:8080/v1'),
      apiKey: configured.apiKey,
      modelDirs: Array.isArray(configured.modelDirs) ? configured.modelDirs : [],
      backendId: configured.backendId ?? 'auto',
      autoStart: configured.autoStart !== false,
    }
  }

  /** Probe the machine, memoized briefly because it can spawn a process. */
  async hardware(force = false) {
    if (!force && this.hardwareCache && Date.now() - this.hardwareCache.at < HARDWARE_TTL_MS) {
      return this.hardwareCache.value
    }
    const value = await probeHardware()
    this.hardwareCache = { at: Date.now(), value }
    return value
  }

  /** Locate LM Studio, its CLI and its llama.cpp runtimes. */
  async lmStudio(force = false) {
    if (!force && this.lmStudioCache && Date.now() - this.lmStudioCache.at < HARDWARE_TTL_MS) {
      return this.lmStudioCache.value
    }
    const hardware = await this.hardware(force)
    const value = await discoverLmStudio(hardware, process.env, this.config())
    this.lmStudioCache = { at: Date.now(), value }
    return value
  }

  /**
   * Scan the configured model directories, plus LM Studio's own.
   *
   * Only `llm` files are returned for selection; projectors and embedding models
   * are reported separately because they cannot serve a chat request on their own.
   */
  async catalog({ refresh = false, signal } = {}) {
    if (!refresh && this.catalogCache && Date.now() - this.catalogCache.at < CATALOG_TTL_MS) {
      return this.catalogCache.value
    }
    const configured = this.config()
    const directories = [
      ...(Array.isArray(configured.modelDirs) ? configured.modelDirs : []),
      ...DEFAULT_ROUTES.flatMap((id) => this.route(id)?.modelDirs ?? []),
    ]
    const scan = await scanModels({
      directories,
      includeLmStudioDirs: configured.includeLmStudioDirs ?? true,
      refresh,
      signal,
    })
    const value = {
      ...scan,
      chatModels: scan.models.filter((m) => m.kind === 'llm'),
      projectors: scan.models.filter((m) => m.kind === 'mmproj'),
      embeddingModels: scan.models.filter((m) => m.kind === 'embedding'),
      at: Date.now(),
    }
    this.catalogCache = { at: Date.now(), value }
    return value
  }

  /** Pair each multimodal model with a projector sitting in the same directory. */
  projectorFor(model, catalog) {
    if (!model || model.mmproj) return undefined
    const dir = path.dirname(model.path).toLowerCase()
    return catalog.projectors.find((p) => path.dirname(p.path).toLowerCase() === dir)
  }

  /**
   * Models one route advertises.
   *
   * The catalog is the source of truth for ids so a selection survives the server
   * being restarted, and server metadata is merged in when it is reachable.
   */
  async listModels(provider) {
    const route = this.route(provider)
    if (!route) return []
    const catalog = await this.catalog()
    if (route.kind === 'lmstudio') {
      const probe = await listServerModels(route.baseURL, { apiKey: route.apiKey, timeoutMs: 2500 })
      if (probe.reachable && probe.models.length) {
        return probe.models.map((model) => ({
          provider,
          id: model.id,
          name: model.name ?? model.id,
          description: [
            model.state ? `state: ${model.state}` : undefined,
            model.quantization,
            model.contextWindow ? `ctx ${model.contextWindow.toLocaleString()}` : undefined,
          ]
            .filter(Boolean)
            .join(' · ') || undefined,
          inputModalities: ['text'],
        }))
      }
    }
    return catalog.chatModels.map((model) => ({
      provider,
      // The `lmstudio` route addresses models the way LM Studio does — its
      // publisher/repository/file key — while the direct route addresses the file.
      id: route.kind === 'lmstudio' ? model.relative : model.path,
      name: model.displayName,
      description: [model.quantization, model.arch, model.contextLength ? `ctx ${model.contextLength.toLocaleString()}` : undefined]
        .filter(Boolean)
        .join(' · '),
      inputModalities: ['text'],
    }))
  }

  /** Build model metadata without I/O, for the synchronous `prepareCall` path. */
  resolveModelSync(provider, model) {
    const cached = this.modelInfoCache.get(`${provider}\u0000${model}`)
    if (cached) return cached
    // A route that has never been resolved still needs a usable answer, so the
    // conservative default window is used until `resolveModel` fills the cache.
    return {
      provider,
      id: model,
      name: model,
      context: { contextWindow: 8192 },
      inputModalities: ['text'],
    }
  }

  /** Resolve one exact model, consulting the catalog for its context window. */
  async resolveModel(provider, model, _signal) {
    const route = this.route(provider)
    if (!route) throw new Error(`no local route configured for provider "${provider}"`)
    const catalog = await this.catalog()
    const matched = route.kind === 'lmstudio' ? matchModel(catalog.chatModels, model) : matchModel(catalog.chatModels, model)
    const contextWindow = matched?.contextLength ?? (await this.#serverContext(route, model)) ?? 8192
    const info = {
      provider,
      id: model,
      name: matched?.displayName ?? model,
      context: { contextWindow },
      inputModalities: ['text'],
    }
    this.modelInfoCache.set(`${provider}\u0000${model}`, info)
    return info
  }

  /** Ask a reachable server for the context window it reports for one model. */
  async #serverContext(route, model) {
    if (route.kind === 'lmstudio') {
      const probe = await listServerModels(route.baseURL, { apiKey: route.apiKey, timeoutMs: 2500 })
      return probe.models.find((m) => m.id === model)?.contextWindow
    }
    return undefined
  }

  /**
   * Resolve everything one request needs to be dispatched.
   *
   * This is where "it just works" lives: an unreachable LM Studio is started, and
   * a model that has no running backend gets one.
   */
  async connectionFor(provider, model, signal) {
    const route = this.route(provider)
    if (!route) throw new Error(`no local route configured for provider "${provider}"`)

    if (route.kind === 'lmstudio') {
      const probe = await listServerModels(route.baseURL, { apiKey: route.apiKey, timeoutMs: 2500, signal })
      if (probe.reachable) return { baseUrl: route.baseURL, apiKey: route.apiKey, servedModelId: model }
      if (!route.autoStart) {
        throw new Error(`LM Studio is not reachable at ${route.baseURL} and autoStart is disabled`)
      }
      await this.#startLmStudio(route, signal)
      return { baseUrl: route.baseURL, apiKey: route.apiKey, servedModelId: model }
    }

    if (route.kind === 'openai') {
      return { baseUrl: route.baseURL, apiKey: route.apiKey, servedModelId: model }
    }

    // Direct llama.cpp route: start (or reuse) a managed backend for this file and
    // send the request to the alias that backend was given.
    const catalog = await this.catalog()
    const matched = matchModel(catalog.chatModels, model) ?? (await this.#byPath(model))
    if (!matched) throw new Error(`no GGUF file matches model id "${model}"`)
    const running = this.backends.list().find((entry) => path.resolve(entry.modelPath) === path.resolve(matched.path))
    if (running) {
      return { baseUrl: running.baseUrl, apiKey: route.apiKey, servedModelId: path.basename(matched.path, '.gguf') }
    }
    if (!route.autoStart) throw new Error(`no backend is running for ${matched.path} and autoStart is disabled`)

    const started = await this.startBackend(matched.path, { signal })
    return { baseUrl: started.baseUrl, apiKey: route.apiKey, servedModelId: path.basename(matched.path, '.gguf'), backend: started }
  }

  /** Load one GGUF header directly, for a path that is not in the catalog. */
  async #byPath(model) {
    try {
      return await modelByPath(model)
    } catch {
      return undefined
    }
  }

  /**
   * Start a managed llama-server for one file.
   *
   * The load configuration comes from the recommendation engine unless the route
   * pins explicit values, so an unconfigured model still gets sensible parameters
   * for this machine.
   */
  async startBackend(modelPath, { config: explicitConfig, signal, port } = {}) {
    const route = this.route('local-gguf')
    const summary = await modelByPath(modelPath)
    const hardware = await this.hardware()
    const lmStudio = await this.lmStudio()
    const backend = this.#selectBackend(route, lmStudio, hardware)
    if (!backend) {
      throw new Error(
        'no llama-server runtime found. Install LM Studio, or set providers["local-gguf"].llamaServerPath to a llama-server executable.',
      )
    }
    const recommended = recommendConfig(summary, hardware)
    const config = {
      ...defaultLoadConfig(summary),
      threads: hardware.cpu.recommendedThreads,
      ...pickDefined({
        gpuLayers: route.gpuLayers,
        contextLength: route.contextLength,
      }),
      ...recommended.config,
      ...pickDefined({
        gpuLayers: route.gpuLayers,
        contextLength: route.contextLength,
      }),
      ...explicitConfig,
    }
    const projector = this.projectorFor(summary, await this.catalog())
    const extraArgs = [...(config.extraArgs ?? [])]
    // A vision projector is passed as a side file; the adapter still advertises
    // text-only input, so this only matters for direct API users.
    if (projector) extraArgs.push('--mmproj', projector.path)
    const started = await this.backends.ensure({
      modelPath,
      backend,
      config: { ...config, extraArgs },
      alias: path.basename(modelPath, '.gguf'),
      port: port ?? route.port,
      signal,
    })
    return { ...started, summary, config, backend: { id: backend.id, acceleration: backend.acceleration }, recommended }
  }

  /** Choose which llama.cpp runtime to run. */
  #selectBackend(route, lmStudio, hardware) {
    if (route.llamaServerPath) {
      return {
        id: 'configured',
        serverExe: route.llamaServerPath,
        dir: path.dirname(route.llamaServerPath),
        vendorDir: route.llamaServerVendorDir,
        acceleration: 'configured',
      }
    }
    if (route.backendId && route.backendId !== 'auto') {
      const exact = lmStudio.backends.find((b) => b.id === route.backendId)
      if (exact) return exact
    }
    // `findLlamaCppBackends` already sorts CUDA ahead of Vulkan ahead of CPU using
    // the probed hardware, so the first entry is the right default.
    return lmStudio.backends[0]
  }

  /** Start LM Studio's server, launching the app first when it is not running. */
  async #startLmStudio(route, signal) {
    const lmStudio = await this.lmStudio()
    if (!lmStudio.installed && !lmStudio.lmsPath) {
      throw new Error(`LM Studio is not reachable at ${route.baseURL} and no LM Studio installation was found`)
    }
    // `lms` talks to the app over a local socket, so the app has to be up first.
    if (lmStudio.appExe) {
      try {
        launchLmStudioApp(lmStudio.appExe)
        this.log('info', 'launching LM Studio to serve the request')
      } catch (error) {
        this.log('warn', `could not launch LM Studio: ${error instanceof Error ? error.message : String(error)}`)
      }
    }
    const port = Number(new URL(route.baseURL).port || DEFAULT_LMSTUDIO_PORT)
    if (lmStudio.lmsPath) {
      try {
        await lmsServerStart(lmStudio.lmsPath, { port })
      } catch (error) {
        this.log('warn', `lms server start failed: ${error instanceof Error ? error.message : String(error)}`)
      }
    }
    await waitForServer(route.baseURL, { timeoutMs: 180000, signal, apiKey: route.apiKey })
  }

  /** LM Studio status: installation, runtimes, CLI, and server reachability. */
  async status({ refresh = false } = {}) {
    const hardware = await this.hardware(refresh)
    const lmStudio = await this.lmStudio(refresh)
    const route = this.route('lmstudio')
    const [probe, managed] = await Promise.all([
      listServerModels(route.baseURL, { apiKey: route.apiKey, timeoutMs: 2500 }),
      Promise.resolve(this.backends.list()),
    ])
    let cli = undefined
    if (lmStudio.lmsPath) {
      try {
        const result = await lmsServerStatus(lmStudio.lmsPath, { timeoutMs: 8000 })
        cli = { ok: result.code === 0, output: result.stdout || result.stderr }
      } catch (error) {
        cli = { ok: false, output: error instanceof Error ? error.message : String(error) }
      }
    }
    return {
      hardware,
      lmStudio: {
        installed: lmStudio.installed,
        installDir: lmStudio.installDir,
        lmsPath: lmStudio.lmsPath,
        modelDirs: lmStudio.modelDirs,
        backends: lmStudio.backends.map((b) => ({ id: b.id, acceleration: b.acceleration, version: b.version })),
      },
      server: {
        baseUrl: route.baseURL,
        reachable: probe.reachable,
        nativeApi: probe.native,
        models: probe.models.length,
        error: probe.error,
        cli,
      },
      managedBackends: managed,
      tokensPerSecond: Object.fromEntries(this.lastThroughput),
    }
  }

  /** Load a model into LM Studio with explicit parameters. */
  async loadIntoLmStudio(modelKey, config = {}, identifier) {
    const lmStudio = await this.lmStudio()
    const route = this.route('lmstudio')
    const summary = await this.#byPath(modelKey)
    return loadModel(route.baseURL, {
      lmsPath: lmStudio.lmsPath,
      modelKey,
      identifier,
      apiKey: route.apiKey,
      config: {
        ...config,
        ...(summary?.blockCount ? { totalLayers: summary.blockCount } : {}),
      },
    })
  }

  /** List what LM Studio currently has resident. */
  async lmStudioLoaded() {
    const lmStudio = await this.lmStudio()
    const route = this.route('lmstudio')
    return lmStudioLoadedInstances(route.baseURL, { apiKey: route.apiKey, lmsPath: lmStudio.lmsPath })
  }

  /**
   * Inspect one model without loading it: header facts, load configuration,
   * memory estimate and the reasoning behind each recommendation.
   */
  async inspect(modelId, { contextLength, gpuLayers, kvTypeK, kvTypeV, refresh = false } = {}) {
    const catalog = await this.catalog({ refresh })
    const matched = matchModel(catalog.models, modelId) ?? (await this.#byPath(modelId))
    if (!matched) throw new Error(`no GGUF file matches "${modelId}"`)
    const hardware = await this.hardware(refresh)
    const recommendation = recommendConfig(matched, hardware)
    const config = {
      ...recommendation.config,
      ...pickDefined({ contextLength, gpuLayers, kvTypeK, kvTypeV }),
      threads: recommendation.config.threads ?? hardware.cpu.recommendedThreads,
    }
    const estimate = estimateMemory(matched, config)
    const projector = this.projectorFor(matched, catalog)
    return {
      model: matched,
      projector,
      hardware,
      config,
      estimate,
      recommendation,
      // The measured numbers a running backend reported, when there is one.
      measured: this.backends.list().find((entry) => entry.modelPath === matched.path)?.measured,
      labels: { total: formatBytes(estimate.totalBytes), gpu: formatBytes(estimate.gpuBytes), ram: formatBytes(estimate.cpuBytes) },
    }
  }

  /** Force a fresh hardware and catalog read. */
  invalidate() {
    this.hardwareCache = undefined
    this.lmStudioCache = undefined
    this.catalogCache = undefined
    clearHeaderCache()
  }

  /** Stop every backend this plugin started and drop caches. */
  dispose() {
    this.backends.stopAll()
    this.modelInfoCache.clear()
    this.lastThroughput.clear()
  }
}

/** Copy only the entries whose value is neither undefined nor null. */
function pickDefined(source) {
  const out = {}
  for (const [key, value] of Object.entries(source)) {
    if (value !== undefined && value !== null) out[key] = value
  }
  return out
}
