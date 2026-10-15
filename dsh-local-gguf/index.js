/**
 * Local GGUF models for the DeepSeek Harness.
 *
 * This plugin makes DSH able to talk to models that run on the user's own
 * machine in the GGUF format, by registering LLM provider routes for the local
 * servers that serve them:
 *
 *  - `lmstudio`   LM Studio's OpenAI-compatible server, started on demand.
 *  - `local-gguf` a `llama-server` this plugin starts directly on one GGUF file,
 *                 using the CUDA/Vulkan/CPU runtime LM Studio already downloaded.
 *
 * Alongside the routes it ships what a local-model workflow actually needs: a
 * dependency-free GGUF header reader, a CPU/RAM/VRAM probe, a memory estimator
 * built from real tensor sizes, and a recommendation engine that turns those into
 * the load parameters DSH has no vocabulary for.
 *
 * The plugin imports nothing outside `node:*` and its own modules. A bundle
 * installed into a profile is linked from the workspace, so the profile's
 * `node_modules` is not an ancestor of this directory and `@deepseek-ai/*`
 * specifiers would not resolve; see `lib/dsh-compat.js`.
 */

import { DEFAULT_ROUTES, LocalRuntime } from './lib/runtime.js'
import { registerLocalModelTools } from './lib/tools.js'
import { registerWebRoutes } from './lib/web-routes.js'

/** Stable plugin identity; also the settings namespace for the provider directory. */
const name = 'local-gguf'

/**
 * Services this plugin cannot work without.
 *
 * `llm` is the adapter registry. Everything else is injected optionally so a
 * profile without the tool registry or settings still gets the model routes.
 */
const inject = ['llm']

/**
 * Mount the plugin.
 *
 * Registration happens in one place so disposal is uniform: the adapter, the
 * configurable-provider directory, model discovery and every tool are owned by
 * this plugin's fiber, and the managed llama-server processes are stopped with it.
 *
 * @param {object} ctx the plugin's Cordis context.
 * @param {object} [config] the raw `config` from this row's patch, read
 *   defensively: this plugin declares no Config schema, so nothing has validated
 *   or defaulted it before it arrives.
 */
function apply(ctx, config = {}) {
  const raw = config ?? {}
  const logger = {
    info: (message) => ctx.logger?.info?.(`local-gguf: ${message}`),
    warn: (message) => ctx.logger?.warn?.(`local-gguf: ${message}`),
    error: (message) => ctx.logger?.error?.(`local-gguf: ${message}`),
  }

  const runtime = new LocalRuntime({
    readConfig: () => raw,
    log: (level, message) => (logger[level] ?? logger.info)(message),
  })

  // The settings namespace is the Loader entry id when the row declares one, so a
  // profile that renames the entry keeps its Models-page address stable.
  const settingsNs = ctx.fiber?.entry?.options?.id ?? name

  // `auto: false` keeps this plugin's settings out of any automatic persisting
  // flow; the routes it owns are declared by the plugin, not created by a user.
  ctx.inject(['settings'], (child) => {
    child.effect(() => child.settings.configure({ auto: false }, ctx.fiber))
  })

  const routes = [...DEFAULT_ROUTES]
  ctx.llm.registerAdapter(routes, runtime.adapter)
  ctx.llm.registerConfigurableProviders(
    routes.map((provider) => ({
      provider,
      displayName: runtime.route(provider)?.displayName ?? provider,
      settingsNs,
      settingsPath: ['providers', provider],
    })),
  )

  // Model discovery for the configuration surfaces: interrogate whichever
  // endpoint the draft describes, without touching stored settings.
  ctx.llm.registerModelDiscovery(settingsNs, async (request) => {
    const { listServerModels } = await import('./lib/lmstudio.js')
    const probe = await listServerModels(request.baseURL, { apiKey: request.apiKey, timeoutMs: 5000 })
    if (!probe.reachable) {
      // Falling back to the on-disk catalog keeps the picker useful when no
      // server is running yet.
      const catalog = await runtime.catalog({ refresh: true })
      return catalog.chatModels.map((model) => ({
        id: model.path,
        name: model.displayName,
        contextWindow: model.contextLength,
        inputModalities: ['text'],
      }))
    }
    return probe.models.map((model) => ({
      id: model.id,
      name: model.name,
      contextWindow: model.contextWindow,
      inputModalities: ['text'],
    }))
  })

  if (raw.tools !== false) {
    ctx.inject(['tools'], (child) => {
      registerLocalModelTools(child, runtime)
    })
  }

  // Endpoints for the Client half, which has no filesystem, subprocess or GPU
  // access of its own. Absent in a composition without a web server.
  ctx.inject(['webServer'], (child) => {
    if (!registerWebRoutes(child, runtime, settingsNs)) {
      logger.warn('no web server is mounted; the Local GGUF settings panel stays hidden')
    }
  })

  // The recommendation engine's numbers move with free memory and the catalog
  // moves with the disk, so both caches are dropped whenever this plugin's
  // configuration changes.
  ctx.on('loader/volatile-update', () => {
    try {
      runtime.invalidate()
    } catch (error) {
      logger.warn(`could not refresh caches: ${error instanceof Error ? error.message : String(error)}`)
    }
  })

  // A managed llama-server holds VRAM until it is killed, so plugin unload has to
  // take the processes with it.
  ctx.effect(() => () => runtime.dispose())

  logger.info(`routes registered: ${routes.join(', ')}`)
}

export { apply, inject, name }
