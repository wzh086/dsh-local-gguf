/**
 * The DSH LLM adapter for locally served GGUF models.
 *
 * One adapter instance serves every route this plugin owns, because all of them
 * speak the same OpenAI chat-completions dialect; the route only decides how a
 * base URL is obtained and whether a backend may be started on demand. The wire
 * translation itself lives in `lib/openai-wire.js`, which has no Harness imports
 * and can be tested on its own.
 */

import { LlmAdapter, LlmError } from './dsh-compat.js'
import { WireError, sseEvents, toChatMessages, toChatTools, translateStream } from './openai-wire.js'

export { normalizeBaseUrl } from './lmstudio.js'

/** HTTP status codes that mean "come back later" rather than "your request is wrong". */
const RETRYABLE_STATUS = new Set([408, 409, 425, 429, 500, 502, 503, 504])

/** Map an HTTP failure onto the provider-neutral failure codes DSH routes on. */
function failureForStatus(status, message) {
  if (status === 401 || status === 403) {
    return new LlmError(`local model server rejected the credential: ${message}`, 'AUTH')
  }
  if (status === 429) return new LlmError(`local model server is busy: ${message}`, 'RATE_LIMIT')
  if (status === 400 && /context|too (?:long|large)|exceed|n_ctx/i.test(message)) {
    return new LlmError(`local model server refused the request as too long: ${message}`, 'CONTEXT_WINDOW_EXCEEDED')
  }
  if (RETRYABLE_STATUS.has(status)) {
    return new LlmError(`local model server unavailable: ${message}`, 'TRANSPORT')
  }
  return new LlmError(`local model server rejected the request: ${message}`, 'PROVIDER_ERROR')
}

/** Re-throw a wire failure as the Harness failure taxonomy. */
function asLlmError(error, fallbackMessage) {
  if (error instanceof LlmError) return error
  if (error instanceof WireError) return new LlmError(fallbackMessage ?? error.message, error.code, { cause: error })
  return new LlmError(fallbackMessage ?? (error instanceof Error ? error.message : String(error)), 'TRANSPORT', { cause: error })
}

/**
 * The adapter registered for every route this plugin owns.
 *
 * It resolves a route to a base URL and an optional managed backend through
 * `runtime`, then speaks OpenAI chat completions to it.
 */
export class LocalGgufAdapter extends LlmAdapter {
  /**
   * @param {import('./runtime.js').LocalRuntime} runtime plugin-owned runtime.
   */
  constructor(runtime) {
    super()
    this.runtime = runtime
  }

  providerInfo(provider) {
    const route = this.runtime.route(provider)
    return { id: provider, name: route?.displayName ?? provider }
  }

  providerRetryPolicy() {
    // Local servers fail fast and restart quickly, so the default policy is right;
    // anything stronger belongs to `dsh-llm-retry`.
    return undefined
  }

  listModels(provider) {
    return this.runtime.listModels(provider)
  }

  resolveModel(provider, model, signal) {
    return this.runtime.resolveModel(provider, model, signal)
  }

  prepareCall(provider, model, _signal) {
    const route = this.runtime.route(provider)
    return Promise.resolve({
      model: this.runtime.resolveModelSync(provider, model),
      stream: (options) => this.generate(options, route),
    })
  }

  stream(options) {
    return this.generate(options, this.runtime.route(options.provider))
  }

  /** Resolve the endpoint, POST the request, and translate the reply. */
  async *generate(options, route) {
    if (!route) throw new LlmError(`no local route is configured for provider "${options.provider}"`, 'NO_ADAPTER')
    if (options.stop !== undefined) {
      // llama.cpp accepts `stop`, but DSH only surfaces it for adapters that
      // declare support; refusing is louder than silently dropping it.
      throw new LlmError('local GGUF routes do not support GenerateOptions.stop', 'UNSUPPORTED_OPTION')
    }

    let connection
    try {
      connection = await this.runtime.connectionFor(options.provider, options.model, options.signal)
    } catch (error) {
      throw new LlmError(
        `cannot prepare a local backend for "${options.model}": ${error instanceof Error ? error.message : String(error)}`,
        'TRANSPORT',
        { cause: error },
      )
    }

    const body = {
      model: connection.servedModelId,
      messages: toChatMessages(options),
      stream: true,
      // Asking for usage in the final frame is what makes token accounting work;
      // servers that ignore it simply omit the field.
      stream_options: { include_usage: true },
      ...(options.temperature !== undefined ? { temperature: options.temperature } : {}),
      ...(options.maxTokens !== undefined ? { max_tokens: options.maxTokens } : {}),
      ...(() => {
        const tools = toChatTools(options.tools)
        return tools ? { tools, tool_choice: 'auto' } : {}
      })(),
    }

    const controller = new AbortController()
    const signals = options.signal ? [options.signal, controller.signal] : [controller.signal]
    let response
    try {
      response = await fetch(`${connection.baseUrl}/chat/completions`, {
        method: 'POST',
        signal: AbortSignal.any(signals),
        headers: {
          'content-type': 'application/json',
          accept: 'text/event-stream',
          ...(connection.apiKey ? { authorization: `Bearer ${connection.apiKey}` } : {}),
        },
        body: JSON.stringify(body),
      })
    } catch (error) {
      if (options.signal?.aborted) throw new LlmError('local model request aborted', 'ABORTED', { cause: error })
      throw new LlmError(
        `cannot reach the local model server at ${connection.baseUrl}: ${error instanceof Error ? error.message : String(error)}`,
        'TRANSPORT',
        { cause: error },
      )
    }

    if (!response.ok) {
      const detail = await response.text().catch(() => '')
      throw failureForStatus(response.status, detail.slice(0, 500) || response.statusText)
    }
    if (!response.body) throw new LlmError('local model server returned no body', 'TRANSPORT')

    try {
      yield* translateStream(sseEvents(response.body, options.signal), { signal: options.signal })
    } catch (error) {
      if (options.signal?.aborted) throw new LlmError('local model request aborted', 'ABORTED', { cause: error })
      throw asLlmError(error, `local model stream failed: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      controller.abort()
    }
  }
}
