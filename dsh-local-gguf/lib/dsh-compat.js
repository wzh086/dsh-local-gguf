/**
 * The small slice of Harness runtime shape this plugin needs, implemented
 * locally.
 *
 * A bundle installed into a profile lives outside the Harness installation, so
 * bare `@deepseek-ai/*` specifiers do not resolve from it: Node resolves an ESM
 * import relative to the importing module's real path, and the profile's
 * `node_modules` is not an ancestor of a linked package directory. The registry
 * versions on npm (`0.0.1-rc.1`) do not match the running runtime
 * (`0.2.0-rc.2`) either, so depending on them would be worse than not depending
 * on them.
 *
 * Both contracts below are therefore reproduced against the running runtime's
 * own behaviour, which was read from the installed packages:
 *
 *  - `ctx.llm.registerAdapter()` stores the adapter without an `instanceof`
 *    check, so a plain class with the same method names is accepted.
 *  - A thrown adapter failure is normalised by reading the error's own `failure`
 *    data property (validated as `{message, code, ...}`) and comparing its
 *    `code` with the error's own `code`. An `Error` carrying those two own
 *    properties takes exactly the same path as a real `LlmError`.
 *  - `ctx.tools.register()` requires `{name, description, parameters, output:
 *    {schema, render}, execute}` and validates `output.schema` as JSON Schema.
 */

/**
 * A provider-neutral model-call failure.
 *
 * Mirrors the runtime's `LlmError`: an `Error` whose `code` routes the failure
 * and whose `failure` property carries the serializable facts the runtime
 * snapshots into the terminal `finish` chunk.
 */
export class LlmError extends Error {
  /**
   * @param {string} message non-empty human-readable summary.
   * @param {string} code non-empty stable machine-routable code.
   * @param {{cause?: unknown, status?: number, providerRetryAfterMs?: number, requestId?: string}} [options]
   */
  constructor(message, code, options) {
    if (typeof message !== 'string' || message.length === 0) throw new Error('LlmError message must be a non-empty string')
    if (typeof code !== 'string' || code.length === 0) throw new Error('LlmError code must be a non-empty string')
    super(message, options?.cause === undefined ? undefined : { cause: options.cause })
    // `code` and `failure` must be OWN data properties: the runtime reads them
    // with `Object.getOwnPropertyDescriptor` to avoid invoking SDK accessors.
    this.code = code
    this.name = 'LlmError'
    this.failure = Object.freeze({
      message,
      code,
      ...(options?.status !== undefined ? { status: options.status } : {}),
      ...(options?.providerRetryAfterMs !== undefined ? { providerRetryAfterMs: options.providerRetryAfterMs } : {}),
      ...(options?.requestId !== undefined ? { requestId: options.requestId } : {}),
    })
  }
}

/** Base class for adapters registered through `ctx.llm.registerAdapter()`. */
export class LlmAdapter {
  /** @param {string} provider */
  providerInfo(provider) {
    return { id: provider, name: provider }
  }

  /** @param {string} _provider */
  providerRetryPolicy(_provider) {
    return undefined
  }

  /**
   * @param {string} _provider
   * @param {string} _model
   */
  imageRequestPricing(_provider, _model) {
    return undefined
  }

  /** @param {string} _provider */
  async listModels(_provider) {
    return []
  }

  /**
   * @param {string} provider
   * @param {string} model
   */
  async resolveModel(provider, model) {
    return { provider, id: model, name: model }
  }

  /**
   * @param {string} provider
   * @param {string} model
   */
  async prepareCall(provider, model) {
    return { model: await this.resolveModel(provider, model), stream: (options) => this.stream(options) }
  }

  /** @param {object} _options */
  async *stream(_options) {
    throw new LlmError('this adapter does not implement stream()', 'INVALID_ADAPTER')
  }
}

/**
 * A tool argument validation failure, shaped like the runtime's own.
 *
 * The registry surfaces it to the model as a correctable call error rather than
 * a plugin fault.
 */
export class ToolArgsError extends Error {
  /** @param {string[]} violations */
  constructor(violations) {
    super(`invalid arguments: ${violations.join('; ')}`)
    this.name = 'ToolArgsError'
    this.violations = violations
  }
}

/** Check one value against the small JSON Schema subset this plugin uses. */
function checkValue(value, schema, path, violations) {
  if (value === undefined || value === null) {
    if (schema.required === true || schema.type === undefined) return
  }
  const type = schema.type
  if (type === 'number' && typeof value !== 'number') violations.push(`${path} must be a number`)
  else if (type === 'boolean' && typeof value !== 'boolean') violations.push(`${path} must be a boolean`)
  else if (type === 'string' && typeof value !== 'string') violations.push(`${path} must be a string`)
  else if (type === 'array' && !Array.isArray(value)) violations.push(`${path} must be an array`)
  else if (type === 'object' && (typeof value !== 'object' || Array.isArray(value))) violations.push(`${path} must be an object`)
  if (schema.enum !== undefined && value !== undefined && !schema.enum.includes(value)) {
    violations.push(`${path} must be one of ${schema.enum.join(', ')}`)
  }
  if (Array.isArray(schema.items) === false && schema.items !== undefined && Array.isArray(value)) {
    for (const [index, item] of value.entries()) checkValue(item, schema.items, `${path}[${index}]`, violations)
  }
}

/**
 * Deep-remove `undefined` members and coerce non-finite numbers to `null`.
 *
 * Tool results are validated as lossless JSON by the registry, and a property
 * whose value is `undefined` survives object construction but not that check, so
 * every result is normalized rather than each call site remembering to.
 *
 * @param {unknown} value
 * @returns {unknown}
 */
export function toLosslessJson(value) {
  if (value === undefined) return undefined
  if (value === null) return null
  if (typeof value === 'number') return Number.isFinite(value) ? value : null
  if (typeof value === 'bigint') return value.toString()
  if (Array.isArray(value)) return value.map((item) => toLosslessJson(item) ?? null)
  if (typeof value === 'object') {
    if (value instanceof Date) return value.toISOString()
    if (value instanceof Error) return { name: value.name, message: value.message, code: value.code }
    const out = {}
    for (const [key, item] of Object.entries(value)) {
      const normalized = toLosslessJson(item)
      // Dropping the key entirely is what makes the JSON lossless: `undefined`
      // is not a JSON value, and `null` would claim the fact is known-absent.
      if (normalized !== undefined) out[key] = normalized
    }
    return out
  }
  if (typeof value === 'function' || typeof value === 'symbol') return undefined
  return value
}

/**
 * Build one tool definition for `ctx.tools.register()`.
 *
 * Arguments are validated before `execute` runs, so a malformed model call is
 * reported as a call error the model can correct rather than reaching plugin code.
 *
 * @param {object} spec
 * @param {string} spec.name
 * @param {string} spec.description
 * @param {Record<string, unknown>} spec.parameters JSON Schema for the arguments.
 * @param {(args: any, exec: any) => unknown} spec.execute
 * @param {(args: any, value: any) => unknown[]} spec.render
 */
export function defineTool(spec) {
  const schema = spec.parameters ?? { type: 'object', properties: {} }
  const properties = schema.properties ?? {}
  const required = schema.required ?? []
  const validate = (args) => {
    const violations = []
    if (args === null || typeof args !== 'object') return ['arguments must be an object']
    for (const name of required) {
      if (args[name] === undefined || args[name] === null) violations.push(`${name} is required`)
    }
    for (const [name, propertySchema] of Object.entries(properties)) {
      if (args[name] === undefined || args[name] === null) continue
      checkValue(args[name], propertySchema, name, violations)
    }
    return violations
  }
  return {
    name: spec.name,
    description: spec.description,
    parameters: schema,
    output: {
      schema: { type: 'object', additionalProperties: true },
      render: (args, value) => spec.render(args, value),
    },
    async execute(args, exec) {
      const violations = validate(args)
      if (violations.length > 0) throw new ToolArgsError(violations)
      return jsonRoundTrip(toLosslessJson(await spec.execute(args, exec)))
    },
  }
}

/**
 * Reduce a value to exactly what JSON can represent.
 *
 * `toLosslessJson` already removes what JSON cannot express, but the registry's
 * validator is stricter than `JSON.stringify` in two ways a plain object walk
 * misses: it rejects negative zero, and it requires plain prototypes, dense
 * arrays, and only own enumerable string keys. Round-tripping through JSON
 * satisfies all of those by construction, because the result *is* parsed JSON.
 *
 * @param {unknown} value
 * @returns {unknown}
 */
function jsonRoundTrip(value) {
  try {
    return JSON.parse(JSON.stringify(value))
  } catch {
    // A cycle or a value JSON cannot reach at all: fall back to the walk, whose
    // output is acyclic by construction.
    return toLosslessJson(value)
  }
}

/** Build an object JSON Schema from a compact `name -> property` description. */
export function objectSchema(properties, required = []) {
  return { type: 'object', properties, required, additionalProperties: false }
}
