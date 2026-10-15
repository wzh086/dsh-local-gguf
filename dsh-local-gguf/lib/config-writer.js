/**
 * Persist this plugin's own configuration through the Loader.
 *
 * The Local GGUF panel needs to add and remove model directories, and those have
 * to survive a restart. `ctx.configEditor` is the sanctioned seam for that: it
 * validates and persists a plugin's next raw config and reconciles it through the
 * normal Loader path, so the write behaves exactly like a hand edit to the profile
 * patch — no side channel, no second source of truth.
 *
 * Only this plugin's own entry is ever addressed, and only keys this plugin
 * documents are touched; every other field of the stored config is carried
 * through unchanged.
 */

/** The package name a fallback lookup matches when the entry id is not declared. */
const PACKAGE_NAME = '@local/dsh-local-gguf'

/**
 * Find this plugin's own Loader entry.
 *
 * The row's `id` is the primary key, with the package name as a fallback so a
 * profile that renamed the row still resolves.
 *
 * @param {object} ctx a context carrying the optional `configEditor` service.
 * @param {string} entryId the id this plugin's patch declared.
 * @returns {object|undefined} the live Loader entry.
 */
export function findPluginEntry(ctx, entryId) {
  const editor = ctx.get('configEditor')
  if (!editor || typeof editor.entries !== 'function') return undefined
  const entries = editor.entries()
  return (
    entries.find((entry) => entry?.options?.id === entryId) ??
    entries.find((entry) => entry?.options?.name === PACKAGE_NAME)
  )
}

/** Whether configuration writes are possible in this composition at all. */
export function canWriteConfig(ctx) {
  return ctx.get('configEditor') !== undefined
}

/**
 * Persist a change to this plugin's configuration.
 *
 * @param {object} ctx a context carrying `configEditor`.
 * @param {string} entryId the id this plugin's patch declared.
 * @param {(current: Record<string, unknown>) => Record<string, unknown>} mutate
 *   receives a shallow copy of the stored raw config and returns the next one.
 * @returns {Promise<{changed: boolean, config: object}>}
 */
export async function updatePluginConfig(ctx, entryId, mutate) {
  const editor = ctx.get('configEditor')
  if (!editor || typeof editor.edit !== 'function') {
    throw new Error('this composition has no configuration editor, so settings cannot be saved from the UI')
  }
  const entry = findPluginEntry(ctx, entryId)
  if (!entry) {
    throw new Error(`could not find this plugin's own loader entry (looked for id "${entryId}" and name "${PACKAGE_NAME}")`)
  }
  let next
  await editor.edit(entry, (current) => {
    // `current` is the stored raw config for this row; spread it so unmanaged keys
    // (a hand-written provider block, for example) survive the write.
    next = mutate({ ...(current ?? {}) })
    return next
  })
  return { changed: true, config: next }
}

/**
 * Add or remove one model directory, keeping the list deduplicated and ordered.
 *
 * @param {object} ctx a context carrying `configEditor`.
 * @param {string} entryId the id this plugin's patch declared.
 * @param {'add'|'remove'} action
 * @param {string} directory an absolute directory path.
 */
export async function updateModelDirectories(ctx, entryId, action, directory) {
  const target = String(directory ?? '').trim()
  if (!target) throw new Error('a directory path is required')
  if (action !== 'add' && action !== 'remove') throw new Error(`unknown action "${action}"; expected add or remove`)

  return updatePluginConfig(ctx, entryId, (current) => {
    const existing = Array.isArray(current.modelDirs)
      ? current.modelDirs.filter((entry) => typeof entry === 'string')
      : []
    const without = existing.filter((entry) => entry.toLowerCase() !== target.toLowerCase())
    const next = action === 'add' ? [...without, target] : without
    return { ...current, modelDirs: next }
  })
}

/**
 * Pin load parameters for one provider route, or clear them.
 *
 * Values are written under `providers.<route>` so the recommendation engine can
 * keep deciding anything the user has not pinned. A `null` value removes the key,
 * which restores the recommended default.
 *
 * @param {object} ctx a context carrying `configEditor`.
 * @param {string} entryId the id this plugin's patch declared.
 * @param {string} provider the route to configure.
 * @param {Record<string, unknown>} values parameters to write.
 */
export async function updateProviderOptions(ctx, entryId, provider, values) {
  if (!provider) throw new Error('a provider route is required')
  if (!values || typeof values !== 'object') throw new Error('a values object is required')

  return updatePluginConfig(ctx, entryId, (current) => {
    const providers =
      current.providers && typeof current.providers === 'object' && !Array.isArray(current.providers)
        ? { ...current.providers }
        : {}
    const route = providers[provider] && typeof providers[provider] === 'object' ? { ...providers[provider] } : {}
    for (const [key, value] of Object.entries(values)) {
      if (value === null || value === undefined) delete route[key]
      else route[key] = value
    }
    // An empty route object would still materialize a settings section, so the
    // key is dropped entirely once nothing is pinned for it.
    if (Object.keys(route).length === 0) delete providers[provider]
    else providers[provider] = route
    return { ...current, providers }
  })
}
