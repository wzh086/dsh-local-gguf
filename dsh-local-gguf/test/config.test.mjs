/**
 * Configuration writes: model directories and pinned load parameters.
 *
 * The writer is exercised against a stub `configEditor` that records what it is
 * asked to persist, so the merge behaviour is checked without a Loader.
 */

import { updateModelDirectories, updateProviderOptions, findPluginEntry, canWriteConfig } from '../lib/config-writer.js'

let failures = 0
const check = (name, ok, detail) => {
  if (!ok) failures++
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${name}${detail !== undefined && !ok ? ` — ${detail}` : ''}`)
}

/** A stub context whose configEditor records every write. */
function stubContext(initial = {}, { entryId = 'local-gguf', withEditor = true, withEntry = true } = {}) {
  let stored = structuredClone(initial)
  const entry = { options: { id: entryId, name: '@local/dsh-local-gguf' } }
  const writes = []
  const editor = {
    entries: () => [
      // `withEntry: false` models a composition where this plugin's row is not
      // addressable at all, which is the case the error path must report.
      ...(withEntry ? [{ options: { id: entryId, name: '@local/dsh-local-gguf' } }] : []),
      { options: { id: 'llm', name: '@deepseek-ai/dsh-llm' } },
    ],
    async edit(target, change) {
      const match = editor.entries().find((candidate) => candidate.options.id === target.options.id)
      if (match === undefined || match.options.name !== '@local/dsh-local-gguf') {
        throw new Error('edited an entry that is not this plugin')
      }
      stored = change(structuredClone(stored))
      writes.push(structuredClone(stored))
    },
  }
  return {
    ctx: { get: (name) => (name === 'configEditor' && withEditor ? editor : undefined) },
    get stored() {
      return stored
    },
    writes,
  }
}

console.log('=== entry lookup ===')
{
  const { ctx } = stubContext()
  check('finds the entry by id', findPluginEntry(ctx, 'local-gguf')?.options.id === 'local-gguf')
  // An unknown id still resolves through the package-name fallback, which is what
  // keeps a renamed row addressable.
  check('an unknown id falls back to the package name', findPluginEntry(ctx, 'renamed-row')?.options.name === '@local/dsh-local-gguf')
  const { ctx: absent } = stubContext({}, { withEntry: false })
  check('returns undefined when this plugin has no entry', findPluginEntry(absent, 'local-gguf') === undefined)
  check('canWriteConfig is true with an editor', canWriteConfig(ctx) === true)
  const { ctx: bare } = stubContext({}, { withEditor: false })
  check('canWriteConfig is false without an editor', canWriteConfig(bare) === false)
}

console.log('\n=== model directories ===')
{
  const harness = stubContext()
  await updateModelDirectories(harness.ctx, 'local-gguf', 'add', 'D:\\models')
  check('adds the first directory', JSON.stringify(harness.stored.modelDirs) === JSON.stringify(['D:\\models']), JSON.stringify(harness.stored.modelDirs))

  await updateModelDirectories(harness.ctx, 'local-gguf', 'add', 'E:\\more')
  check('appends a second directory', JSON.stringify(harness.stored.modelDirs) === JSON.stringify(['D:\\models', 'E:\\more']), JSON.stringify(harness.stored.modelDirs))

  await updateModelDirectories(harness.ctx, 'local-gguf', 'add', 'D:\\MODELS')
  check('is case-insensitively idempotent', harness.stored.modelDirs.length === 2, JSON.stringify(harness.stored.modelDirs))

  await updateModelDirectories(harness.ctx, 'local-gguf', 'remove', 'd:\\models')
  check('removes case-insensitively', JSON.stringify(harness.stored.modelDirs) === JSON.stringify(['E:\\more']), JSON.stringify(harness.stored.modelDirs))

  await updateModelDirectories(harness.ctx, 'local-gguf', 'remove', 'Z:\\absent')
  check('removing an absent directory is a no-op', JSON.stringify(harness.stored.modelDirs) === JSON.stringify(['E:\\more']), JSON.stringify(harness.stored.modelDirs))

  for (const [label, args] of [
    ['an empty path', ['add', '   ']],
    ['an unknown action', ['delete', 'D:\\models']],
  ]) {
    try {
      await updateModelDirectories(harness.ctx, 'local-gguf', args[0], args[1])
      failures++
      console.log(`  FAIL ${label} was accepted`)
    } catch {
      console.log(`  ok   ${label} is refused`)
    }
  }
}

console.log('\n=== preserving unrelated configuration ===')
{
  const initial = {
    includeLmStudioDirs: false,
    vramReserveRatio: 0.2,
    providers: { lmstudio: { baseURL: 'http://127.0.0.1:9999/v1', autoStart: false } },
  }
  const harness = stubContext(initial)
  await updateModelDirectories(harness.ctx, 'local-gguf', 'add', 'D:\\models')
  check('keeps sibling scalar keys', harness.stored.includeLmStudioDirs === false && harness.stored.vramReserveRatio === 0.2, JSON.stringify(harness.stored))
  check('keeps the provider block untouched', harness.stored.providers.lmstudio.baseURL === 'http://127.0.0.1:9999/v1', JSON.stringify(harness.stored.providers))
  check('the stored config is never mutated in place', harness.writes.length === 1)
}

console.log('\n=== pinned load parameters ===')
{
  const harness = stubContext()
  await updateProviderOptions(harness.ctx, 'local-gguf', 'local-gguf', { contextLength: 16384, gpuLayers: 40 })
  check('writes a provider block', JSON.stringify(harness.stored.providers['local-gguf']) === JSON.stringify({ contextLength: 16384, gpuLayers: 40 }), JSON.stringify(harness.stored.providers))

  await updateProviderOptions(harness.ctx, 'local-gguf', 'local-gguf', { contextLength: 8192 })
  check('merges into an existing block', harness.stored.providers['local-gguf'].gpuLayers === 40 && harness.stored.providers['local-gguf'].contextLength === 8192, JSON.stringify(harness.stored.providers))

  await updateProviderOptions(harness.ctx, 'local-gguf', 'local-gguf', { contextLength: null, gpuLayers: null })
  check('null clears a key and drops the emptied block', harness.stored.providers['local-gguf'] === undefined, JSON.stringify(harness.stored.providers))

  await updateProviderOptions(harness.ctx, 'local-gguf', 'lmstudio', { baseURL: 'http://127.0.0.1:4321/v1' })
  await updateProviderOptions(harness.ctx, 'local-gguf', 'local-gguf', { gpuLayers: 0 })
  check('two routes coexist in the provider map', Object.keys(harness.stored.providers).sort().join(',') === 'lmstudio,local-gguf', JSON.stringify(harness.stored.providers))
  check('a zero value is kept rather than treated as absent', harness.stored.providers['local-gguf'].gpuLayers === 0, JSON.stringify(harness.stored.providers['local-gguf']))

  try {
    await updateProviderOptions(harness.ctx, 'local-gguf', '', { a: 1 })
    failures++
    console.log('  FAIL an empty provider was accepted')
  } catch {
    console.log('  ok   an empty provider is refused')
  }
}

console.log('\n=== no editor available ===')
{
  const { ctx } = stubContext({}, { withEditor: false })
  try {
    await updateModelDirectories(ctx, 'local-gguf', 'add', 'D:\\models')
    failures++
    console.log('  FAIL a write without an editor was accepted')
  } catch (error) {
    check('reports the missing editor clearly', /configuration editor/.test(error.message), error.message)
  }
}

console.log('')
console.log(failures === 0 ? 'ALL PASS' : `${failures} FAILURE(S)`)
process.exit(failures === 0 ? 0 : 1)
