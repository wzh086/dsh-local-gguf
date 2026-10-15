import { validatorProblems, isJsonValue } from './lossless-validator.mjs'

const mod = await import('../index.js')

const registered = []
const stub = {
  logger: { info: () => {}, warn: () => {}, error: () => {} },
  fiber: { entry: { options: { id: 'local-gguf' } } },
  get: () => undefined,
  inject: (deps, cb) => { for (const d of deps) cb(stub) },
  effect: (fn) => fn(),
  on: () => {},
  llm: { registerAdapter: () => () => {}, registerConfigurableProviders: () => () => {}, registerModelDiscovery: () => () => {} },
  settings: { configure: () => {} },
  tools: { register: (definition) => { registered.push(definition); return () => {} } },
}
mod.apply(stub, {})

let failures = 0
function report(label, value) {
  const problems = validatorProblems(value)
  const accepted = isJsonValue(value)
  if (problems.length || !accepted) {
    failures++
    console.log(`  FAIL ${label}`)
    console.log(`       validator=${accepted} problems=${JSON.stringify(problems.slice(0, 8))}`)
  } else {
    console.log(`  ok   ${label} (${JSON.stringify(value).length} bytes)`)
  }
}

console.log('=== validator self-test ===')
report('sanity: plain object', { a: 1, b: 'x', c: [1, 2], d: null })
// These must be REJECTED, proving the replica is not vacuous.
const sparseArray = [1]
sparseArray[3] = 4
for (const [label, bad] of [
  ['undefined property', { a: undefined }],
  ['negative zero', { a: -0 }],
  ['NaN', { a: NaN }],
  ['sparse array', sparseArray],
  ['Map instance', { a: new Map() }],
  ['function', { a: () => 1 }],
]) {
  const rejected = !isJsonValue(bad)
  console.log(`  ${rejected ? 'ok  ' : 'FAIL'} replica rejects ${label}`)
  if (!rejected) failures++
}

console.log('\n=== real tool outputs ===')
const models = registered.find((t) => t.name === 'local_models')
const recommend = registered.find((t) => t.name === 'local_model_recommend')
const backend = registered.find((t) => t.name === 'local_backend')

const modelsValue = await models.execute({}, {})
report('local_models {}', modelsValue)
report('local_models {refresh:true}', await models.execute({ refresh: true }, {}))

const target = modelsValue.models.find((m) => m.name.includes('Lite')) ?? modelsValue.models[0]
report('local_model_recommend', await recommend.execute({ model: target.id }, {}))
report(
  'local_model_recommend overrides',
  await recommend.execute({ model: target.id, context_length: 32768, gpu_layers: 20, kv_cache_type: 'q8_0' }, {}),
)
report('local_backend status', await backend.execute({ action: 'status' }, {}))

console.log('\n=== argument validation ===')
for (const [label, args] of [
  ['missing required model', {}],
  ['wrong type', { model: 42 }],
  ['bad enum', { action: 'nonsense' }],
]) {
  try {
    await (label.includes('model') ? recommend : backend).execute(args, {})
    failures++
    console.log(`  FAIL ${label}: expected a rejection`)
  } catch (error) {
    console.log(`  ok   ${label}: ${error.message}`)
  }
}

console.log('')
console.log(failures === 0 ? 'ALL PASS' : `${failures} FAILURE(S)`)
process.exit(failures === 0 ? 0 : 1)
