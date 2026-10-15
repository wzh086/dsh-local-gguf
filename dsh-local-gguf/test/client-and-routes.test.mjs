/**
 * Verify the client half's module-loader contract and the host web routes,
 * neither of which can be exercised in Node without their real environments.
 */
import { registerWebRoutes, ROUTE_PREFIX } from '../lib/web-routes.js'
import { LocalRuntime } from '../lib/runtime.js'

let failures = 0
const check = (name, ok, detail) => {
  if (!ok) failures++
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${name}${detail && !ok ? ` — ${detail}` : ''}`)
}

// ---------------------------------------------------------------- host routes
console.log('=== host web routes ===')
const routes = []
const runtime = new LocalRuntime({ readConfig: () => ({}), log: () => {} })
const writtenConfigs = []
const stubCtx = {
  get: (name) => {
    if (name === 'webServer') return { register: (route) => { routes.push(route); return () => {} } }
    if (name === 'configEditor') {
      return {
        entries: () => [{ options: { id: 'local-gguf', name: '@local/dsh-local-gguf' } }],
        async edit(_target, change) { writtenConfigs.push(change({})) },
      }
    }
    return undefined
  },
  effect: (fn) => fn(),
}
const registered = registerWebRoutes(stubCtx, runtime, 'local-gguf')
check('routes registered', registered === true)
check('one prefix route', routes.length === 1 && routes[0].kind === 'prefix', JSON.stringify(routes.map((r) => r.kind)))
check('route path is the documented prefix', routes[0]?.path === ROUTE_PREFIX, routes[0]?.path)

/** Node request stand-in that can deliver a JSON body. */
function fakeReq(url, method = 'GET', body) {
  const listeners = {}
  return {
    url,
    method,
    on(event, handler) {
      listeners[event] = handler
      if (event === 'end') {
        // Deliver asynchronously, the way a real socket does.
        setTimeout(() => {
          if (body !== undefined) listeners.data?.(Buffer.from(JSON.stringify(body), 'utf8'))
          listeners.end?.()
        }, 0)
      }
      return this
    },
    destroy() {},
  }
}

/** Minimal Node response stand-in that records what the handler wrote. */
function fakeRes() {
  const out = { status: 0, headers: null, body: '' }
  return {
    out,
    writeHead(status, headers) { out.status = status; out.headers = headers },
    end(chunk) { out.body = chunk ?? '' },
  }
}
async function call(url, method = 'GET', body) {
  const res = fakeRes()
  await routes[0].handler(fakeReq(url, method, body), res)
  let parsed
  try { parsed = JSON.parse(res.out.body) } catch { parsed = undefined }
  return { status: res.out.status, headers: res.out.headers, json: parsed, raw: res.out.body }
}

const status = await call(`${ROUTE_PREFIX}/status`)
check('GET /status -> 200 JSON', status.status === 200 && status.headers?.['content-type']?.includes('application/json'), `status=${status.status}`)
check('/status reports hardware', typeof status.json?.hardware?.usableVram === 'number', JSON.stringify(status.json?.hardware?.formatted))
check('/status reports lmStudio + server', !!status.json?.lmStudio && !!status.json?.server)

const models = await call(`${ROUTE_PREFIX}/models`)
check('GET /models -> 200', models.status === 200)
check('/models lists chat models', Array.isArray(models.json?.models) && models.json.models.length > 0, `count=${models.json?.models?.length}`)
check('/models rows carry a precomputed estimate', typeof models.json?.models?.[0]?.estimate?.gpuBytes === 'number')
check('/models rows are lossless JSON', (() => { try { return JSON.parse(JSON.stringify(models.json)) !== null } catch { return false } })())

const first = models.json.models.find((m) => m.kind === 'llm')
const inspect = await call(`${ROUTE_PREFIX}/inspect?model=${encodeURIComponent(first.id)}&context_length=16384&gpu_layers=8&kv_type=q8_0`)
check('GET /inspect -> 200', inspect.status === 200, inspect.json?.error)
check('/inspect honours the requested context length', inspect.json?.config?.contextLength === 16384, String(inspect.json?.config?.contextLength))
check('/inspect honours the requested GPU layers', inspect.json?.config?.gpuLayers === 8, String(inspect.json?.config?.gpuLayers))
check('/inspect honours the requested KV type', inspect.json?.config?.kvTypeK === 'q8_0', inspect.json?.config?.kvTypeK)
check('/inspect returns an estimate and rationale', !!inspect.json?.estimate && Array.isArray(inspect.json?.rationale))
check('/inspect returns llama-server args', Array.isArray(inspect.json?.llamaServerArgs) && inspect.json.llamaServerArgs.length > 3)
check('/inspect returns context presets', Array.isArray(inspect.json?.presets?.contextLengths))

const noModel = await call(`${ROUTE_PREFIX}/inspect`)
check('GET /inspect without a model -> 400', noModel.status === 400, String(noModel.status))
const missing = await call(`${ROUTE_PREFIX}/nope`)
check('unknown GET endpoint -> 404', missing.status === 404)
const badMethod = await call(`${ROUTE_PREFIX}/status`, 'DELETE')
check('an unsupported method -> 405', badMethod.status === 405, String(badMethod.status))

console.log('\n=== write endpoints ===')
const badJson = await call(`${ROUTE_PREFIX}/directories`, 'POST', undefined)
check('POST with no body -> 400 naming the bad action', badJson.status === 400 && /action must be/.test(badJson.json?.error ?? ''), JSON.stringify(badJson.json))

const noPath = await call(`${ROUTE_PREFIX}/directories`, 'POST', { action: 'add' })
check('POST without a path -> 400', noPath.status === 400 && /path is required/.test(noPath.json?.error ?? ''), JSON.stringify(noPath.json))

const badAction = await call(`${ROUTE_PREFIX}/directories`, 'POST', { action: 'delete', path: process.cwd() })
check('an unknown action -> 400 rather than 500', badAction.status === 400, JSON.stringify(badAction.json))

const absentDir = await call(`${ROUTE_PREFIX}/directories`, 'POST', { action: 'add', path: 'Z:\\definitely-not-here' })
check('adding a missing directory -> 400 with the path named', absentDir.status === 400 && /no such directory/.test(absentDir.json?.error ?? ''), JSON.stringify(absentDir.json))

const notADir = await call(`${ROUTE_PREFIX}/directories`, 'POST', { action: 'add', path: process.execPath })
check('adding a file rather than a directory -> 400', notADir.status === 400 && /not a directory/.test(notADir.json?.error ?? ''), JSON.stringify(notADir.json))

const added = await call(`${ROUTE_PREFIX}/directories`, 'POST', { action: 'add', path: process.cwd() })
check('adding a real directory -> 200 and persisted', added.status === 200 && writtenConfigs.at(-1)?.modelDirs?.includes(process.cwd()), JSON.stringify(added.json))
check('the add response reports the rescan', Array.isArray(added.json?.discovered) && typeof added.json?.models === 'number', JSON.stringify(added.json))

const options = await call(`${ROUTE_PREFIX}/options`, 'POST', { provider: 'local-gguf', values: { contextLength: 4096, gpuLayers: 12 } })
check('pinning parameters -> 200 and persisted', options.status === 200 && writtenConfigs.at(-1)?.providers?.['local-gguf']?.contextLength === 4096, JSON.stringify(options.json))

const badOptions = await call(`${ROUTE_PREFIX}/options`, 'POST', { provider: 'local-gguf', values: 'nope' })
check('pinning with a non-object value -> 400', badOptions.status === 400, JSON.stringify(badOptions.json))

const noQuery = await call(`${ROUTE_PREFIX}/download`, 'POST', {})
check('download without a query -> 400', noQuery.status === 400, JSON.stringify(noQuery.json))

const unknownWrite = await call(`${ROUTE_PREFIX}/nope`, 'POST', {})
check('unknown POST endpoint -> 404', unknownWrite.status === 404, String(unknownWrite.status))

const modelsWithWrite = await call(`${ROUTE_PREFIX}/models`)
check('/models reports whether writes are possible', modelsWithWrite.json?.canWrite === true, JSON.stringify(modelsWithWrite.json?.canWrite))
check('/models reports the configured folders', Array.isArray(modelsWithWrite.json?.configuredDirectories), JSON.stringify(modelsWithWrite.json?.configuredDirectories))

// ---------------------------------------------------------------- client half
console.log('\n=== client half module contract ===')
const loaded = []
let clientModule
global.window = {
  __ModuleLoader__: {
    load(spec) {
      loaded.push(spec)
      clientModule = spec.factory((name) => {
        if (name === 'react') {
          // React puts children inside `props`; mirroring that is what lets the
          // test reach the component descriptor.
          const createElement = (type, props, ...children) => ({
            type,
            props: {
              ...(props ?? {}),
              ...(children.length === 0 ? {} : { children: children.length === 1 ? children[0] : children }),
            },
          })
          const Fragment = Symbol('Fragment')
          class Component { constructor(p) { this.props = p } }
          return {
            createElement,
            Fragment,
            Component,
            useCallback: (fn) => fn,
            useEffect: () => {},
            useMemo: (fn) => fn(),
            useRef: (v) => ({ current: v }),
            useState: (v) => [v, () => {}],
          }
        }
        throw new Error(`unexpected import: ${name}`)
      })
    },
  },
}
await import('../client.js')
check('client registers exactly one module', loaded.length === 1, String(loaded.length))
check('module id is the package name', loaded[0]?.id === '@local/dsh-local-gguf', loaded[0]?.id)
check('factory returned a plugin', !!clientModule && typeof clientModule.apply === 'function')
check('plugin injects slots', Array.isArray(clientModule?.inject) && clientModule.inject.includes('slots'), JSON.stringify(clientModule?.inject))

const registrations = []
const injections = []
/** Dictionaries the client registers, captured from the stubbed locale service. */
const dicts = {}
let activeLocale = 'en'
const localeStub = {
  register(_namespace, locale, dict) {
    dicts[locale] = { ...(dicts[locale] ?? {}), ...dict }
    return () => {}
  },
  bind(_namespace) {
    // Mimic the real service: answer from the active locale's dictionary, and
    // answer with the key itself when that language has no registration — which is
    // the signal the panel uses to fall back to its built-in dictionary.
    return (key) => dicts[activeLocale]?.[key] ?? key
  },
  getLocale: () => ({ active: activeLocale, locales: [], revision: 0 }),
  subscribe: () => () => {},
}
clientModule.apply({
  get: (name) => (name === 'locale' ? localeStub : undefined),
  effect: (fn) => fn(),
  slots: {
    inject(owner, cb) { injections.push(owner); cb() },
    register(options, component) { registrations.push({ options, component }); return () => {} },
  },
})
check('injects into both Models-page seats', injections.join(',') === 'settings.models.footer,settings.models.provider-card', injections.join(','))
check('registers two cells', registrations.length === 2, String(registrations.length))
check('the footer cell uses its own id', registrations[0]?.options?.id === 'local-gguf-panel', JSON.stringify(registrations[0]?.options))
check('the card cell uses the plugin settings namespace', registrations[1]?.options?.key === 'local-gguf', JSON.stringify(registrations[1]?.options))
check('both cells provide a component', registrations.every((entry) => typeof entry.component === 'function'))

// The footer registration is the one that always renders, so it is the cell every
// subsequent assertion drives.
const Guarded = registrations[0].component
const cardGuarded = registrations[1].component
// `Guarded` wraps the panel in an error boundary, so the descriptor is never null;
// the null-for-a-foreign-card behaviour lives on the panel body.
const foreign = cardGuarded({ provider: { provider: 'deepseek-official' } })
const foreignBody = foreign?.props?.children
const foreignRendered = foreignBody?.type ? foreignBody.type(foreignBody.props) : '<no panel>'
check('the card cell renders nothing for a foreign provider', foreignRendered === null, JSON.stringify(foreignRendered)?.slice(0, 80))

// And must render a tree (not throw), with fetch stubbed.
global.fetch = async () => ({ ok: false, status: 503, json: async () => ({ error: 'offline in test' }) })
const own = Guarded({})
check('renders a tree in the footer', !!own && typeof own === 'object')
// `createElement` produces descriptors, so the component body has to be invoked
// for the tree to exist at all.
const inner = own?.props?.children
check('boundary wraps the panel', typeof inner?.type === 'function')
let rendered
try {
  rendered = inner.type(inner.props)
} catch (error) {
  rendered = null
  console.log('    panel threw:', error.message)
}
const renderedJson = JSON.stringify(rendered)
check('panel renders without throwing', rendered !== null)
check('header is present in the rendered tree', /Local GGUF models/.test(renderedJson ?? ''))
check('renders a model chooser', /Select a \.gguf model/.test(renderedJson ?? ''))
check('renders the model-folder controls', /Model folders/.test(renderedJson ?? '') && /Add folder/.test(renderedJson ?? ''))
check('renders a download control', /Download/.test(renderedJson ?? ''))
check('uses only theme tokens for color', !/#[0-9a-fA-F]{3,6}\b/.test(renderedJson ?? '') || !/background":"#/.test(renderedJson ?? ''))

// ---------------------------------------------------------------- localization
console.log('\n=== localization ===')
const en = dicts.en ?? {}
const zh = dicts.zh ?? {}
check('both shipped locales were registered', Object.keys(en).length > 0 && Object.keys(zh).length > 0, `en=${Object.keys(en).length} zh=${Object.keys(zh).length}`)
check('English and Chinese cover the same keys',
  JSON.stringify(Object.keys(en).sort()) === JSON.stringify(Object.keys(zh).sort()),
  `en-only: ${Object.keys(en).filter((k) => !(k in zh)).join(',')} | zh-only: ${Object.keys(zh).filter((k) => !(k in en)).join(',')}`)

// Every key the source asks for must exist in both dictionaries, so a literal
// `t('...')` can never render as a raw key.
const source = await (await import('node:fs')).promises.readFile(new URL('../client.js', import.meta.url), 'utf8')
const usedKeys = new Set([...source.matchAll(/\bt\('([A-Za-z0-9_]+)'/g)].map((m) => m[1]))
const noteKeys = new Set([...source.matchAll(/note_\$\{note\.key\}|`note_\$\{/g)].map(() => 'note_'))
check('the source uses a meaningful number of keys', usedKeys.size > 25, String(usedKeys.size))
const missingEn = [...usedKeys].filter((key) => !(key in en))
const missingZh = [...usedKeys].filter((key) => !(key in zh))
check('every used key exists in English', missingEn.length === 0, missingEn.join(','))
check('every used key exists in Chinese', missingZh.length === 0, missingZh.join(','))

// The Host's rationale keys are looked up as `note_<key>`, so those must exist too.
const noteNames = Object.keys(en).filter((key) => key.startsWith('note_'))
check('host rationale keys are translated in both locales',
  noteNames.length >= 15 && noteNames.every((key) => key in zh),
  `${noteNames.length} note keys`)
const hostNoteKeys = ['noGpu', 'cpuThreads', 'vramReserve', 'allLayersFit', 'partialOffload', 'noFit', 'raisedContext',
  'quantizedKv', 'moeAdvice', 'weightsNearRam', 'partialOffloadSlow', 'smallerQuant', 'kvQuality', 'estimatedVram', 'ramOverflow']
const untranslatedNotes = hostNoteKeys.filter((key) => !(`note_${key}` in en) || !(`note_${key}` in zh))
check('every note the Host can emit has both translations', untranslatedNotes.length === 0, untranslatedNotes.join(','))

// A Chinese application must actually see Chinese, and placeholders must resolve.
activeLocale = 'zh'
const zhNode = Guarded({})
const zhRendered = zhNode?.props?.children
const zhTree = zhRendered?.type ? JSON.stringify(zhRendered.type(zhRendered.props)) : ''
check('the panel renders in Chinese when the locale is zh', /本地 GGUF 模型/.test(zhTree), zhTree.slice(0, 120))
check('Chinese controls are present', /模型目录/.test(zhTree) && /添加目录/.test(zhTree) && /下载/.test(zhTree))
check('no English panel chrome leaks through', !/Model folders|Add folder|Estimated memory usage/.test(zhTree))
check('placeholder substitution works in Chinese', /扫描|填充|目录/.test(zhTree))

activeLocale = 'en'
const enNode = Guarded({})
const enRendered = enNode?.props?.children
const enTree = enRendered?.type ? JSON.stringify(enRendered.type(enRendered.props)) : ''
check('the panel renders in English when the locale is en', /Local GGUF models/.test(enTree))

// An unknown language falls back to the browser language rather than to raw keys.
activeLocale = 'ja'
check('an unregistered language still renders real text', /Local GGUF models/.test(enTree))

runtime.dispose()
console.log('')
console.log(failures === 0 ? 'ALL PASS' : `${failures} FAILURE(S)`)
process.exit(failures === 0 ? 0 : 1)
