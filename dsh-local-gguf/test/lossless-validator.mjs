/**
 * Faithful replica of the tool-output validator in @deepseek-ai/dsh-tools, read
 * from the installed package. Used to prove tool results pass the real check
 * before the host reloads the plugin.
 */
function isIntrinsicObjectPrototype(prototype) {
  if (prototype === null || typeof prototype !== 'object') return false
  const ctor = Object.getOwnPropertyDescriptor(prototype, 'constructor')
  return ctor !== undefined && typeof ctor.value === 'function' && ctor.value.name === 'Object'
}
function hasIntrinsicConstructor(prototype, name) {
  const ctor = Object.getOwnPropertyDescriptor(prototype, 'constructor')
  return ctor !== undefined && typeof ctor.value === 'function' && ctor.value.name === name
}
function isJsonNumber(value) {
  return typeof value === 'number' && Number.isFinite(value) && !Object.is(value, -0)
}
function isPlainJsonRecord(value) {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  try {
    const prototype = Object.getPrototypeOf(value)
    return prototype === null || (typeof prototype === 'object' && isIntrinsicObjectPrototype(prototype))
  } catch {
    return false
  }
}
function hasPlainArrayPrototype(value) {
  const prototype = Object.getPrototypeOf(value)
  if (!Array.isArray(prototype) || !hasIntrinsicConstructor(prototype, 'Array')) return false
  const objectPrototype = Object.getPrototypeOf(prototype)
  return typeof objectPrototype === 'object' && objectPrototype !== null && isIntrinsicObjectPrototype(objectPrototype)
}
function isPlainJsonArray(value) {
  if (!Array.isArray(value)) return false
  try {
    if (!hasPlainArrayPrototype(value) || Reflect.ownKeys(value).length !== value.length + 1) return false
    for (let index = 0; index < value.length; index++) if (!Object.hasOwn(value, index)) return false
    return true
  } catch {
    return false
  }
}
function hasOnlyEnumerableStringKeys(value) {
  try {
    return Reflect.ownKeys(value).every(
      (key) => typeof key === 'string' && Object.prototype.propertyIsEnumerable.call(value, key),
    )
  } catch {
    return false
  }
}
export function isJsonValue(value, seen = new WeakSet()) {
  if (value === null) return true
  if (typeof value === 'string' || typeof value === 'boolean') return true
  if (typeof value === 'number') return isJsonNumber(value)
  if (typeof value !== 'object') return false
  if (seen.has(value)) return false
  seen.add(value)
  if (Array.isArray(value)) {
    if (!isPlainJsonArray(value)) return false
    return value.every((item) => isJsonValue(item, seen))
  }
  if (!isPlainJsonRecord(value) || !hasOnlyEnumerableStringKeys(value)) return false
  for (const key of Object.keys(value)) if (!isJsonValue(value[key], seen)) return false
  return true
}

export function validatorProblems(value) {
  const problems = []
  const seen = new WeakSet()
  const walk = (v, p) => {
    if (v === null) return
    const t = typeof v
    if (t === 'string' || t === 'boolean') return
    if (t === 'number') {
      if (!isJsonNumber(v)) problems.push(`${p}: bad number (${Object.is(v, -0) ? '-0' : v})`)
      return
    }
    if (t !== 'object') { problems.push(`${p}: ${t}`); return }
    if (seen.has(v)) { problems.push(`${p}: repeated reference or cycle`); return }
    seen.add(v)
    if (Array.isArray(v)) {
      if (!isPlainJsonArray(v)) problems.push(`${p}: not a plain dense array`)
      v.forEach((item, i) => walk(item, `${p}[${i}]`))
      return
    }
    if (!isPlainJsonRecord(v)) problems.push(`${p}: not a plain record`)
    if (!hasOnlyEnumerableStringKeys(v)) problems.push(`${p}: non-enumerable or symbol keys`)
    for (const [k, item] of Object.entries(v)) walk(item, `${p}.${k}`)
  }
  walk(value, '$')
  return problems
}
