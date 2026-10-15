/**
 * Offline test suite for the Local GGUF plugin.
 *
 * These run under plain Node with no Harness installed: the plugin imports
 * nothing outside `node:*`, and the two runtime contracts it depends on are
 * reproduced in `lib/dsh-compat.js` and re-checked here against a faithful copy of
 * the runtime's own lossless-JSON validator.
 *
 *   node test/run.mjs
 *
 * Exits non-zero on the first failing suite.
 */

import { spawn } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const here = path.dirname(fileURLToPath(import.meta.url))
const suites = [
  ['GGUF reader + estimator + recommender', 'gguf.test.mjs'],
  ['llama-server log parsing + estimator accuracy', 'backend.test.mjs'],
  ['OpenAI <-> Harness wire translation', 'wire.test.mjs'],
  ['Tool definitions and lossless-JSON output', 'tools.test.mjs'],
  ['Configuration writes (model folders, pinned parameters)', 'config.test.mjs'],
  ['Web routes and the Client module contract', 'client-and-routes.test.mjs'],
]

let failed = 0
for (const [title, file] of suites) {
  console.log(`\n${'='.repeat(68)}\n${title}  (${file})\n${'='.repeat(68)}`)
  const code = await new Promise((resolve) => {
    const child = spawn(process.execPath, [path.join(here, file)], { stdio: 'inherit' })
    child.on('close', resolve)
    child.on('error', () => resolve(1))
  })
  if (code !== 0) failed++
}

console.log(`\n${failed === 0 ? 'ALL SUITES PASSED' : `${failed} SUITE(S) FAILED`}`)
process.exit(failed === 0 ? 0 : 1)
