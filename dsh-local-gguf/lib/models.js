/**
 * The local model catalog: discovery of `.gguf` files on disk, cached header
 * parsing, and the merge with whatever a running server advertises.
 *
 * Discovery walks the configured model directories plus the directories LM Studio
 * itself has indexed. A GGUF header parse costs one small read, but a directory
 * can hold many files, so results are memoized on `(path, size, mtime)` and a
 * bounded number of parses run concurrently.
 */

import fs from 'node:fs'
import path from 'node:path'
import { readGguf } from './gguf.js'
import { lmStudioModelDirs } from './lmstudio.js'

/** Guard against a symlink loop or an accidentally huge tree. */
const MAX_DEPTH = 6
/** Files parsed at once; each parse is I/O-bound on a small header read. */
const PARSE_CONCURRENCY = 4

/** `path -> { size, mtimeMs, summary }` memo for parsed GGUF headers. */
const headerCache = new Map()

/**
 * Recursively collect `.gguf` files under one root.
 *
 * `.gguf.part` and in-progress download artifacts are skipped because a partial
 * file has no valid header.
 */
async function collectGgufFiles(root, depth = 0, out = []) {
  if (depth > MAX_DEPTH) return out
  let entries
  try {
    entries = await fs.promises.readdir(root, { withFileTypes: true })
  } catch {
    return out
  }
  for (const entry of entries) {
    const full = path.join(root, entry.name)
    if (entry.isDirectory()) {
      if (entry.name.startsWith('.')) continue
      await collectGgufFiles(full, depth + 1, out)
    } else if (entry.isFile() && entry.name.toLowerCase().endsWith('.gguf')) {
      out.push(full)
    }
  }
  return out
}

/** Split a model path into the publisher/repository labels LM Studio uses. */
export function labelsForPath(filePath, rootDir) {
  const relative = rootDir ? path.relative(rootDir, filePath) : path.basename(filePath)
  const parts = relative.split(/[\\/]/).filter(Boolean)
  const file = parts.pop() ?? path.basename(filePath)
  const repository = parts.pop()
  const publisher = parts.pop()
  return { file, repository, publisher, relative }
}

/** Parse a header with the process-wide memo, refreshing when the file changes. */
async function parseWithCache(filePath) {
  let stat
  try {
    stat = await fs.promises.stat(filePath)
  } catch {
    return undefined
  }
  const cached = headerCache.get(filePath)
  if (cached && cached.size === stat.size && cached.mtimeMs === stat.mtimeMs) return cached.summary
  try {
    const summary = await readGguf(filePath)
    headerCache.set(filePath, { size: stat.size, mtimeMs: stat.mtimeMs, summary })
    return summary
  } catch (error) {
    // Record the failure so a corrupt file is reported once rather than retried
    // on every catalog refresh.
    const failure = { error: error instanceof Error ? error.message : String(error) }
    headerCache.set(filePath, { size: stat.size, mtimeMs: stat.mtimeMs, summary: failure })
    return failure
  }
}

/** Run `worker` over `items` with bounded concurrency, preserving order. */
async function mapLimited(items, limit, worker) {
  const results = new Array(items.length)
  let cursor = 0
  const runners = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (true) {
      const index = cursor++
      if (index >= items.length) return
      results[index] = await worker(items[index], index)
    }
  })
  await Promise.all(runners)
  return results
}

/**
 * Scan model directories and parse every GGUF header found.
 *
 * @param {object} options
 * @param {string[]} options.directories roots to scan.
 * @param {boolean} options.includeLmStudioDirs also scan LM Studio's own dirs.
 * @param {boolean} options.refresh ignore the header memo.
 * @returns {Promise<{models: object[], directories: string[], scanned: number, failures: object[]}>}
 */
export async function scanModels({ directories = [], includeLmStudioDirs = true, refresh = false, signal } = {}) {
  if (refresh) headerCache.clear()
  const roots = new Set()
  for (const dir of directories) {
    if (typeof dir === 'string' && dir.trim()) roots.add(path.resolve(dir.trim()))
  }
  if (includeLmStudioDirs) {
    for (const dir of await lmStudioModelDirs()) roots.add(path.resolve(dir))
  }

  const located = []
  for (const root of roots) {
    if (signal?.aborted) throw signal.reason ?? new Error('aborted')
    for (const file of await collectGgufFiles(root)) located.push({ file, root })
  }

  const failures = []
  const parsed = await mapLimited(located, PARSE_CONCURRENCY, async ({ file, root }) => {
    if (signal?.aborted) throw signal.reason ?? new Error('aborted')
    const summary = await parseWithCache(file)
    if (!summary || summary.error) {
      failures.push({ path: file, error: summary?.error ?? 'unreadable' })
      return undefined
    }
    const labels = labelsForPath(file, root)
    return {
      ...summary,
      root,
      ...labels,
      displayName: summary.name || labels.file.replace(/\.gguf$/i, ''),
    }
  })

  const models = parsed.filter(Boolean).sort((a, b) => a.displayName.localeCompare(b.displayName))
  return { models, directories: [...roots], scanned: located.length, failures }
}

/** Look up one model summary by absolute path, using the memo. */
export async function modelByPath(filePath, refresh = false) {
  if (refresh) headerCache.delete(filePath)
  const summary = await parseWithCache(filePath)
  if (!summary || summary.error) throw new Error(`cannot read GGUF header: ${summary?.error ?? filePath}`)
  return { ...summary, displayName: summary.name || path.basename(filePath) }
}

/**
 * Best-effort match between a requested model id and a file in the catalog.
 *
 * Model ids arrive in three shapes depending on which surface chose them: an
 * absolute path, LM Studio's `publisher/repo/file.gguf` key, or a bare file name.
 * Only the first is unambiguous, so the others are matched by suffix and the
 * caller decides what to do with an ambiguous result.
 */
export function matchModel(models, id) {
  const wanted = String(id).replace(/\//g, path.sep)
  const wantedLower = wanted.toLowerCase()
  const byPath = models.find((m) => m.path.toLowerCase() === wantedLower)
  if (byPath) return byPath
  const byRelative = models.find((m) => m.relative?.toLowerCase() === wantedLower)
  if (byRelative) return byRelative
  const byFile = models.filter((m) => m.file?.toLowerCase() === path.basename(wantedLower))
  if (byFile.length === 1) return byFile[0]
  const bySuffix = models.filter((m) => m.path.toLowerCase().endsWith(wantedLower))
  if (bySuffix.length === 1) return bySuffix[0]
  return undefined
}

/** Drop every memoized header. Exposed for tests and manual refreshes. */
export function clearHeaderCache() {
  headerCache.clear()
}
