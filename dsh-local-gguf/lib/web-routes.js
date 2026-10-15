/**
 * HTTP endpoints for the plugin's Client half.
 *
 * The Web UI needs the catalog, the hardware probe and the estimates, none of
 * which the Client can compute — it has no filesystem, no child processes and no
 * GPU access. A Host service with `@Remote` methods would need the Typert code
 * generator, which a workspace bundle cannot run, so these routes are the direct
 * carrier instead.
 *
 * GET routes only ever serve local machine metadata (model paths, quantizations,
 * VRAM figures); nothing credential-bearing is exposed. The three POST routes are
 * the panel's write actions — add or remove a model directory, pin load
 * parameters, and run an LM Studio download — and each one either lands in the
 * profile configuration through `configEditor` or is reported back verbatim.
 */

import fs from 'node:fs'
import { formatBytes } from './hardware.js'
import { CONTEXT_PRESETS, KV_CACHE_TYPES, estimateMemory, recommendConfig, toLlamaServerArgs } from './estimate.js'
import { updateModelDirectories, updateProviderOptions, canWriteConfig } from './config-writer.js'
import { discoverLmStudio, runCommand } from './lmstudio.js'

/** The route prefix every endpoint lives under. */
export const ROUTE_PREFIX = '/local-gguf'

/** Largest request body accepted, so a malformed client cannot exhaust memory. */
const MAX_BODY_BYTES = 64 * 1024

/** Write one JSON response. */
function json(res, status, body) {
  const payload = JSON.stringify(body)
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'no-store',
    'content-length': Buffer.byteLength(payload),
  })
  res.end(payload)
}

/**
 * Register the plugin's endpoints.
 *
 * @param {object} ctx the plugin's context.
 * @param {import('./runtime.js').LocalRuntime} runtime
 * @param {string} entryId this plugin's Loader entry id, used for config writes.
 * @returns {boolean} whether the routes were registered.
 */
export function registerWebRoutes(ctx, runtime, entryId) {
  const webServer = ctx.get('webServer')
  if (!webServer) return false

  const handle = async (req, res, pathname, query) => {
    try {
      if (pathname === `${ROUTE_PREFIX}/status`) {
        const status = await runtime.status({ refresh: query.get('refresh') === '1' })
        return json(res, 200, {
          hardware: {
            cpu: status.hardware.cpu,
            memory: { total: status.hardware.memory.totalBytes, free: status.hardware.memory.freeBytes },
            gpus: status.hardware.gpus,
            usableVram: status.hardware.totalVramBytes,
            formatted: {
              ram: formatBytes(status.hardware.memory.totalBytes),
              vram: formatBytes(status.hardware.totalVramBytes),
            },
          },
          lmStudio: status.lmStudio,
          server: status.server,
          managedBackends: status.managedBackends.map((backend) => ({
            modelPath: backend.modelPath,
            baseUrl: backend.baseUrl,
            pid: backend.pid,
            running: backend.running,
            measured: backend.measured,
            config: backend.config,
          })),
        })
      }

      if (pathname === `${ROUTE_PREFIX}/models`) {
        const catalog = await runtime.catalog({ refresh: query.get('refresh') === '1' })
        const hardware = await runtime.hardware(false)
        return json(res, 200, {
          directories: catalog.directories,
          configuredDirectories: runtime.config().modelDirs ?? [],
          canWrite: canWriteConfig(ctx),
          failures: catalog.failures,
          usableVram: hardware.totalVramBytes,
          totalRam: hardware.memory.totalBytes,
          models: catalog.models.map((model) => ({
            id: model.path,
            name: model.displayName,
            kind: model.kind,
            arch: model.arch,
            quantization: model.quantization,
            sizeLabel: model.sizeLabel,
            fileSize: model.fileSize,
            fileSizeText: formatBytes(model.fileSize),
            weightsBytes: model.weightsBytes,
            weightsText: formatBytes(model.weightsBytes),
            blockCount: model.blockCount,
            contextLength: model.contextLength,
            headCount: model.headCount,
            headCountKv: model.headCountKv,
            keyLength: model.keyLength,
            expertCount: model.expertCount,
            expertUsedCount: model.expertUsedCount,
            slidingWindow: model.slidingWindow,
            hasChatTemplate: model.hasChatTemplate,
            // Served so the panel can render a memory bar before any interaction.
            estimate: estimateRow(recommendConfig(model, hardware).estimate),
            recommendation: recommendConfig(model, hardware).config,
          })),
        })
      }

      if (pathname === `${ROUTE_PREFIX}/inspect`) {
        const model = query.get('model')
        if (!model) return json(res, 400, { error: 'model is required' })
        const contextLength = intOrUndefined(query.get('context_length'))
        const gpuLayers = intOrUndefined(query.get('gpu_layers'))
        const kvType = KV_CACHE_TYPES.includes(query.get('kv_type')) ? query.get('kv_type') : undefined
        const batchSize = intOrUndefined(query.get('batch_size'))
        const ubatchSize = intOrUndefined(query.get('ubatch_size'))
        const threads = intOrUndefined(query.get('threads'))
        const flashAttention = query.get('flash_attention') === null ? undefined : query.get('flash_attention') === '1'
        const offloadKvToGpu = query.get('offload_kv') === null ? undefined : query.get('offload_kv') === '1'

        const inspection = await runtime.inspect(model, {
          contextLength,
          gpuLayers,
          kvTypeK: kvType,
          kvTypeV: kvType,
        })
        const config = {
          ...inspection.config,
          ...(kvType ? { kvTypeK: kvType, kvTypeV: kvType } : {}),
          ...(batchSize !== undefined ? { batchSize } : {}),
          ...(ubatchSize !== undefined ? { ubatchSize } : {}),
          ...(threads !== undefined ? { threads } : {}),
          ...(flashAttention !== undefined ? { flashAttention } : {}),
          ...(offloadKvToGpu !== undefined ? { offloadKvToGpu } : {}),
        }
        // Recompute against the overrides the panel sent, so the bar and the
        // numbers always describe the controls the user is looking at.
        const estimate = estimateMemory(inspection.model, config)
        const recommended = recommendConfig(inspection.model, inspection.hardware)
        return json(res, 200, {
          model: {
            id: inspection.model.path,
            name: inspection.model.displayName,
            arch: inspection.model.arch,
            quantization: inspection.model.quantization,
            blockCount: inspection.model.blockCount,
            contextLength: inspection.model.contextLength,
            fileSizeText: formatBytes(inspection.model.fileSize),
            weightsText: formatBytes(inspection.model.weightsBytes),
          },
          projector: inspection.projector?.path ?? null,
          config,
          estimate: estimateRow(estimate),
          measured: inspection.measured ?? null,
          rationale: recommended.rationale,
          warnings: recommended.warnings,
          // Machine-readable companions, so the panel renders the same statements
          // in the active language while the agent tools keep the prose.
          rationaleNotes: recommended.rationaleNotes ?? [],
          warningNotes: recommended.warningNotes ?? [],
          recommendedConfig: recommended.config,
          presets: {
            contextLengths: CONTEXT_PRESETS.filter((value) => value <= (inspection.model.contextLength ?? value)),
            kvCacheTypes: KV_CACHE_TYPES,
          },
          llamaServerArgs: toLlamaServerArgs(inspection.model, config, {
            modelPath: inspection.model.path,
            port: 8080,
          }),
        })
      }

      return json(res, 404, { error: `unknown endpoint ${pathname}` })
    } catch (error) {
      return json(res, 500, { error: error instanceof Error ? error.message : String(error) })
    }
  }

  ctx.effect(() =>
    webServer.register({
      kind: 'prefix',
      path: ROUTE_PREFIX,
      handler: (req, res) => {
        const url = new URL(req.url ?? '/', 'http://127.0.0.1')
        if (req.method === 'GET') return handle(req, res, url.pathname, url.searchParams)
        if (req.method === 'POST') return handleWrite(req, res, url.pathname, ctx, entryId, runtime)
        return json(res, 405, { error: `method ${req.method} is not supported` })
      },
    }),
  )
  return true
}

/** Read and parse a bounded JSON request body. */
function readBody(req) {
  return new Promise((resolve, reject) => {
    let size = 0
    const chunks = []
    req.on('data', (chunk) => {
      size += chunk.length
      if (size > MAX_BODY_BYTES) {
        reject(new Error(`request body exceeds ${MAX_BODY_BYTES} bytes`))
        req.destroy()
        return
      }
      chunks.push(chunk)
    })
    req.on('end', () => {
      const text = Buffer.concat(chunks).toString('utf8').trim()
      if (!text) return resolve({})
      try {
        const parsed = JSON.parse(text)
        if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
          reject(new Error('request body must be a JSON object'))
          return
        }
        resolve(parsed)
      } catch (error) {
        reject(new Error(`request body is not valid JSON: ${error instanceof Error ? error.message : String(error)}`))
      }
    })
    req.on('error', reject)
  })
}

/**
 * Handle one write action from the panel.
 *
 * `ctx` and `entryId` are threaded in because the write actions persist through
 * `configEditor`, which only the registering context can reach.
 */
async function handleWrite(req, res, pathname, ctx, entryId, runtime) {
  let body
  try {
    body = await readBody(req)
  } catch (error) {
    return json(res, 400, { error: error instanceof Error ? error.message : String(error) })
  }

  try {
    if (pathname === `${ROUTE_PREFIX}/directories`) {
      const action = String(body.action ?? '')
      const directory = String(body.path ?? '').trim()
      // A malformed request is the caller's fault, so it is answered with 400
      // rather than surfacing as an internal failure from the writer.
      if (action !== 'add' && action !== 'remove') {
        return json(res, 400, { error: `action must be "add" or "remove", received ${JSON.stringify(body.action)}` })
      }
      if (!directory) return json(res, 400, { error: 'path is required' })
      if (action === 'add') {
        // Validate here rather than in the writer, so the panel gets a specific
        // message about a path that is missing or is not a directory.
        let stat
        try {
          stat = await fs.promises.stat(directory)
        } catch {
          return json(res, 400, { error: `no such directory: ${directory}` })
        }
        if (!stat.isDirectory()) return json(res, 400, { error: `not a directory: ${directory}` })
      }
      if (!canWriteConfig(ctx)) {
        return json(res, 501, {
          error: 'this composition has no configuration editor, so model directories cannot be saved from here',
        })
      }
      const result = await updateModelDirectories(ctx, entryId, action, directory)
      // Force a rescan so the response already reflects the new directory.
      runtime.invalidate()
      const catalog = await runtime.catalog({ refresh: true })
      return json(res, 200, {
        saved: true,
        modelDirs: result.config.modelDirs ?? [],
        discovered: catalog.directories,
        models: catalog.chatModels.length,
      })
    }

    if (pathname === `${ROUTE_PREFIX}/options`) {
      const provider = String(body.provider ?? '')
      const values = body.values
      if (!values || typeof values !== 'object') return json(res, 400, { error: 'values must be an object' })
      if (!canWriteConfig(ctx)) {
        return json(res, 501, { error: 'this composition has no configuration editor, so parameters cannot be saved' })
      }
      const result = await updateProviderOptions(ctx, entryId, provider, values)
      runtime.invalidate()
      return json(res, 200, { saved: true, providers: result.config.providers ?? {} })
    }

    if (pathname === `${ROUTE_PREFIX}/download`) {
      const query = String(body.query ?? '').trim()
      if (!query) return json(res, 400, { error: 'a model name or search term is required' })
      const result = await downloadModel(ctx, runtime, query, body.limit)
      return json(res, result.ok ? 200 : 502, result)
    }

    return json(res, 404, { error: `unknown endpoint ${pathname}` })
  } catch (error) {
    return json(res, 500, { error: error instanceof Error ? error.message : String(error) })
  }
}

/**
 * Run an LM Studio download.
 *
 * `lms get` is the only supported path: it searches LM Studio's Hub and Hugging
 * Face and writes into the directory LM Studio already indexes, so a downloaded
 * model appears in this plugin's catalog with no further configuration. It talks
 * to the LM Studio daemon, so a cold machine needs the app started first — the
 * failure text says so rather than looking like a plugin fault.
 */
async function downloadModel(ctx, runtime, query, limit) {
  const hardware = await runtime.hardware()
  const lmStudio = await discoverLmStudio(hardware, process.env, runtime.config())
  if (!lmStudio.lmsPath) {
    return {
      ok: false,
      error: 'the LM Studio `lms` CLI was not found; install LM Studio to download models from inside Harness',
      hint: 'You can also add an existing model directory below, or download in LM Studio and press Rescan.',
    }
  }
  const args = ['get', query, '--gguf', '-y']
  if (Number.isFinite(limit) && limit > 0) args.push('--limit', String(Math.min(50, Math.round(limit))))
  try {
    const { runCommand } = await import('./lmstudio.js')
    const result = await runCommand(lmStudio.lmsPath, args, { timeoutMs: 1800000 })
    const output = `${result.stdout ?? ''}\n${result.stderr ?? ''}`.trim()
    return {
      ok: result.code === 0,
      code: result.code,
      command: `lms ${args.join(' ')}`,
      output: output.slice(-4000),
      hint:
        result.code === 0
          ? 'Press Rescan to pick up the downloaded model.'
          : 'LM Studio must be running for `lms` to work; start it and try again.',
    }
  } catch (error) {
    return {
      ok: false,
      command: `lms ${args.join(' ')}`,
      error: error instanceof Error ? error.message : String(error),
      hint: 'LM Studio must be running for `lms` to work; start it and try again.',
    }
  }
}

/** The estimate fields the panel renders, with human-readable sizes alongside. */
function estimateRow(estimate) {
  return {
    total: formatBytes(estimate.totalBytes),
    gpu: formatBytes(estimate.gpuBytes),
    ram: formatBytes(estimate.cpuBytes),
    totalBytes: estimate.totalBytes,
    gpuBytes: estimate.gpuBytes,
    ramBytes: estimate.cpuBytes,
    weightsBytes: estimate.weightsBytes,
    weightsText: formatBytes(estimate.weightsBytes),
    kvCacheBytes: estimate.kvCacheBytes,
    kvCacheText: formatBytes(estimate.kvCacheBytes),
    computeBytes: estimate.computeBytes,
    computeText: formatBytes(estimate.computeBytes),
  }
}

function intOrUndefined(value) {
  if (value === null || value === '') return undefined
  const parsed = Number(value)
  return Number.isFinite(parsed) && parsed >= 0 ? Math.round(parsed) : undefined
}
