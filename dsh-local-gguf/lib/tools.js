/**
 * Agent tools over the local-model runtime.
 *
 * These are the surfaces the model itself uses to answer "what can I run here?",
 * "how much memory will it take?", and "start it". Every tool returns JSON so a
 * result stays readable and diffable in the transcript.
 *
 * Tool definitions are built with `defineTool` from `lib/dsh-compat.js`, which
 * reproduces the `{name, description, parameters, output, execute}` shape
 * `ctx.tools.register()` validates.
 */

import { defineTool, objectSchema } from './dsh-compat.js'
import { formatBytes } from './hardware.js'
import { toLlamaServerArgs } from './estimate.js'

/** The compact shape a tool returns for one model. */
function modelRow(model, hardware) {
  return {
    id: model.path,
    name: model.displayName,
    kind: model.kind,
    arch: model.arch,
    quantization: model.quantization,
    parameterLabel: model.sizeLabel || undefined,
    fileSize: formatBytes(model.fileSize),
    weights: formatBytes(model.weightsBytes),
    layers: model.blockCount,
    nativeContext: model.contextLength,
    attention: model.headCount
      ? { queryHeads: model.headCount, kvHeads: model.headCountKv, headDim: model.keyLength }
      : undefined,
    experts: model.expertCount ? { count: model.expertCount, usedPerToken: model.expertUsedCount } : undefined,
    slidingWindow: model.slidingWindow,
    hasChatTemplate: model.hasChatTemplate,
    fitsVramEntirely:
      hardware?.totalVramBytes > 0 ? model.weightsBytes <= hardware.totalVramBytes * 0.9 : false,
  }
}

/** The compact shape a tool returns for one memory estimate. */
function estimateRow(estimate) {
  return {
    total: formatBytes(estimate.totalBytes),
    gpu: formatBytes(estimate.gpuBytes),
    ram: formatBytes(estimate.cpuBytes),
    breakdown: {
      weights: formatBytes(estimate.weightsBytes),
      kvCache: formatBytes(estimate.kvCacheBytes),
      computeAndOverhead: formatBytes(estimate.computeBytes),
    },
    totalBytes: estimate.totalBytes,
    gpuBytes: estimate.gpuBytes,
    ramBytes: estimate.cpuBytes,
  }
}

/** The compact shape a tool returns for llama.cpp's own measured allocation. */
function measuredRow(measured) {
  if (!measured || measured.totalBytes === undefined) {
    return measured?.contextSize
      ? { contextSize: measured.contextSize, unavailable: 'llama.cpp reported no allocation lines' }
      : undefined
  }
  return {
    gpu: formatBytes(measured.gpuBytes),
    ram: formatBytes(measured.cpuBytes),
    total: formatBytes(measured.totalBytes),
    gpuBytes: measured.gpuBytes,
    ramBytes: measured.cpuBytes,
    breakdown: {
      weights: `${formatBytes(measured.weightsGpuBytes)} GPU + ${formatBytes(measured.weightsCpuBytes)} RAM`,
      kvCache: `${formatBytes(measured.kvGpuBytes)} GPU + ${formatBytes(measured.kvCpuBytes)} RAM`,
      compute: `${formatBytes(measured.computeGpuBytes)} GPU + ${formatBytes(measured.computeCpuBytes)} RAM`,
    },
    contextSize: measured.contextSize,
    offloadedLayers: measured.offloadedLayers,
    byDevice: {
      weights: measured.weightsByDevice,
      kvCache: measured.kvByDevice,
      compute: measured.computeByDevice,
    },
  }
}

/**
 * Register every local-model tool on one context.
 *
 * @param {object} ctx owning context; tools are disposed with it.
 * @param {import('./runtime.js').LocalRuntime} runtime
 */
export function registerLocalModelTools(ctx, runtime) {
  ctx.tools.register(
    defineTool({
      name: 'local_models',
      description:
        'List the GGUF models available on this machine, with architecture, quantization, size, and whether each fits in the GPU memory detected here. Use this before recommending or starting a local model.',
      parameters: objectSchema({
        refresh: { type: 'boolean', description: 'Rescan model directories instead of using the cached catalog.' },
        include_non_chat: {
          type: 'boolean',
          description: 'Also list vision projectors (mmproj) and embedding models, which cannot serve a chat request.',
        },
      }),
      render: (_args, value) => [{ type: 'text', text: JSON.stringify(value, null, 2) }],
      async execute(args) {
        const catalog = await runtime.catalog({ refresh: args.refresh === true })
        const hardware = await runtime.hardware(args.refresh === true)
        const models = args.include_non_chat === true ? catalog.models : catalog.chatModels
        return {
          hardware: {
            cpu: `${hardware.cpu.model} (${hardware.cpu.logicalCores} logical / ~${hardware.cpu.physicalCores} physical cores)`,
            memory: formatBytes(hardware.memory.totalBytes),
            gpus: hardware.gpus.map((gpu) => ({
              name: gpu.name,
              vram: formatBytes(gpu.vramBytes),
              dedicated: !gpu.integrated,
              source: gpu.source,
            })),
            usableVram: formatBytes(hardware.totalVramBytes),
          },
          modelDirectories: catalog.directories,
          count: models.length,
          models: models.map((model) => modelRow(model, hardware)),
          ...(catalog.failures.length ? { unreadable: catalog.failures } : {}),
        }
      },
    }),
  )

  ctx.tools.register(
    defineTool({
      name: 'local_model_recommend',
      description:
        'Recommend load parameters for one local GGUF model on this machine: GPU layers, context length, KV cache type, batch sizes and thread count, with the memory estimate for that configuration and the reasoning behind it. Also returns the equivalent llama-server command line.',
      parameters: objectSchema(
        {
          model: {
            type: 'string',
            description: 'Model id from local_models (an absolute .gguf path), or a publisher/repository/file.gguf key.',
          },
          context_length: { type: 'number', description: 'Override the recommended context length.' },
          gpu_layers: { type: 'number', description: 'Override the recommended number of layers offloaded to the GPU.' },
          kv_cache_type: { type: 'string', description: 'Override the KV cache type: f16, q8_0 or q4_0.' },
        },
        ['model'],
      ),
      render: (_args, value) => [{ type: 'text', text: JSON.stringify(value, null, 2) }],
      async execute(args) {
        const inspection = await runtime.inspect(args.model, {
          contextLength: args.context_length,
          gpuLayers: args.gpu_layers,
          kvTypeK: args.kv_cache_type,
          kvTypeV: args.kv_cache_type,
        })
        const { model, config, estimate, recommendation } = inspection
        const lmStudio = await runtime.lmStudio()
        return {
          model: modelRow(model, inspection.hardware),
          projector: inspection.projector ? inspection.projector.path : undefined,
          recommendedConfig: {
            contextLength: config.contextLength,
            gpuLayers: config.gpuLayers,
            totalLayers: model.blockCount,
            kvCacheType: { k: config.kvTypeK, v: config.kvTypeV },
            offloadKvToGpu: config.offloadKvToGpu,
            flashAttention: config.flashAttention,
            batchSize: config.batchSize,
            physicalBatchSize: config.ubatchSize,
            threads: config.threads,
            mmap: config.mmap,
          },
          estimatedMemory: estimateRow(estimate),
          // llama.cpp's own numbers for a running backend, when there is one. The
          // estimate is the prediction; this is the measurement.
          measuredMemory: measuredRow(inspection.measured),
          kvCachePerToken:
            estimate.kvCacheBytes && config.contextLength
              ? formatBytes(estimate.kvCacheBytes / config.contextLength)
              : undefined,
          rationale: recommendation.rationale,
          warnings: recommendation.warnings,
          llamaServerArgs: toLlamaServerArgs(model, config, {
            modelPath: model.path,
            alias: String(model.file ?? '').replace(/\.gguf$/i, ''),
            port: 8080,
          }),
          availableRuntimes: lmStudio.backends.slice(0, 3).map((b) => ({ id: b.id, acceleration: b.acceleration })),
          ...(inspection.measured ? { measuredByBackend: inspection.measured } : {}),
        }
      },
    }),
  )

  ctx.tools.register(
    defineTool({
      name: 'local_backend',
      description:
        'Inspect or control the local inference backends: the LM Studio server, its installed llama.cpp runtimes, and the llama-server processes this plugin started directly on GGUF files. Actions: status, start, stop, stop_all.',
      parameters: objectSchema(
        {
          action: {
            type: 'string',
            enum: ['status', 'start', 'stop', 'stop_all'],
            description: 'One of: status, start, stop, stop_all.',
          },
          model: { type: 'string', description: 'For start/stop: model id from local_models (an absolute .gguf path).' },
          context_length: { type: 'number', description: 'For start: override the context length.' },
          gpu_layers: { type: 'number', description: 'For start: override the number of GPU layers.' },
          port: { type: 'number', description: 'For start: bind this port instead of choosing a free one.' },
        },
        ['action'],
      ),
      render: (_args, value) => [{ type: 'text', text: JSON.stringify(value, null, 2) }],
      async execute(args) {
        const action = String(args.action ?? '').toLowerCase()

        if (action === 'status') {
          const status = await runtime.status()
          return {
            lmStudio: status.lmStudio,
            server: status.server,
            managedBackends: status.managedBackends.map((backend) => ({
              model: backend.modelPath,
              baseUrl: backend.baseUrl,
              pid: backend.pid,
              running: backend.running,
              measured: measuredRow(backend.measured),
            })),
            hardware: {
              usableVram: formatBytes(status.hardware.totalVramBytes),
              ram: formatBytes(status.hardware.memory.totalBytes),
              gpus: status.hardware.gpus.map(
                (g) => `${g.name} (${formatBytes(g.vramBytes)}${g.integrated ? ', integrated' : ''})`,
              ),
            },
          }
        }

        if (action === 'start') {
          if (!args.model) throw new Error('action "start" requires a model')
          const catalog = await runtime.catalog()
          const matched =
            catalog.models.find((m) => m.path === args.model) ??
            catalog.chatModels.find((m) => m.displayName === args.model)
          if (!matched) throw new Error(`no GGUF file matches "${args.model}"; call local_models first`)
          const started = await runtime.startBackend(matched.path, {
            port: args.port,
            config: {
              ...(args.context_length !== undefined ? { contextLength: args.context_length } : {}),
              ...(args.gpu_layers !== undefined ? { gpuLayers: args.gpu_layers } : {}),
            },
          })
          return {
            started: true,
            model: matched.path,
            baseUrl: started.baseUrl,
            pid: started.pid,
            runtime: started.backend,
            config: started.config,
            estimate: estimateRow(started.recommended.estimate),
            measured: measuredRow(started.measured),
            openAiCompatible: `${started.baseUrl}/chat/completions`,
          }
        }

        if (action === 'stop') {
          if (!args.model) throw new Error('action "stop" requires a model')
          return { stopped: runtime.backends.stop(args.model), model: args.model }
        }

        if (action === 'stop_all') {
          const before = runtime.backends.list().length
          runtime.backends.stopAll()
          return { stopped: before }
        }

        throw new Error(`unknown action "${args.action}"; expected status, start, stop or stop_all`)
      },
    }),
  )
}
