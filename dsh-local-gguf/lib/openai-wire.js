/**
 * Translation between DSH's message/stream vocabulary and the OpenAI
 * `chat/completions` dialect that local servers speak.
 *
 * This module is deliberately free of Harness imports: it is the part most likely
 * to be wrong, so it can be exercised on its own against recorded streams.
 * Failures are thrown as `Error` carrying a `code`, which the adapter converts
 * into the provider-neutral `LlmError` taxonomy.
 */

/** An `Error` that already carries a provider-neutral failure code. */
export class WireError extends Error {
  constructor(message, code) {
    super(message)
    this.name = 'WireError'
    this.code = code
  }
}

/**
 * Render one DSH content block as the text an OpenAI-compatible message carries.
 *
 * Images and files are not sent: this adapter advertises text-only input, so DSH
 * replaces durable images with deterministic placeholder text before dispatch. A
 * block that still reaches here is summarised rather than dropped silently.
 */
function blockText(block) {
  switch (block?.type) {
    case 'text':
      return block.text ?? ''
    case 'reasoning':
      // Provider-native thinking is not replayed back to local models; smaller
      // instruction-tuned templates tend to treat it as content.
      return ''
    case 'image':
      return '[image omitted: the selected local model is configured as text-only]'
    case 'file':
      return `[file: ${block.attachment?.name ?? 'attachment'}]`
    default:
      return ''
  }
}

/** Join the text blocks of one message. */
function messageText(message) {
  const parts = []
  for (const block of message.content ?? []) {
    const text = blockText(block)
    if (text) parts.push(text)
  }
  return parts.join('')
}

/**
 * Convert DSH request messages into OpenAI chat messages.
 *
 * The two shapes that need care are assistant turns carrying tool calls (which
 * become `tool_calls` with stringified arguments, and whose text may be empty)
 * and tool results (which become `role: "tool"` messages keyed by call id).
 */
export function toChatMessages(options) {
  const messages = []
  if (typeof options.system === 'string' && options.system.trim()) {
    messages.push({ role: 'system', content: options.system })
  }
  for (const message of options.messages ?? []) {
    if (message.role === 'system' || message.role === 'developer') {
      const text = messageText(message)
      if (text) messages.push({ role: 'system', content: text })
      continue
    }
    if (message.role === 'tool') {
      messages.push({
        role: 'tool',
        tool_call_id: String(message.toolCallId ?? message.source?.callId ?? ''),
        content: messageText(message) || '(no output)',
      })
      continue
    }
    if (message.role === 'assistant') {
      const toolCalls = []
      for (const block of message.content ?? []) {
        if (block?.type !== 'tool-call') continue
        toolCalls.push({
          id: String(block.id),
          type: 'function',
          function: { name: block.name, arguments: block.arguments || '{}' },
        })
      }
      const text = messageText(message)
      // An assistant turn that is neither text nor a tool call carries nothing a
      // local model can use, and an empty content field upsets some templates.
      if (!text && !toolCalls.length) continue
      const entry = { role: 'assistant', content: text }
      if (toolCalls.length) entry.tool_calls = toolCalls
      messages.push(entry)
      continue
    }
    messages.push({ role: 'user', content: messageText(message) })
  }
  return messages
}

/** Convert DSH tool schemas into OpenAI function definitions. */
export function toChatTools(tools) {
  if (!Array.isArray(tools) || tools.length === 0) return undefined
  return tools.map((tool) => ({
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters ?? { type: 'object', properties: {} },
    },
  }))
}

/**
 * Incrementally parse an SSE byte stream into `data:` payloads.
 *
 * Chunk boundaries fall anywhere, including mid-JSON, so the parser buffers until
 * it sees a blank line and tolerates the `[DONE]` sentinel and comment
 * keep-alives that some servers emit.
 */
export async function* sseEvents(body, signal) {
  const decoder = new TextDecoder()
  let buffer = ''
  for await (const chunk of body) {
    if (signal?.aborted) throw signal.reason ?? new Error('aborted')
    buffer += typeof chunk === 'string' ? chunk : decoder.decode(chunk, { stream: true })
    let boundary
    while ((boundary = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      for (const line of frame.split(/\r?\n/)) {
        if (!line.startsWith('data:')) continue
        const payload = line.slice(5).trim()
        if (!payload || payload === '[DONE]') continue
        try {
          yield JSON.parse(payload)
        } catch {
          // A malformed frame is a server-side defect. Skipping it keeps the rest
          // of the stream usable, and a truncated stream still ends in a
          // terminal failure chunk from the caller.
        }
      }
    }
  }
}

/** Repair the malformations local models produce most often in tool arguments. */
export function coerceToolArguments(raw) {
  const text = String(raw ?? '').trim()
  if (!text) return '{}'
  try {
    JSON.parse(text)
    return text
  } catch {
    // Some templates wrap the object in prose or leave a trailing comma.
    const start = text.indexOf('{')
    const end = text.lastIndexOf('}')
    if (start !== -1 && end > start) {
      const candidate = text.slice(start, end + 1).replace(/,\s*([}\]])/g, '$1')
      try {
        JSON.parse(candidate)
        return candidate
      } catch {
        // Fall through to the empty-object fallback.
      }
    }
    return '{}'
  }
}

/** Map an OpenAI `finish_reason` onto a DSH terminal reason. */
export function finishReasonOf(raw) {
  switch (raw) {
    case 'tool_calls':
    case 'function_call':
      return { kind: 'tool-calls' }
    case 'length':
      return { kind: 'max-tokens' }
    default:
      return { kind: 'stop' }
  }
}

function numberOr(value, fallback) {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

/**
 * Translate one OpenAI-compatible SSE stream into DSH stream chunks.
 *
 * Blocks open lazily on their first delta and close in index order at the end,
 * which is what `BlockAssembler` requires: reasoning and text may interleave and
 * each tool call owns its own index.
 */
export async function* translateStream(events, { signal } = {}) {
  /** Blocks in creation order; a block's `index` is its position here. */
  const blocks = []
  /** wire tool-call index -> block, because a call's frames share an index. */
  const toolBlocks = new Map()
  let textBlock
  let reasoningBlock
  let usage
  let finishRaw
  let sawAnyFrame = false

  const open = (content) => {
    const block = { index: blocks.length, content, closed: false }
    blocks.push(block)
    return block
  }

  for await (const frame of events) {
    if (signal?.aborted) throw signal.reason ?? new Error('aborted')
    sawAnyFrame = true

    if (frame?.usage) {
      usage = {
        inputTokens: numberOr(frame.usage.prompt_tokens, 0),
        outputTokens: numberOr(frame.usage.completion_tokens, 0),
        totalTokens: numberOr(frame.usage.total_tokens, undefined),
        cacheReadTokens: numberOr(frame.usage.prompt_tokens_details?.cached_tokens, undefined),
      }
    }

    const choice = frame?.choices?.[0]
    if (!choice) continue
    const delta = choice.delta ?? choice.message ?? {}

    // Reasoning arrives under two keys depending on the server and its
    // `--reasoning-format` setting.
    const reasoning = delta.reasoning_content ?? delta.reasoning
    if (typeof reasoning === 'string' && reasoning.length > 0) {
      if (!reasoningBlock) {
        reasoningBlock = open({ type: 'reasoning', text: '' })
        yield { type: 'block-start', index: reasoningBlock.index, blockType: 'reasoning' }
      }
      reasoningBlock.content.text += reasoning
      yield { type: 'reasoning-delta', index: reasoningBlock.index, text: reasoning }
    }

    const text = delta.content
    if (typeof text === 'string' && text.length > 0) {
      if (!textBlock) {
        textBlock = open({ type: 'text', text: '' })
        yield { type: 'block-start', index: textBlock.index, blockType: 'text' }
      }
      textBlock.content.text += text
      yield { type: 'text-delta', index: textBlock.index, text }
    }

    for (const call of delta.tool_calls ?? []) {
      const wireIndex = numberOr(call.index, 0)
      let block = toolBlocks.get(wireIndex)
      if (!block) {
        block = open({
          type: 'tool-call',
          id: String(call.id ?? `call_${wireIndex}`),
          name: call.function?.name ?? '',
          arguments: '',
        })
        toolBlocks.set(wireIndex, block)
        yield { type: 'block-start', index: block.index, blockType: 'tool-call' }
        yield { type: 'tool-call-delta', index: block.index, id: block.content.id, name: block.content.name, argumentsDelta: '' }
      }
      // A continuation frame carries only `arguments`, and occasionally only the
      // name, so both are merged rather than overwritten.
      if (call.function?.name && !block.content.name) block.content.name = call.function.name
      const argumentsDelta = call.function?.arguments
      if (typeof argumentsDelta === 'string' && argumentsDelta.length > 0) {
        block.content.arguments += argumentsDelta
        yield { type: 'tool-call-delta', index: block.index, id: block.content.id, argumentsDelta }
      }
    }

    if (choice.finish_reason) finishRaw = choice.finish_reason
  }

  if (!sawAnyFrame) throw new WireError('local model server returned an empty stream', 'EMPTY_RESPONSE')

  // Close every open block in index order before settling.
  for (const block of blocks) {
    if (block.closed) continue
    block.closed = true
    if (block.content.type === 'tool-call') {
      block.content.arguments = coerceToolArguments(block.content.arguments)
      if (!block.content.name) block.content.name = 'unknown_tool'
    }
    yield { type: 'block-end', index: block.index, block: { ...block.content } }
  }

  const reason = finishReasonOf(finishRaw)
  // A tool-call finish with no tool call captured would leave the agent with
  // nothing to run; report a normal stop instead of a stuck turn.
  if (reason.kind === 'tool-calls' && toolBlocks.size === 0) reason.kind = 'stop'
  if (blocks.length === 0 && reason.kind === 'stop') {
    throw new WireError('local model returned no content', 'EMPTY_RESPONSE')
  }

  const finalUsage = usage ?? { inputTokens: 0, outputTokens: 0 }
  finalUsage.totalTokens = finalUsage.totalTokens ?? finalUsage.inputTokens + finalUsage.outputTokens
  yield { type: 'usage', usage: finalUsage }
  yield { type: 'finish', reason }
}
