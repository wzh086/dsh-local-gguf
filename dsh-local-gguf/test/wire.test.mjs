import { translateStream, toChatMessages, toChatTools, coerceToolArguments } from '../lib/openai-wire.js'

let failures = 0
function check(name, actual, expected) {
  const a = JSON.stringify(actual)
  const e = JSON.stringify(expected)
  if (a === e) { console.log(`  ok   ${name}`) } else { failures++; console.log(`  FAIL ${name}\n       got      ${a}\n       expected ${e}`) }
}

/** Feed raw SSE text through the full pipeline, chunked at awkward boundaries. */
async function run(sseText, chunkSize = 7) {
  async function* body() {
    for (let i = 0; i < sseText.length; i += chunkSize) yield sseText.slice(i, i + chunkSize)
  }
  const { sseEvents } = await import('../lib/openai-wire.js')
  const out = []
  for await (const chunk of translateStream(sseEvents(body()))) out.push(chunk)
  return out
}

const frame = (o) => `data: ${JSON.stringify(o)}\n\n`

console.log('=== 1. plain text stream ===')
{
  const sse =
    frame({ choices: [{ delta: { role: 'assistant', content: '' } }] }) +
    frame({ choices: [{ delta: { content: 'Hello' } }] }) +
    frame({ choices: [{ delta: { content: ', world' } }] }) +
    frame({ choices: [{ delta: {}, finish_reason: 'stop' }] }) +
    frame({ choices: [], usage: { prompt_tokens: 12, completion_tokens: 4, total_tokens: 16 } }) +
    'data: [DONE]\n\n'
  const out = await run(sse)
  check('chunks', out, [
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'text-delta', index: 0, text: 'Hello' },
    { type: 'text-delta', index: 0, text: ', world' },
    { type: 'block-end', index: 0, block: { type: 'text', text: 'Hello, world' } },
    { type: 'usage', usage: { inputTokens: 12, outputTokens: 4, totalTokens: 16 } },
    { type: 'finish', reason: { kind: 'stop' } },
  ])
}

console.log('=== 2. reasoning then text (reasoning_content key) ===')
{
  const sse =
    frame({ choices: [{ delta: { reasoning_content: 'Let me think' } }] }) +
    frame({ choices: [{ delta: { reasoning_content: '...' } }] }) +
    frame({ choices: [{ delta: { content: 'Answer' } }] }) +
    frame({ choices: [{ delta: {}, finish_reason: 'stop' }] })
  const out = await run(sse)
  check('indexes and order', out, [
    { type: 'block-start', index: 0, blockType: 'reasoning' },
    { type: 'reasoning-delta', index: 0, text: 'Let me think' },
    { type: 'reasoning-delta', index: 0, text: '...' },
    { type: 'block-start', index: 1, blockType: 'text' },
    { type: 'text-delta', index: 1, text: 'Answer' },
    { type: 'block-end', index: 0, block: { type: 'reasoning', text: 'Let me think...' } },
    { type: 'block-end', index: 1, block: { type: 'text', text: 'Answer' } },
    { type: 'usage', usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 } },
    { type: 'finish', reason: { kind: 'stop' } },
  ])
}

console.log('=== 3. tool call split across deltas, no id on continuation ===')
{
  const sse =
    frame({ choices: [{ delta: { tool_calls: [{ index: 0, id: 'call_abc', type: 'function', function: { name: 'read', arguments: '' } }] } }] }) +
    frame({ choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: '{"file' } }] } }] }) +
    frame({ choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: '_path":"a.txt"}' } }] } }] }) +
    frame({ choices: [{ delta: {}, finish_reason: 'tool_calls' }] })
  const out = await run(sse)
  check('chunks', out, [
    { type: 'block-start', index: 0, blockType: 'tool-call' },
    { type: 'tool-call-delta', index: 0, id: 'call_abc', name: 'read', argumentsDelta: '' },
    { type: 'tool-call-delta', index: 0, id: 'call_abc', argumentsDelta: '{"file' },
    { type: 'tool-call-delta', index: 0, id: 'call_abc', argumentsDelta: '_path":"a.txt"}' },
    { type: 'block-end', index: 0, block: { type: 'tool-call', id: 'call_abc', name: 'read', arguments: '{"file_path":"a.txt"}' } },
    { type: 'usage', usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 } },
    { type: 'finish', reason: { kind: 'tool-calls' } },
  ])
}

console.log('=== 4. two parallel tool calls ===')
{
  const sse =
    frame({ choices: [{ delta: { tool_calls: [
      { index: 0, id: 'c0', function: { name: 'a', arguments: '{}' } },
      { index: 1, id: 'c1', function: { name: 'b', arguments: '{}' } },
    ] } }] }) +
    frame({ choices: [{ delta: {}, finish_reason: 'tool_calls' }] })
  const out = await run(sse)
  const ends = out.filter((c) => c.type === 'block-end').map((c) => c.block.name)
  check('both blocks closed', ends, ['a', 'b'])
  check('finish', out.at(-1), { type: 'finish', reason: { kind: 'tool-calls' } })
}

console.log('=== 5. length finish maps to max-tokens ===')
{
  const sse =
    frame({ choices: [{ delta: { content: 'truncated' } }] }) +
    frame({ choices: [{ delta: {}, finish_reason: 'length' }] })
  const out = await run(sse)
  check('finish', out.at(-1), { type: 'finish', reason: { kind: 'max-tokens' } })
}

console.log('=== 6. empty stream throws EMPTY_RESPONSE ===')
{
  try {
    await run('data: [DONE]\n\n')
    failures++; console.log('  FAIL expected throw')
  } catch (e) { check('code', e.code, 'EMPTY_RESPONSE') }
}

console.log('=== 7. malformed tool JSON is repaired / coerced ===')
{
  check('trailing comma', coerceToolArguments('{"a":1,}'), '{"a":1}')
  check('prose wrapped', coerceToolArguments('Sure! {"a":1} hope that helps'), '{"a":1}')
  check('garbage', coerceToolArguments('not json at all'), '{}')
  check('empty', coerceToolArguments(''), '{}')
}

console.log('=== 8. message translation ===')
{
  const messages = toChatMessages({
    system: 'You are helpful.',
    messages: [
      { role: 'user', content: [{ type: 'text', text: 'hi' }] },
      { role: 'assistant', source: { kind: 'model', provider: 'p', model: 'm' }, content: [
        { type: 'reasoning', text: 'hidden' },
        { type: 'text', text: 'calling' },
        { type: 'tool-call', id: 'c1', name: 'read', arguments: '{"p":1}' },
      ] },
      { role: 'tool', toolCallId: 'c1', source: { kind: 'tool', callId: 'c1' }, content: [{ type: 'text', text: 'file body' }] },
      { role: 'assistant', content: [{ type: 'reasoning', text: 'only thinking' }] },
    ],
    tools: [{ name: 'read', description: 'Read a file', parameters: { type: 'object', properties: {} } }],
  })
  check('messages', messages, [
    { role: 'system', content: 'You are helpful.' },
    { role: 'user', content: 'hi' },
    { role: 'assistant', content: 'calling', tool_calls: [{ id: 'c1', type: 'function', function: { name: 'read', arguments: '{"p":1}' } }] },
    { role: 'tool', tool_call_id: 'c1', content: 'file body' },
  ])
  check('tools', toChatTools([{ name: 'read', description: 'Read a file', parameters: { type: 'object', properties: {} } }]), [
    { type: 'function', function: { name: 'read', description: 'Read a file', parameters: { type: 'object', properties: {} } } },
  ])
}

console.log('')
console.log(failures === 0 ? 'ALL PASS' : `${failures} FAILURE(S)`)
process.exit(failures === 0 ? 0 : 1)
