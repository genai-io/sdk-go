// Capture with the official @agentclientprotocol/sdk 2.0.2 client (ClientApp).
// Upstream tag commit: f2f218d314ee8f42c37c466b55d2b917ddc0006c.
// Target schema: schema/v1/schema.json at e2d83349bc9e28a4059ea3acefa1c91c966f74e2.
// Caveat: SDK bundled schema equality with target schema is NOT asserted.
// Run outside the repository (no repo node_modules):
//   tmp=$(mktemp -d); cp capture.mjs package.json package-lock.json "$tmp/"
//   (cd "$tmp" && npm ci --ignore-scripts && node capture.mjs > baseline.jsonl)
// Each frame record retains the exact serialized JSON line (without LF),
// not a reconstructed message. Only the mock agent frames are authored here.
import * as acp from '@agentclientprotocol/sdk';
import assert from 'node:assert/strict';

const records = [{ kind: 'metadata', sdk: '@agentclientprotocol/sdk', version: '2.0.2',
  commit: 'f2f218d314ee8f42c37c466b55d2b917ddc0006c',
  targetSchemaCommit: 'e2d83349bc9e28a4059ea3acefa1c91c966f74e2',
  schemaCaveat: 'SDK bundled schema equality with target schema is not asserted',
  transport: 'in-memory NDJSON', scenarios: ['selected', 'cancelled'] }];
let inbound;
let pendingPrompt;
let turn = 0;
let buffer = '';
const input = new ReadableStream({ start(controller) { inbound = controller; } });
const encoder = new TextEncoder();
const decoder = new TextDecoder();
function send(message) {
  const line = JSON.stringify(message);
  records.push({ direction: 'agent_to_client', line });
  inbound.enqueue(encoder.encode(line + '\n'));
}
function result(id, value) { send({ jsonrpc: '2.0', id, result: value }); }
function receive(message) {
  if (message.method === 'initialize') {
    assert.equal(message.params.protocolVersion, 1);
    result(message.id, { protocolVersion: 1, agentCapabilities: {}, authMethods: [],
      agentInfo: { name: 'deterministic-raw-peer', version: '1' } });
  } else if (message.method === 'session/new') {
    assert.deepEqual(message.params.mcpServers, []);
    result(message.id, { sessionId: 'fixture-session' });
  } else if (message.method === 'session/prompt') {
    pendingPrompt = message.id;
    turn++;
    send({ jsonrpc: '2.0', id: `permission-${turn}`, method: 'session/request_permission',
      params: { sessionId: 'fixture-session', toolCall: { toolCallId: `tool-${turn}`, title: 'Fixture operation' },
        options: [{ optionId: 'allow-once', name: 'Allow once', kind: 'allow_once' }] } });
  } else if (message.method === 'session/cancel') {
    assert.equal(turn, 2);
  } else if (message.id === 'permission-1') {
    assert.deepEqual(message.result, { outcome: { outcome: 'selected', optionId: 'allow-once' } });
    send({ jsonrpc: '2.0', method: 'session/update', params: { sessionId: 'fixture-session',
      update: { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'Approved fixture response.' } } } });
    result(pendingPrompt, { stopReason: 'end_turn' });
  } else if (message.id === 'permission-2') {
    assert.deepEqual(message.result, { outcome: { outcome: 'cancelled' } });
    result(pendingPrompt, { stopReason: 'cancelled' });
  } else {
    throw new Error(`Unexpected client frame: ${JSON.stringify(message)}`);
  }
}
const output = new WritableStream({ write(bytes) {
  buffer += decoder.decode(bytes, { stream: true });
  for (;;) {
    const end = buffer.indexOf('\n');
    if (end < 0) break;
    const line = buffer.slice(0, end);
    buffer = buffer.slice(end + 1);
    records.push({ direction: 'client_to_agent', line });
    receive(JSON.parse(line));
  }
} });
let activeClient;
let permissionCount = 0;
let updateCount = 0;
const app = acp.client({ name: 'fixture-client' })
  .onRequest('session/request_permission', async (ctx) => {
    permissionCount++;
    if (permissionCount === 2) {
      await activeClient.notify('session/cancel', { sessionId: 'fixture-session' });
      return { outcome: { outcome: 'cancelled' } };
    }
    return { outcome: { outcome: 'selected', optionId: 'allow-once' } };
  })
  .onNotification('session/update', (ctx) => {
    assert.equal(ctx.params.update.content.text, 'Approved fixture response.');
    updateCount++;
  });
await app.connectWith(acp.ndJsonStream(output, input), async (ctx) => {
  activeClient = ctx;
  await ctx.request('initialize', { protocolVersion: 1, clientCapabilities: {},
    clientInfo: { name: 'fixture-client', version: '1' } });
  const session = await ctx.request('session/new', { cwd: '/fixture', mcpServers: [] });
  for (const [text, stopReason] of [['Approve fixture operation.', 'end_turn'], ['Cancel fixture operation.', 'cancelled']]) {
    const response = await ctx.request('session/prompt', { sessionId: session.sessionId,
      prompt: [{ type: 'text', text }] });
    assert.equal(response.stopReason, stopReason);
  }
});
assert.equal(permissionCount, 2);
assert.equal(updateCount, 1);
assert.equal(buffer, '');
for (const record of records) console.log(JSON.stringify(record));
