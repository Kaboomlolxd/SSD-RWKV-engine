"""Small built-in browser client for the local inference server."""

from __future__ import annotations

CHAT_HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RWKV SSD</title>
  <style>
    :root { color-scheme: light dark; font: 16px/1.5 system-ui, sans-serif; }
    body { margin: 0; background: Canvas; color: CanvasText; }
    main { width: min(900px, calc(100% - 28px)); min-height: 100vh; margin: auto; display: flex; flex-direction: column; }
    header { padding: 22px 4px 10px; border-bottom: 1px solid color-mix(in srgb, CanvasText 18%, transparent); }
    h1 { margin: 0; font-size: 1.35rem; }
    #status { color: color-mix(in srgb, CanvasText 66%, transparent); font-size: .9rem; }
    #messages { flex: 1; padding: 18px 4px; }
    .message { white-space: pre-wrap; overflow-wrap: anywhere; padding: 14px 16px; margin: 12px 0; border-radius: 12px; background: color-mix(in srgb, CanvasText 7%, Canvas); }
    .user { background: color-mix(in srgb, #4589ee 18%, Canvas); }
    .error { color: #d33; }
    form { position: sticky; bottom: 0; padding: 14px 0 20px; background: Canvas; }
    textarea { width: 100%; min-height: 92px; box-sizing: border-box; resize: vertical; border-radius: 10px; padding: 12px; font: inherit; }
    .controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-top: 9px; }
    button { border: 0; border-radius: 8px; padding: 9px 16px; background: #286bd6; color: white; font: inherit; cursor: pointer; }
    button:disabled { opacity: .55; cursor: default; }
    input { max-width: 110px; padding: 6px; }
    details { margin-left: auto; }
    #apiKey { max-width: 190px; }
  </style>
</head>
<body>
<main>
  <header><h1>RWKV SSD</h1><div id="status">Checking local model...</div></header>
  <section id="messages" aria-live="polite"></section>
  <form id="form">
    <textarea id="prompt" placeholder="Write a message. Enter sends, Shift+Enter adds a line." aria-label="Message" required></textarea>
    <div class="controls">
      <label>Tokens <input id="tokens" type="number" min="1" max="8192" value="128"></label>
      <label>Temperature <input id="temperature" type="number" min="0" max="2" step="0.1" value="0.8"></label>
      <button id="send" type="submit">Send</button>
      <button id="cancel" type="button" disabled>Stop</button>
      <button id="clear" type="button">New chat</button>
      <details><summary>Connection</summary><label>API key <input id="apiKey" type="password" autocomplete="off"></label></details>
    </div>
  </form>
</main>
<script>
const messagesEl = document.querySelector('#messages');
const form = document.querySelector('#form');
const input = document.querySelector('#prompt');
const statusEl = document.querySelector('#status');
const sendButton = document.querySelector('#send');
const cancelButton = document.querySelector('#cancel');
const clearButton = document.querySelector('#clear');
const history = [];
let activeController = null;
const keyInput = document.querySelector('#apiKey');
keyInput.value = sessionStorage.getItem('rwkv-ssd-api-key') || '';
keyInput.addEventListener('input', () => sessionStorage.setItem('rwkv-ssd-api-key', keyInput.value));

function addMessage(role, text) {
  const node = document.createElement('article');
  node.className = `message ${role}`;
  node.textContent = text;
  messagesEl.append(node);
  node.scrollIntoView({block: 'end'});
  return node;
}
function authHeaders() {
  const headers = {'Content-Type': 'application/json'};
  if (keyInput.value) headers['X-API-Key'] = keyInput.value;
  return headers;
}
async function checkHealth() {
  try {
    const response = await fetch('/health');
    const data = await response.json();
    statusEl.textContent = response.ok ? `Local engine ${data.status || 'ready'} · ${data.backend || 'CPU'}` : 'Local engine is not ready';
  } catch (_) { statusEl.textContent = 'Cannot reach the local engine'; }
}
form.addEventListener('submit', async event => {
  event.preventDefault();
  const content = input.value.trim();
  if (!content || activeController) return;
  input.value = '';
  history.push({role: 'user', content});
  addMessage('user', content);
  const answer = addMessage('assistant', '');
  activeController = new AbortController();
  sendButton.disabled = true;
  cancelButton.disabled = false;
  clearButton.disabled = true;
  statusEl.textContent = 'Generating...';
  try {
    const response = await fetch('/v1/chat/completions', {
      method: 'POST', headers: authHeaders(), signal: activeController.signal,
      body: JSON.stringify({messages: history, max_tokens: Number(document.querySelector('#tokens').value), temperature: Number(document.querySelector('#temperature').value), stream: true})
    });
    if (!response.ok) {
      let detail = `Request failed (${response.status})`;
      try { detail = (await response.json()).error || detail; } catch (_) {}
      throw new Error(detail);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pending = '';
    let output = '';
    let completed = false;
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      pending += decoder.decode(value, {stream: true});
      const lines = pending.split('\n');
      pending = lines.pop() || '';
      for (const line of lines) {
        if (line === 'data: [DONE]') { completed = true; continue; }
        if (!line.startsWith('data: ')) continue;
        try {
          const item = JSON.parse(line.slice(6));
          output += item.choices?.[0]?.delta?.content || '';
          answer.textContent = output;
          answer.scrollIntoView({block: 'end'});
        } catch (_) {}
      }
    }
    if (!completed) throw new Error('The connection closed before generation finished');
    history.push({role: 'assistant', content: output});
    statusEl.textContent = 'Ready';
  } catch (error) {
    if (error.name === 'AbortError') {
      statusEl.textContent = 'Generation stopped';
      if (!answer.textContent) answer.remove();
    } else {
      answer.classList.add('error');
      answer.textContent = error.message || 'The request failed';
      history.pop();
      statusEl.textContent = 'Request failed';
    }
  } finally {
    activeController = null;
    sendButton.disabled = false;
    cancelButton.disabled = true;
    clearButton.disabled = false;
    input.focus();
  }
});
cancelButton.addEventListener('click', () => activeController?.abort());
clearButton.addEventListener('click', () => { history.length = 0; messagesEl.replaceChildren(); statusEl.textContent = 'Ready'; input.focus(); });
input.addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); } });
checkHealth();
</script>
</body>
</html>'''
