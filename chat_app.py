"""
Omni-Stack Chat App
A lightweight FastAPI chat web app with multi-file/image upload
and API-key permission gate. Proxies to the gateway on :8000.

Env vars:
  GATEWAY_URL      Upstream gateway base URL          (default: http://127.0.0.1:8000)
  CHAT_API_KEY     Required access key for the UI     (default: auto-generated, printed to log)
  CHAT_DEFAULT_MODEL  Default model shown in UI       (default: qwen3-coder)
  CHAT_PORT        Port to listen on                  (default: 8888)
"""
import base64
import json
import mimetypes
import os
import secrets
import sys
import uuid

import httpx
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

# ── Config ──────────────────────────────────────────────────────────────────
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://127.0.0.1:8000")
CHAT_API_KEY = os.getenv("CHAT_API_KEY", "")
CHAT_DEFAULT_MODEL = os.getenv("CHAT_DEFAULT_MODEL", "qwen3-coder")
CHAT_PORT = int(os.getenv("CHAT_PORT", "8888"))

# Generate a key if none was provided so the app is never wide-open.
if not CHAT_API_KEY:
    CHAT_API_KEY = secrets.token_urlsafe(24)
    print(f"[chat_app] No CHAT_API_KEY set — generated key: {CHAT_API_KEY}", flush=True)

# ── Max upload size: 50 MB per file ─────────────────────────────────────────
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
# Max bytes of decoded file text injected per attachment (avoids context overflow)
MAX_FILE_TEXT_BYTES = 8192
# Max upstream error body length captured for error messages
MAX_ERROR_BODY_BYTES = 500

app = FastAPI(title="Omni-Stack Chat", version="1.0.0")

# ── HTML page ────────────────────────────────────────────────────────────────
_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Omni-Stack Chat</title>
<style>
  :root{--bg:#0f1117;--surface:#1a1d27;--border:#2e3047;--accent:#7c6af7;
        --accent-hover:#9d8fff;--text:#e8e8f0;--sub:#8b8da8;--danger:#e05c5c;
        --green:#4caf80;--radius:12px;--font:'Segoe UI',system-ui,sans-serif}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:var(--font);
       display:flex;flex-direction:column;height:100vh;overflow:hidden}

  /* ── Gate overlay ── */
  #gate{position:fixed;inset:0;background:rgba(0,0,0,.7);backdrop-filter:blur(6px);
        display:flex;align-items:center;justify-content:center;z-index:999}
  #gate-box{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
            padding:2rem;width:min(400px,90vw);text-align:center}
  #gate-box h2{margin-bottom:.5rem}
  #gate-box p{color:var(--sub);font-size:.9rem;margin-bottom:1.5rem}
  #gate-key{width:100%;padding:.7rem 1rem;border:1px solid var(--border);border-radius:8px;
            background:var(--bg);color:var(--text);font-size:1rem;outline:none;
            transition:border-color .2s}
  #gate-key:focus{border-color:var(--accent)}
  #gate-btn{margin-top:.8rem;width:100%;padding:.75rem;border:none;border-radius:8px;
            background:var(--accent);color:#fff;font-size:1rem;cursor:pointer;
            transition:background .2s}
  #gate-btn:hover{background:var(--accent-hover)}
  #gate-err{color:var(--danger);font-size:.85rem;margin-top:.5rem;min-height:1.2em}

  /* ── Header ── */
  header{background:var(--surface);border-bottom:1px solid var(--border);
         padding:.75rem 1.25rem;display:flex;align-items:center;gap:1rem;flex-shrink:0}
  header h1{font-size:1.1rem;font-weight:600;flex:1}
  #model-select{background:var(--bg);color:var(--text);border:1px solid var(--border);
                border-radius:8px;padding:.4rem .75rem;font-size:.85rem;outline:none;cursor:pointer}
  #status-dot{width:10px;height:10px;border-radius:50%;background:var(--sub);flex-shrink:0}
  #status-dot.ok{background:var(--green)}
  #status-dot.warn{background:#f0b429}

  /* ── Messages ── */
  #messages{flex:1;overflow-y:auto;padding:1rem;display:flex;flex-direction:column;gap:.75rem;
            scroll-behavior:smooth}
  .msg{max-width:78%;display:flex;flex-direction:column;gap:.3rem;animation:fadeIn .2s ease}
  @keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
  .msg.user{align-self:flex-end}
  .msg.assistant{align-self:flex-start}
  .bubble{padding:.7rem 1rem;border-radius:var(--radius);line-height:1.6;
          white-space:pre-wrap;word-break:break-word;font-size:.92rem}
  .msg.user .bubble{background:var(--accent);color:#fff;border-bottom-right-radius:4px}
  .msg.assistant .bubble{background:var(--surface);border:1px solid var(--border);
                          border-bottom-left-radius:4px}
  .msg.error .bubble{background:#2e1a1a;border:1px solid var(--danger);color:var(--danger)}
  .meta{font-size:.72rem;color:var(--sub);padding:0 .25rem}
  .msg.user .meta{text-align:right}

  /* Code blocks */
  .bubble pre{background:#0d0f18;border:1px solid var(--border);border-radius:8px;
              padding:.75rem;overflow-x:auto;margin:.4rem 0;font-size:.83rem;font-family:monospace}
  .bubble code{font-family:monospace;font-size:.85em;background:#0d0f18;
               padding:.1em .35em;border-radius:4px}
  .bubble pre code{background:none;padding:0;font-size:inherit}

  /* Inline images */
  .bubble img{max-width:100%;max-height:240px;border-radius:8px;margin:.3rem 0;
              display:block;cursor:pointer}

  /* File attachments */
  .attachments{display:flex;flex-wrap:wrap;gap:.4rem;padding:0 .1rem}
  .att-chip{background:var(--surface);border:1px solid var(--border);border-radius:6px;
            padding:.25rem .6rem;font-size:.78rem;display:flex;align-items:center;gap:.35rem;
            color:var(--sub)}
  .att-chip .ico{font-size:.9rem}

  /* Typing indicator */
  #typing{display:none;align-self:flex-start;background:var(--surface);
          border:1px solid var(--border);border-radius:var(--radius);
          border-bottom-left-radius:4px;padding:.6rem 1rem;font-size:.85rem;color:var(--sub)}
  #typing.show{display:block}
  .dot-anim::after{content:'...';animation:dots 1.2s steps(4,end) infinite}
  @keyframes dots{0%,20%{content:''}40%{content:'.'}60%{content:'..'}80%,100%{content:'...'}}

  /* ── Input area ── */
  footer{background:var(--surface);border-top:1px solid var(--border);padding:.75rem 1rem;
         flex-shrink:0}
  #preview-strip{display:flex;flex-wrap:wrap;gap:.4rem;margin-bottom:.5rem}
  .prev-thumb{position:relative;border-radius:8px;overflow:hidden;
              background:var(--bg);border:1px solid var(--border)}
  .prev-thumb img{width:64px;height:64px;object-fit:cover;display:block}
  .prev-thumb .file-label{padding:.4rem .5rem;font-size:.72rem;color:var(--sub);
                           max-width:100px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .prev-thumb button{position:absolute;top:2px;right:2px;background:rgba(0,0,0,.6);
                     border:none;color:#fff;border-radius:50%;width:18px;height:18px;
                     font-size:.7rem;cursor:pointer;line-height:18px;text-align:center;padding:0}
  #input-row{display:flex;gap:.5rem;align-items:flex-end}
  #prompt{flex:1;padding:.65rem .9rem;border:1px solid var(--border);border-radius:10px;
          background:var(--bg);color:var(--text);font-size:.92rem;resize:none;
          min-height:44px;max-height:180px;outline:none;line-height:1.5;transition:border-color .2s;
          font-family:var(--font)}
  #prompt:focus{border-color:var(--accent)}
  .icon-btn{background:none;border:1px solid var(--border);border-radius:10px;
            padding:.55rem .75rem;color:var(--sub);cursor:pointer;font-size:1.1rem;
            transition:color .2s,border-color .2s;line-height:1;flex-shrink:0}
  .icon-btn:hover{color:var(--text);border-color:var(--accent)}
  #send-btn{background:var(--accent);border-color:var(--accent);color:#fff;
            border-radius:10px;padding:.55rem .9rem;font-size:1rem;cursor:pointer;
            transition:background .2s;flex-shrink:0}
  #send-btn:hover:not(:disabled){background:var(--accent-hover)}
  #send-btn:disabled{opacity:.5;cursor:not-allowed}
  #file-input{display:none}
  #hint{text-align:center;font-size:.72rem;color:var(--sub);margin-top:.4rem}
</style>
</head>
<body>

<!-- Permission gate -->
<div id="gate">
  <div id="gate-box">
    <h2>🔑 Access Required</h2>
    <p>Enter your API key to access Omni-Stack Chat</p>
    <input id="gate-key" type="password" placeholder="Paste your API key…" autocomplete="off"/>
    <button id="gate-btn">Unlock</button>
    <div id="gate-err"></div>
  </div>
</div>

<header>
  <h1>⚡ Omni-Stack Chat</h1>
  <select id="model-select">
    <option value="qwen3-coder">Qwen3-Coder 480B</option>
    <option value="qwen3-80b">Qwen3-80B</option>
    <option value="vision">Vision (VL-32B)</option>
    <option value="lilith">Lilith-70B</option>
  </select>
  <div id="status-dot" title="Gateway status"></div>
</header>

<div id="messages">
  <div class="msg assistant">
    <div class="bubble">👋 Hello! I'm <strong>Omni-Stack Chat</strong>.
Upload files or images with the 📎 button, then ask me anything.</div>
  </div>
</div>
<div id="typing"><span class="dot-anim">Thinking</span></div>

<footer>
  <div id="preview-strip"></div>
  <div id="input-row">
    <button class="icon-btn" id="upload-btn" title="Attach files/images">📎</button>
    <input type="file" id="file-input" multiple accept="image/*,text/*,.pdf,.csv,.json,.md,.py,.js,.ts,.sh,.yaml,.yml,.toml,.txt,.log,.rs,.go,.cpp,.c,.h,.java,.rb,.php"/>
    <textarea id="prompt" rows="1" placeholder="Type a message… (Shift+Enter for new line)"></textarea>
    <button id="send-btn" title="Send (Enter)">➤</button>
  </div>
  <div id="hint">Shift+Enter = new line · Enter = send · 📎 accepts images &amp; text files</div>
</footer>

<script>
"use strict";
// ── State ──────────────────────────────────────────────────────────────────
const API_BASE = '';
let apiKey = sessionStorage.getItem('omni_api_key') || '';
let pendingFiles = [];   // [{name, type, b64, dataUrl, isImage}]
let history = [];        // [{role, content}]
const DEFAULT_MODEL = '__DEFAULT__';

// ── Gate logic ─────────────────────────────────────────────────────────────
async function verifyKey(key) {
  const r = await fetch(`${API_BASE}/api/health`, {
    headers: { 'x-api-key': key }
  });
  return r.ok;
}

async function tryUnlock() {
  const key = document.getElementById('gate-key').value.trim();
  document.getElementById('gate-err').textContent = '';
  if (!key) { document.getElementById('gate-err').textContent = 'Please enter a key.'; return; }
  const ok = await verifyKey(key);
  if (ok) {
    apiKey = key;
    sessionStorage.setItem('omni_api_key', key);
    document.getElementById('gate').style.display = 'none';
    loadModels();
    pollHealth();
  } else {
    document.getElementById('gate-err').textContent = 'Invalid key — try again.';
  }
}

document.getElementById('gate-btn').addEventListener('click', tryUnlock);
document.getElementById('gate-key').addEventListener('keydown', e => {
  if (e.key === 'Enter') tryUnlock();
});

// Auto-check stored key on load
(async () => {
  if (apiKey && await verifyKey(apiKey)) {
    document.getElementById('gate').style.display = 'none';
    loadModels();
    pollHealth();
  }
})();

// ── Model list ─────────────────────────────────────────────────────────────
async function loadModels() {
  try {
    const r = await fetch(`${API_BASE}/api/models`, { headers: { 'x-api-key': apiKey } });
    if (!r.ok) return;
    const data = await r.json();
    const sel = document.getElementById('model-select');
    sel.innerHTML = '';
    (data.models || []).forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = m.label || m.id;
      if (m.default) opt.selected = true;
      sel.appendChild(opt);
    });
  } catch(_) {}
}

// ── Health poll ────────────────────────────────────────────────────────────
async function pollHealth() {
  try {
    const r = await fetch(`${API_BASE}/api/health`, { headers: { 'x-api-key': apiKey } });
    const dot = document.getElementById('status-dot');
    if (r.ok) {
      const d = await r.json();
      dot.className = d.status === 'ok' ? 'ok' : 'warn';
      dot.title = 'Gateway: ' + d.status;
    } else { dot.className = ''; dot.title = 'Gateway error'; }
  } catch(_) { document.getElementById('status-dot').className = ''; }
  setTimeout(pollHealth, 30000);
}

// ── File upload ────────────────────────────────────────────────────────────
document.getElementById('upload-btn').addEventListener('click', () =>
  document.getElementById('file-input').click());

document.getElementById('file-input').addEventListener('change', e => {
  Array.from(e.target.files).forEach(addFile);
  e.target.value = '';
});

// Drag-and-drop on messages area
document.getElementById('messages').addEventListener('dragover', e => e.preventDefault());
document.getElementById('messages').addEventListener('drop', e => {
  e.preventDefault();
  Array.from(e.dataTransfer.files).forEach(addFile);
});

function addFile(file) {
  const reader = new FileReader();
  reader.onload = ev => {
    const dataUrl = ev.target.result;
    const b64 = dataUrl.split(',')[1];
    const isImage = file.type.startsWith('image/');
    const entry = { name: file.name, type: file.type || 'application/octet-stream', b64, dataUrl, isImage };
    pendingFiles.push(entry);
    renderPreviews();
  };
  reader.readAsDataURL(file);
}

function renderPreviews() {
  const strip = document.getElementById('preview-strip');
  strip.innerHTML = '';
  pendingFiles.forEach((f, i) => {
    const div = document.createElement('div');
    div.className = 'prev-thumb';
    if (f.isImage) {
      div.innerHTML = `<img src="${f.dataUrl}" title="${esc(f.name)}"/>`;
    } else {
      div.innerHTML = `<div class="file-label">📄 ${esc(f.name)}</div>`;
    }
    const btn = document.createElement('button');
    btn.textContent = '✕';
    btn.title = 'Remove';
    btn.onclick = () => { pendingFiles.splice(i,1); renderPreviews(); };
    div.appendChild(btn);
    strip.appendChild(div);
  });
}

// ── Chat submission ────────────────────────────────────────────────────────
const promptEl = document.getElementById('prompt');
const sendBtn  = document.getElementById('send-btn');

promptEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
// Auto-grow textarea
promptEl.addEventListener('input', () => {
  promptEl.style.height = 'auto';
  promptEl.style.height = Math.min(promptEl.scrollHeight, 180) + 'px';
});
sendBtn.addEventListener('click', sendMessage);

async function sendMessage() {
  const text = promptEl.value.trim();
  if (!text && pendingFiles.length === 0) return;
  promptEl.value = '';
  promptEl.style.height = 'auto';
  sendBtn.disabled = true;

  const files = [...pendingFiles];
  pendingFiles = [];
  renderPreviews();

  // Build display message
  const model = document.getElementById('model-select').value;
  appendUserMessage(text, files);

  // Build API payload
  const attachments = files.map(f => ({
    name: f.name, type: f.type, b64: f.b64, is_image: f.isImage
  }));

  showTyping(true);
  try {
    const r = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-api-key': apiKey },
      body: JSON.stringify({ message: text, model, attachments, history })
    });
    if (!r.ok) {
      const e = await r.json().catch(() => ({detail:'Unknown error'}));
      appendError(e.detail || JSON.stringify(e));
      return;
    }
    const data = await r.json();
    const reply = data.reply || '';
    history.push({ role: 'user', content: text });
    history.push({ role: 'assistant', content: reply });
    if (history.length > 40) history = history.slice(-40);
    appendAssistantMessage(reply);
  } catch(err) {
    appendError('Network error: ' + err.message);
  } finally {
    showTyping(false);
    sendBtn.disabled = false;
    promptEl.focus();
  }
}

// ── Rendering helpers ──────────────────────────────────────────────────────
const msgList = document.getElementById('messages');

function appendUserMessage(text, files) {
  const div = document.createElement('div');
  div.className = 'msg user';
  let html = '';
  if (files.length) {
    html += '<div class="attachments">';
    files.forEach(f => {
      if (f.isImage) {
        html += `<div class="att-chip"><span class="ico">🖼</span>${esc(f.name)}</div>`;
      } else {
        html += `<div class="att-chip"><span class="ico">📄</span>${esc(f.name)}</div>`;
      }
    });
    html += '</div>';
  }
  if (text) html += `<div class="bubble">${esc(text)}</div>`;
  div.innerHTML = html;
  msgList.appendChild(div);
  scrollBottom();
}

function appendAssistantMessage(text) {
  const div = document.createElement('div');
  div.className = 'msg assistant';
  div.innerHTML = `<div class="bubble">${renderMarkdown(text)}</div>
    <div class="meta">${timeNow()}</div>`;
  msgList.appendChild(div);
  scrollBottom();
}

function appendError(msg) {
  const div = document.createElement('div');
  div.className = 'msg error';
  div.innerHTML = `<div class="bubble">⚠ ${esc(msg)}</div>`;
  msgList.appendChild(div);
  scrollBottom();
}

function showTyping(v) {
  document.getElementById('typing').className = v ? 'show' : '';
  if (v) scrollBottom();
}

function scrollBottom() {
  setTimeout(() => msgList.scrollTop = msgList.scrollHeight, 50);
}

function timeNow() {
  return new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

// Basic markdown: code fences, inline code, bold, italic
function renderMarkdown(text) {
  // Code fences
  text = text.replace(/```(\w*)\n?([\s\S]*?)```/g, (_,lang,code) =>
    `<pre><code class="lang-${esc(lang)}">${esc(code.trim())}</code></pre>`);
  // Inline code
  text = text.replace(/`([^`]+)`/g, (_,c) => `<code>${esc(c)}</code>`);
  // Bold
  text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Italic
  text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // Image data URLs from response (safety: only data: urls)
  text = text.replace(/!\[([^\]]*)\]\((data:image[^)]+)\)/g,
    (_,alt,url) => `<img src="${url}" alt="${esc(alt)}" onclick="window.open(this.src)"/>`);
  // Newlines
  text = text.replace(/\n/g, '<br/>');
  return text;
}
</script>
</body>
</html>
"""


# ── Auth dependency ──────────────────────────────────────────────────────────
def _check_key(request: Request):
    key = request.headers.get("x-api-key", "")
    if key != CHAT_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(_HTML)


@app.get("/api/health")
async def api_health(request: Request):
    _check_key(request)
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{GATEWAY_URL}/health")
            data = r.json()
    except Exception as exc:
        data = {"status": "down", "error": str(exc)}
    return JSONResponse(data)


@app.get("/api/models")
async def api_models(request: Request):
    _check_key(request)
    default_model = CHAT_DEFAULT_MODEL
    models = [
        {"id": "qwen3-coder", "label": "Qwen3-Coder 480B (GGUF)",
         "default": default_model == "qwen3-coder"},
        {"id": "qwen3-80b",  "label": "Qwen3-80B (vLLM)",
         "default": default_model == "qwen3-80b"},
        {"id": "vision",     "label": "Vision — Qwen2.5-VL-32B",
         "default": default_model == "vision"},
        {"id": "lilith",     "label": "Lilith-Whisper-70B (Ollama)",
         "default": default_model == "lilith"},
    ]
    return JSONResponse({"models": models})


@app.post("/api/chat")
async def api_chat(request: Request):
    """
    Accept JSON:
      message      str — user's text message
      model        str — model alias
      history      list[{role,content}] — recent turns for context
      attachments  list[{name, type, b64, is_image}] — uploaded files
    Returns JSON: {reply: str}
    """
    _check_key(request)
    body = await request.json()
    message: str = body.get("message", "").strip()
    model: str = body.get("model", CHAT_DEFAULT_MODEL)
    history: list = body.get("history", [])[-20:]  # cap at last 20 turns
    attachments: list = body.get("attachments", [])

    if not message and not attachments:
        raise HTTPException(status_code=400, detail="message or attachments required")

    # ── Build the content block for the user turn ────────────────────────
    # For vision model: use multi-modal content list
    # For text models: inject file contents as text context
    use_vision = model in ("vision", "qwen2.5-vl", "vl")
    images = [a for a in attachments if a.get("is_image")]
    docs = [a for a in attachments if not a.get("is_image")]

    if use_vision and images:
        content = []
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{img['type']};base64,{img['b64']}"
                }
            })
        if message:
            content.append({"type": "text", "text": message})
    else:
        # Text-only: embed image as base64 hint + doc contents
        parts = []
        for img in images:
            parts.append(f"[Image attached: {img['name']}]")
            # For non-vision models still pass base64 so coder model can reference
            parts.append(
                f"<image name=\"{img['name']}\" type=\"{img['type']}\" "
                f"base64=\"{img['b64'][:64]}…(truncated)\">"
            )
        for doc in docs:
            try:
                raw = base64.b64decode(doc["b64"]).decode("utf-8", errors="replace")
                # Truncate to MAX_FILE_TEXT_BYTES per file to avoid context overflow
                if len(raw) > MAX_FILE_TEXT_BYTES:
                    raw = raw[:MAX_FILE_TEXT_BYTES] + "\n… [truncated]"
                parts.append(
                    f"\n<file name=\"{doc['name']}\">\n{raw}\n</file>"
                )
            except Exception:
                parts.append(f"[Binary file attached: {doc['name']} — cannot display]")
        if message:
            parts.append(message)
        content = "\n".join(parts)

    messages = list(history)
    messages.append({"role": "user", "content": content})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": int(os.getenv("CHAT_MAX_TOKENS", "4096")),
        "temperature": float(os.getenv("CHAT_TEMPERATURE", "0.7")),
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=300.0) as c:
            r = await c.post(f"{GATEWAY_URL}/v1/chat/completions", json=payload)
        if r.status_code >= 400:
            detail = r.json() if r.headers.get("content-type", "").startswith("application/json") \
                else r.text[:MAX_ERROR_BODY_BYTES]
            raise HTTPException(status_code=502, detail=str(detail))
        data = r.json()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gateway error: {exc}") from exc

    choices = data.get("choices") or []
    reply = choices[0].get("message", {}).get("content", "") if choices else ""
    return JSONResponse({"reply": reply})


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=CHAT_PORT)
