/**
 * ARIA / Hermes — Unified API Client
 * Handles HTTP requests and WebSocket connections to the FastAPI backend.
 */

const API_BASE = window.location.origin;
const WS_BASE  = API_BASE.replace(/^http/, 'ws');

// ── HTTP Client ────────────────────────────────────────────────
const api = {
  async get(path, opts = {}) {
    const res = await fetch(`${API_BASE}${path}`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json', ...opts.headers },
      ...opts,
    });
    if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
    return res.json();
  },

  async post(path, body, opts = {}) {
    const res = await fetch(`${API_BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...opts.headers },
      body: JSON.stringify(body),
      ...opts,
    });
    if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
    return res.json();
  },

  async patch(path, body, opts = {}) {
    const res = await fetch(`${API_BASE}${path}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json', ...opts.headers },
      body: JSON.stringify(body),
      ...opts,
    });
    if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
    return res.json();
  },

  async delete(path, opts = {}) {
    const res = await fetch(`${API_BASE}${path}`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json', ...opts.headers },
      ...opts,
    });
    if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
    return res.json();
  },
};

// ── WebSocket Chat Client ──────────────────────────────────────
class HermesChat {
  constructor() {
    this.ws = null;
    this.sessionId = localStorage.getItem('hermes_session_id') || null;
    this.onToken = null;       // callback(token: string)
    this.onDone = null;        // callback(metadata: object)
    this.onError = null;       // callback(error: string)
    this.onAck = null;         // callback(session_id: string)
    this._reconnectAttempts = 0;
    this._maxReconnects = 5;
  }

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    this.ws = new WebSocket(`${WS_BASE}/api/chat/ws`);

    this.ws.onopen = () => {
      console.log('[Hermes] WebSocket connected');
      this._reconnectAttempts = 0;
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        switch (data.type) {
          case 'ack':
            this.sessionId = data.session_id;
            localStorage.setItem('hermes_session_id', this.sessionId);
            this.onAck?.(data.session_id);
            break;
          case 'token':
            this.onToken?.(data.content);
            break;
          case 'done':
            this.onDone?.(data.metadata);
            break;
          case 'error':
            this.onError?.(data.content);
            break;
        }
      } catch (e) {
        console.error('[Hermes] Message parse error:', e);
      }
    };

    this.ws.onclose = (event) => {
      console.log('[Hermes] WebSocket disconnected', event.code);
      if (this._reconnectAttempts < this._maxReconnects) {
        const delay = Math.min(1000 * 2 ** this._reconnectAttempts, 10000);
        this._reconnectAttempts++;
        console.log(`[Hermes] Reconnecting in ${delay}ms...`);
        setTimeout(() => this.connect(), delay);
      }
    };

    this.ws.onerror = (err) => {
      console.error('[Hermes] WebSocket error:', err);
      this.onError?.('Connection error. Retrying...');
    };
  }

  send(message) {
    if (this.ws?.readyState !== WebSocket.OPEN) {
      console.warn('[Hermes] WebSocket not ready, reconnecting...');
      this.connect();
      // Queue message after brief delay
      setTimeout(() => this.send(message), 500);
      return;
    }

    this.ws.send(JSON.stringify({
      message,
      session_id: this.sessionId,
    }));
  }

  async sendFeedback(runId, rating, notes = '') {
    return api.post(`/api/chat/${runId}/feedback`, { rating, notes });
  }

  disconnect() {
    this._reconnectAttempts = this._maxReconnects; // prevent auto-reconnect
    this.ws?.close();
  }
}

// ── Audio (Whisper / Voice Input) ──────────────────────────────
class VoiceInput {
  constructor() {
    this.mediaRecorder = null;
    this.chunks = [];
    this.isRecording = false;
  }

  async start() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.chunks = [];
    this.mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    this.mediaRecorder.ondataavailable = (e) => this.chunks.push(e.data);
    this.mediaRecorder.start(100);
    this.isRecording = true;
  }

  async stop() {
    return new Promise((resolve) => {
      this.mediaRecorder.onstop = async () => {
        const blob = new Blob(this.chunks, { type: 'audio/webm' });
        const formData = new FormData();
        formData.append('audio', blob, 'recording.webm');
        this.isRecording = false;
        // Send to backend for Whisper transcription
        const res = await fetch(`${API_BASE}/api/activities/transcribe`, {
          method: 'POST',
          body: formData,
        });
        const data = await res.json();
        resolve(data.transcript || '');
      };
      this.mediaRecorder.stop();
      this.mediaRecorder.stream.getTracks().forEach(t => t.stop());
    });
  }
}

// ── Utilities ──────────────────────────────────────────────────
function formatDate(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleDateString('en-US', {
    month: 'short', day: 'numeric', year: 'numeric'
  });
}

function formatRelative(iso) {
  if (!iso) return '—';
  const diff = Date.now() - new Date(iso).getTime();
  const mins  = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  const days  = Math.floor(diff / 86400000);
  if (mins  < 1)  return 'just now';
  if (mins  < 60) return `${mins}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days  < 7)  return `${days}d ago`;
  return formatDate(iso);
}

function statusBadge(status) {
  const map = {
    active:   'badge-active',
    paused:   'badge-paused',
    done:     'badge-done',
    completed:'badge-done',
    blocked:  'badge-blocked',
    archived: 'badge-archived',
    todo:     'badge-archived',
    in_progress: 'badge-paused',
  };
  return `<span class="badge ${map[status] || 'badge-archived'}">${status?.replace('_', ' ') || 'unknown'}</span>`;
}

function priorityLabel(p) {
  const map = { 1: '🔴 Critical', 2: '🟠 High', 3: '🟡 Medium', 4: '🔵 Low', 5: '⚪ Minimal' };
  return map[p] || '—';
}

// ── Exports ────────────────────────────────────────────────────
window.HermesAPI = api;
window.HermesChat = HermesChat;
window.VoiceInput = VoiceInput;
window.HermesUtils = { formatDate, formatRelative, statusBadge, priorityLabel };
