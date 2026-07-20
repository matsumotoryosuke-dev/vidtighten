// Utility functions — no imports needed
export const $ = id => document.getElementById(id);

export function fmt(s) {
  if (s == null || isNaN(s)) return '—';
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  const ms  = Math.round((s % 1) * 10);
  if (h) return `${h}:${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}`;
  return `${m}:${String(sec).padStart(2,'0')}.${ms}`;
}

export function fmtDur(s) {
  if (s < 0.1) return `${(s*1000).toFixed(0)}ms`;
  return `${s.toFixed(1)}s`;
}

export const plur = (n, w) => `${n} ${w}${n !== 1 ? 's' : ''}`;

export function fmtElapsed(secs) {
  if (secs < 60) return `${Math.floor(secs)}s`;
  return `${Math.floor(secs / 60)}m ${Math.floor(secs % 60)}s`;
}

export function escH(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

export function toast(msg, dur=2400) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), dur);
}

export async function api(url, body) {
  const r = await fetch(url, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  return r.json();
}

export async function apiFetch(url, body) {
  return fetch(url, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
}

export async function dlBlob(blob, name) {
  // Inside pywebview (WKWebView from http://), anchor.click() on a blob: URL is
  // treated as a navigation event — the entire page blanks. We must NEVER reach
  // that path when running inside pywebview, even as a fallback.
  if (window.pywebview) {
    if (!window.pywebview.api?.save_file_content) {
      // Bridge not wired yet — old build. Warn the user; do not crash the page.
      toast('Export requires app restart — please quit and reopen VidTighten.', 5000);
      return { ok: false };
    }
    // Wrap in try-catch: an unhandled Promise rejection from the pywebview bridge
    // can cause the WKWebView to navigate to an error page, blanking the app.
    try {
      const result = await window.pywebview.api.save_file_content(await blob.text(), name);
      if (!result?.ok) {
        toast('Export failed: ' + (result?.error || 'unknown error'), 4000);
        return { ok: false };
      }
      return { ok: true };
    } catch (err) {
      console.error('[dlBlob] bridge error:', err);
      toast('Export error — check ~/Downloads for the file.', 4000);
      return { ok: false };
    }
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 500);
  return { ok: true };
}

export function _fmtBytes(b) {
  if (b < 1024)       return b + ' B';
  if (b < 1048576)    return (b/1024).toFixed(1) + ' KB';
  if (b < 1073741824) return (b/1048576).toFixed(1) + ' MB';
  return (b/1073741824).toFixed(2) + ' GB';
}

// Makes an element behave as a keyboard/ARIA toggle button.
export function setToggleA11y(el, on, label) {
  el.tabIndex = 0;
  el.setAttribute('role', 'button');
  el.setAttribute('aria-pressed', on ? 'true' : 'false');
  el.setAttribute('aria-label', label);
}

// Custom event names — dispatch/listen with these constants to avoid typos.
export const EVT_FILE_LOADED    = 'analysis:fileLoaded';
export const EVT_RESULT_APPLIED = 'analysis:resultApplied';
// Fired by resetState() so subscribers (e.g. telop overlay) can clear stale caches.
export const EVT_STATE_RESET    = 'analysis:stateReset';
