import { $, toast } from './utils.js';

export function initDebug() {
  const btn    = $('btn-copy-logs');
  const pathEl = $('debug-log-path');
  if (!btn) return;

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    try {
      const data = await fetch('/api/debug/logs').then(r => r.json());
      if (data.log_path) pathEl.textContent = data.log_path;

      const lines = data.lines || [];
      if (lines.length === 0) {
        toast('No logs yet — run an analysis first');
        return;
      }

      const text = lines.join('\n');
      try {
        await navigator.clipboard.writeText(text);
      } catch {
        // Fallback for WKWebView clipboard restrictions
        const ta = Object.assign(document.createElement('textarea'), {
          value: text,
          style: 'position:fixed;opacity:0;top:0;left:0',
        });
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
      }
      toast(`Logs copied (${lines.length} lines)`);
    } catch (err) {
      toast('Failed to fetch logs: ' + err.message);
    } finally {
      btn.disabled = false;
    }
  });
}
