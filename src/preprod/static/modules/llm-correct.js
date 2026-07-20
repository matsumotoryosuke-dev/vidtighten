/**
 * Local-LLM transcript correction — frontend.
 *
 * Optional, manually-triggered pass that asks a local LLM (via the backend →
 * Ollama) to find brand/proper-noun mishearings in the transcript, then lets
 * Rio review each suggestion before ANY change is applied. See
 * 04_Context/vidtighten-llm-correct.md.
 *
 * Safety model (all mandatory, from the design review):
 *  - Never auto-apply: every fix is a checkbox in a review panel.
 *  - Per-row direction FLIP: the model reliably knows public names but inverts a
 *    user's private brand — flip lets Rio correct that in one click instead of
 *    the fix being silently wrong.
 *  - Revert buffer: the transcript's text is cloned ONCE right before a batch is
 *    applied, so "Undo corrections" restores it exactly. (Not wired into the
 *    per-toggle history stack — cloning 8000 words per silence-toggle is the
 *    cost tx-search deliberately avoided; this is one clone per LLM apply.)
 *  - "Always apply": graduates a fix into the persistent glossary so the next
 *    transcription fixes it deterministically and free.
 *
 * Applying a fix reuses tx-search's _applyReplace so brand replacements that
 * span multiple word tokens redistribute correctly and word ids/timestamps are
 * preserved (audio-cut + word-render integrity).
 */
import { S } from './state.js';
import { $, api, toast } from './utils.js';
import { _applyReplace } from './tx-search.js';

let _onApplied = () => {};   // re-render hook, provided by app.js
let _revert    = null;       // {entries:Map<id,text>, words:Map<id,text>} or null
let _pollTimer = null;

// ── Model selector ────────────────────────────────────────────────────────────

async function _loadModels() {
  const sel  = $('llm-model');
  const hint = $('llm-model-hint');
  if (!sel) return;
  let data;
  try {
    data = await (await fetch('/api/llm/models')).json();
  } catch {
    data = { available: false, models: [], default: null };
  }
  sel.innerHTML = '';
  if (!data.available || !data.models.length) {
    const opt = document.createElement('option');
    opt.textContent = data.available ? 'No models installed' : 'Ollama not running';
    opt.value = '';
    sel.appendChild(opt);
    sel.disabled = true;
    if (hint) hint.textContent = data.available
      ? 'Install a model with: ollama pull <name>'
      : 'Start Ollama to enable transcript correction.';
    _setTriggerEnabled(false);
    return;
  }
  const saved = _savedModel();
  const names = data.models.map(m => m.name);
  const chosen = names.includes(saved) ? saved : data.default;
  for (const m of data.models) {
    const opt = document.createElement('option');
    opt.value = m.name;
    opt.textContent = m.loaded ? `${m.name} (loaded)` : m.name;
    if (m.name === chosen) opt.selected = true;
    sel.appendChild(opt);
  }
  sel.disabled = false;
  if (hint) hint.textContent = '';
  _setTriggerEnabled(true);
}

function _savedModel() {
  try { return localStorage.getItem('vt_llm_model') || ''; } catch { return ''; }
}
function _saveModel(name) {
  try { localStorage.setItem('vt_llm_model', name); } catch {}
}
function _currentModel() {
  return $('llm-model')?.value || '';
}

function _setTriggerEnabled(on) {
  const btn = $('btn-llm-correct');
  if (btn) btn.disabled = !on;
}

// ── Run a correction pass ──────────────────────────────────────────────────────

function _transcriptText() {
  // Newline-joined display text — same view the anchor-validator checks against.
  return (S.telopEntries || []).map(e => e.text || '').join('\n');
}

async function _run() {
  if (!S.telopEntries?.length) { toast('No transcript to correct yet.', 3000); return; }
  const model = _currentModel();
  if (!model) { toast('No local model available.', 3000); return; }

  _setTriggerEnabled(false);
  toast('Scanning transcript with local model…', 2000);
  try {
    const start = await api('/api/llm/suggest', {
      transcript_text: _transcriptText(), model,
    });
    if (start.error) { toast('Correction failed: ' + start.error, 4000); return; }
    _poll(start.task_id);
  } catch (e) {
    toast('Correction failed: ' + e.message, 4000);
    _setTriggerEnabled(true);
  }
}

function _poll(taskId) {
  clearInterval(_pollTimer);
  let ticks = 0;
  _pollTimer = setInterval(async () => {
    ticks++;
    let d;
    try { d = await (await fetch(`/api/analyze/status/${taskId}`)).json(); }
    catch { return; }
    if (d.status === 'running') {
      if (ticks > 300) {   // ~90s ceiling; the model call itself is short
        clearInterval(_pollTimer);
        toast('Correction timed out.', 4000);
        _setTriggerEnabled(true);
      }
      return;
    }
    clearInterval(_pollTimer);
    _setTriggerEnabled(true);
    if (d.status === 'error') {
      const msg = d.result?.status === 'unavailable'
        ? 'Ollama isn’t running.' : (d.error || 'Correction failed.');
      toast(msg, 4000);
      return;
    }
    const fixes = d.result?.fixes || [];
    if (!fixes.length) { toast('No brand-name corrections found.', 3500); return; }
    _openReview(fixes);
  }, 300);
}

// ── Review UI ───────────────────────────────────────────────────────────────

function _openReview(fixes) {
  const overlay = $('llm-review-overlay');
  const list    = $('llm-review-list');
  if (!overlay || !list) return;
  list.innerHTML = '';
  // Each row owns its own {wrong, correct, count, apply, always} state via the DOM.
  for (const f of fixes) {
    const row = document.createElement('div');
    row.className = 'llm-row';
    row.dataset.wrong   = f.wrong;
    row.dataset.correct = f.correct;
    row.innerHTML = `
      <input type="checkbox" class="llm-row-apply" checked>
      <span class="llm-row-from"></span>
      <button class="llm-row-flip" title="Swap direction">⇄</button>
      <span class="llm-row-to"></span>
      <span class="llm-row-count"></span>
      <label class="llm-row-always"><input type="checkbox" class="llm-row-always-cb"> always</label>
    `;
    row.querySelector('.llm-row-from').textContent  = f.wrong;
    row.querySelector('.llm-row-to').textContent    = f.correct;
    row.querySelector('.llm-row-count').textContent = `×${f.count}`;
    row.querySelector('.llm-row-flip').addEventListener('click', () => {
      const w = row.dataset.wrong, c = row.dataset.correct;
      row.dataset.wrong = c; row.dataset.correct = w;
      row.querySelector('.llm-row-from').textContent = c;
      row.querySelector('.llm-row-to').textContent   = w;
      // Recount against the live transcript so the ×N reflects the new direction.
      const txt = _transcriptText();
      row.querySelector('.llm-row-count').textContent = `×${_countOf(txt, c)}`;
    });
    list.appendChild(row);
  }
  overlay.classList.add('open');
}

function _countOf(text, sub) {
  if (!sub) return 0;
  let n = 0, i = 0;
  while ((i = text.indexOf(sub, i)) !== -1) { n++; i += sub.length; }
  return n;
}

function _closeReview() { $('llm-review-overlay')?.classList.remove('open'); }

async function _applySelected() {
  const rows = [...document.querySelectorAll('#llm-review-list .llm-row')]
    .filter(r => r.querySelector('.llm-row-apply').checked);
  if (!rows.length) { _closeReview(); return; }

  // Revert buffer: clone every entry's + word's text ONCE before mutating.
  _revert = {
    entries: new Map(S.telopEntries.map(e => [e.id, e.text])),
    words:   new Map((S.words || []).map(w => [w.id, w.text])),
  };

  let applied = 0;
  const graduate = [];
  for (const row of rows) {
    const wrong = row.dataset.wrong, correct = row.dataset.correct;
    if (!wrong || wrong === correct) continue;
    for (const entry of S.telopEntries) {
      applied += _applyReplace(entry, S.words || [], wrong, correct);
    }
    if (row.querySelector('.llm-row-always-cb').checked) graduate.push({ wrong, correct });
  }

  // Persist "always apply" mappings to the glossary (best-effort, non-blocking).
  for (const g of graduate) {
    api('/api/llm/glossary/add', g).catch(() => {});
  }

  _closeReview();
  _onApplied();
  _showRevertBar();
  const gtxt = graduate.length ? `, ${graduate.length} saved to glossary` : '';
  toast(`Applied ${applied} correction${applied === 1 ? '' : 's'}${gtxt}.`, 4000);
}

// ── Revert ────────────────────────────────────────────────────────────────────

function _showRevertBar() {
  const bar = $('llm-revert-bar');
  if (bar) bar.classList.add('open');
}
function _hideRevertBar() {
  const bar = $('llm-revert-bar');
  if (bar) bar.classList.remove('open');
}

function _revertCorrections() {
  if (!_revert) { _hideRevertBar(); return; }
  for (const e of S.telopEntries) {
    if (_revert.entries.has(e.id)) e.text = _revert.entries.get(e.id);
  }
  for (const w of (S.words || [])) {
    if (_revert.words.has(w.id)) w.text = _revert.words.get(w.id);
  }
  _revert = null;
  _hideRevertBar();
  _onApplied();
  toast('Reverted to the pre-correction transcript.', 3000);
}

// ── Wiring ──────────────────────────────────────────────────────────────────

export function initLlmCorrect({ onApplied }) {
  _onApplied = onApplied || (() => {});
  const btn = $('btn-llm-correct');
  if (!btn) return;   // page variant without the control — skip silently

  // Populate the model selector lazily (Ollama health-check at point of use,
  // not app startup — a slow/absent Ollama must never delay boot).
  _loadModels();

  btn.addEventListener('click', _run);
  $('llm-model')?.addEventListener('change', () => _saveModel(_currentModel()));
  $('llm-refresh-models')?.addEventListener('click', _loadModels);

  // Auto-run opt-in checkbox: reflect + persist.
  const auto = $('llm-auto');
  if (auto) {
    try { auto.checked = localStorage.getItem('vt_llm_auto') === '1'; } catch {}
    auto.addEventListener('change', () => {
      try { localStorage.setItem('vt_llm_auto', auto.checked ? '1' : '0'); } catch {}
    });
  }
  $('llm-review-apply')?.addEventListener('click', _applySelected);
  $('llm-review-cancel')?.addEventListener('click', _closeReview);
  $('llm-revert-btn')?.addEventListener('click', _revertCorrections);
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') _closeReview();
  });

  // Opt-in: run automatically after an analysis completes, if enabled.
  document.addEventListener('analysis:resultApplied', () => {
    let optIn = false;
    try { optIn = localStorage.getItem('vt_llm_auto') === '1'; } catch {}
    if (optIn && S.telopEntries?.length) _run();
  });
}
