/**
 * Export Dialog — consolidated export popover replacing the 3 toolbar export buttons.
 *
 * Opens anchored below #btn-export as a 360-px popover with three tabs:
 *   Rough-Cut | Telop | Subtitles
 * Telop/Subtitles tabs are disabled when no transcript exists.
 * Padding slider syncs bidirectionally with #s-padding in the Settings drawer.
 * Last-used tab is persisted to localStorage under _PREFS_KEY.
 */
import { S, collectDeletedWordIds } from './state.js';
import { $, apiFetch, dlBlob, toast } from './utils.js';
import { rebuildTelopText } from './export-module.js';

const _PREFS_KEY = 'vidtighten.export.prefs';
let _isOpen = false;

// ── Public API ──────────────────────────────────────────────────────

export function initExportDialog() {
  // Tab switching
  document.querySelectorAll('.export-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      if (tab.disabled) return;
      _setTab(tab.dataset.tab);
    });
  });

  // Format toggle (SRT / VTT)
  document.querySelectorAll('.ed-format-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.ed-format-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });

  // Timing toggle (Edited / Source) for telop export
  document.querySelectorAll('.ed-timing-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.ed-timing-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });

  // Padding slider → update display + sync to Settings drawer slider
  const edPad = $('ed-padding');
  const padVal = $('export-padding-val');
  edPad.addEventListener('input', () => {
    padVal.textContent = edPad.value + ' ms';
    $('s-padding').value = edPad.value;
    $('s-padding-val').textContent = edPad.value + ' ms';
  });

  // Cancel button
  $('btn-export-cancel').addEventListener('click', closeExportDialog);

  // CTA export button
  $('btn-export-do').addEventListener('click', _doExport);

  // Close on outside click
  document.addEventListener('click', e => {
    if (!_isOpen) return;
    const dlg = $('export-dialog');
    const btn = $('btn-export');
    if (dlg && !dlg.contains(e.target) && e.target !== btn) closeExportDialog();
  });
}

export function openExportDialog() {
  if (!S.filePath) return;

  // Sync padding from Settings drawer
  const mainPad = $('s-padding');
  const edPad   = $('ed-padding');
  edPad.value = mainPad.value;
  $('export-padding-val').textContent = mainPad.value + ' ms';

  // Sync telop settings from Settings drawer into dialog
  _syncTelopFromDrawer();

  // Enable / disable Telop + Subtitles tabs
  const hasTelop = S.telopEntries && S.telopEntries.length > 0;
  const tabTelop = document.querySelector('.export-tab[data-tab="telop"]');
  const tabSubs  = document.querySelector('.export-tab[data-tab="subtitles"]');
  if (tabTelop) { tabTelop.disabled = !hasTelop; tabTelop.style.opacity = hasTelop ? '' : '0.4'; }
  if (tabSubs)  { tabSubs.disabled  = !hasTelop; tabSubs.style.opacity  = hasTelop ? '' : '0.4'; }

  // Restore last tab (fall back to roughcut if transcript unavailable)
  const prefs = _loadPrefs();
  let tab = prefs.tab || 'roughcut';
  if (!hasTelop && (tab === 'telop' || tab === 'subtitles')) tab = 'roughcut';
  _setTab(tab);

  // Position below the trigger button
  const trigBtn = $('btn-export');
  const r = trigBtn.getBoundingClientRect();
  const dlg = $('export-dialog');
  dlg.style.top   = (r.bottom + 6) + 'px';
  dlg.style.right  = (window.innerWidth - r.right) + 'px';
  dlg.style.left   = 'auto';
  dlg.classList.add('open');
  _isOpen = true;
}

export function closeExportDialog() {
  $('export-dialog').classList.remove('open');
  _isOpen = false;
}

export function isExportDialogOpen() { return _isOpen; }

// ── Internal helpers ────────────────────────────────────────────────

function _setTab(tabId) {
  document.querySelectorAll('.export-tab').forEach(t =>
    t.classList.toggle('active', t.dataset.tab === tabId));
  document.querySelectorAll('.export-panel').forEach(p =>
    p.classList.toggle('active', p.dataset.panel === tabId));
  const labels = { roughcut: 'Export Rough-Cut', telop: 'Export Telop', subtitles: 'Export Subtitles' };
  $('btn-export-do').textContent = labels[tabId] || 'Export';
  _savePrefs({ tab: tabId });
}

function _syncTelopFromDrawer() {
  const pairs = [
    ['t-fps',          'ed-t-fps'],
    ['t-res',          'ed-t-res'],
    ['t-font',         'ed-t-font'],
    ['t-font-size',    'ed-t-font-size'],
    ['t-font-color',   'ed-t-font-color'],
    ['t-pos-y',        'ed-t-pos-y'],
    ['t-line-spacing', 'ed-t-line-spacing'],
  ];
  for (const [src, dst] of pairs) {
    const s = $(src), d = $(dst);
    if (s && d) d.value = s.value;
  }
}

function _activeTab() {
  return document.querySelector('.export-tab.active')?.dataset.tab || 'roughcut';
}

function _activeFmt() {
  return document.querySelector('.ed-format-btn.active')?.dataset.fmt || 'srt';
}

function _useSourceTiming() {
  return document.querySelector('.ed-timing-btn.active')?.dataset.timing === 'source';
}

function _stem() {
  return S.filePath ? S.filePath.split('/').pop().replace(/\.[^.]+$/, '') : 'export';
}

async function _doExport() {
  const tab = _activeTab();
  const padding = parseInt($('ed-padding').value);
  if (tab === 'roughcut')   await _exportRoughCut(padding);
  else if (tab === 'telop') await _exportTelop(padding);
  else                      await _exportSubtitles(padding);
}

async function _exportRoughCut(padding) {
  if (!S.filePath || !S.media) return;
  const stem = _stem();
  const regions = S.removalCandidates
    .filter(c => S.removalsEnabled.has(c.id))
    .map(c => ({ start: c.start, end: c.end, type: c.type }));
  const thrDb = parseFloat($('s-threshold')?.value ?? '-40');
  const btn = $('btn-export-do');
  btn.disabled = true;
  try {
    if (window.pywebview) {
      // Bypass fetch() entirely — any HTTP fetch from WKWebView can trigger
      // decidePolicyForNavigationResponse and blank the page regardless of MIME type.
      // Call the Python bridge method directly: no HTTP, no navigation delegate.
      const result = await window.pywebview.api.export_roughcut(S.filePath, regions, padding, thrDb);
      if (!result?.ok) { toast('Export failed: ' + (result?.error || 'unknown'), 4000); return; }
      toast('Rough-cut FCPXML saved to ~/Downloads');
      closeExportDialog();
      return;
    }
    const r = await apiFetch('/api/export/roughcut', { path: S.filePath, removal_regions: regions, padding_ms: padding, threshold_db: thrDb });
    if (!r.ok) { const e = await r.json(); toast('Export failed: ' + e.error, 4000); return; }
    const saved = await dlBlob(await r.blob(), `${stem}_cut.fcpxml`);
    if (!saved?.ok) return;
    toast('Rough-cut FCPXML saved to ~/Downloads');
    closeExportDialog();
  } finally { btn.disabled = false; }
}

async function _exportTelop(padding) {
  if (!S.filePath || !S.media) return;
  const stem = _stem();
  const useSourceTiming = _useSourceTiming();

  const deletedWordIds = collectDeletedWordIds();

  const entries = S.telopEntries
    .filter(t => S.telopsEnabled.has(t.id))
    .map(t => {
      const rebuilt = rebuildTelopText(t, S.words || [], deletedWordIds);
      return { start: rebuilt.start, end: rebuilt.end, text: rebuilt.text };
    })
    .filter(t => t.text);
  if (!entries.length) { toast('No telop entries enabled'); return; }
  const regions = S.removalCandidates
    .filter(c => S.removalsEnabled.has(c.id))
    .map(c => ({ start: c.start, end: c.end, type: c.type }));
  const [tw, th] = $('ed-t-res').value.split('x').map(Number);
  const settings = {
    fps:          $('ed-t-fps').value,
    width:        tw,
    height:       th,
    font:         $('ed-t-font').value,
    font_size:    parseInt($('ed-t-font-size').value),
    font_color:   $('ed-t-font-color').value,
    position_y:   parseInt($('ed-t-pos-y').value),
    line_spacing: parseInt($('ed-t-line-spacing').value),
  };
  const thrDb = parseFloat($('s-threshold')?.value ?? '-40');
  const btn = $('btn-export-do');
  btn.disabled = true;
  try {
    if (window.pywebview) {
      const result = await window.pywebview.api.export_telop(
        S.filePath, S.media.duration, entries, regions, padding, settings, stem, useSourceTiming, thrDb
      );
      if (!result?.ok) { toast('Export failed: ' + (result?.error || 'unknown'), 4000); return; }
      toast('Telop FCPXML saved to ~/Downloads');
      closeExportDialog();
      return;
    }
    const r = await apiFetch('/api/export/telop', {
      path: S.filePath, duration: S.media.duration, telop_entries: entries,
      removal_regions: regions, padding_ms: padding, settings, stem,
      use_source_timing: useSourceTiming, threshold_db: thrDb,
    });
    if (!r.ok) { const e = await r.json(); toast('Export failed: ' + e.error, 4000); return; }
    const saved = await dlBlob(await r.blob(), `${stem}_telop.fcpxml`);
    if (!saved?.ok) return;
    toast('Telop FCPXML saved to ~/Downloads');
    closeExportDialog();
  } finally { btn.disabled = false; }
}

async function _exportSubtitles(padding) {
  if (!S.filePath || !S.media || !S.telopEntries.length) return;
  const stem = _stem();
  const fmt = _activeFmt();

  const deletedWordIds = collectDeletedWordIds();

  const entries = S.telopEntries
    .map(t => {
      const rebuilt = rebuildTelopText(t, S.words || [], deletedWordIds);
      return { start: rebuilt.start, end: rebuilt.end, text: rebuilt.text };
    })
    .filter(t => t.text);
  const regions = S.removalCandidates
    .filter(c => S.removalsEnabled.has(c.id))
    .map(c => ({ start: c.start, end: c.end, type: c.type }));
  const thrDb = parseFloat($('s-threshold')?.value ?? '-40');
  const btn = $('btn-export-do');
  btn.disabled = true;
  try {
    if (window.pywebview) {
      const result = await window.pywebview.api.export_subtitles(
        fmt, S.filePath, entries, regions, S.media.duration, padding, stem, thrDb
      );
      if (!result?.ok) { toast(`${fmt.toUpperCase()} export failed: ` + (result?.error || 'unknown'), 4000); return; }
      toast(`${fmt.toUpperCase()} saved to ~/Downloads`);
      closeExportDialog();
      return;
    }
    const r = await apiFetch('/api/export/subtitles', {
      format: fmt, path: S.filePath, telop_entries: entries, removal_regions: regions,
      duration: S.media.duration, padding_ms: padding, stem, threshold_db: thrDb,
    });
    if (!r.ok) { const e2 = await r.json(); toast(`${fmt.toUpperCase()} export failed: ${e2.error}`, 4000); return; }
    const saved = await dlBlob(await r.blob(), `${stem}_subtitles.${fmt}`);
    if (!saved?.ok) return;
    toast(`${fmt.toUpperCase()} saved to ~/Downloads`);
    closeExportDialog();
  } finally { btn.disabled = false; }
}

function _loadPrefs()       { try { return JSON.parse(localStorage.getItem(_PREFS_KEY) || '{}'); } catch { return {}; } }
function _savePrefs(update) { try { localStorage.setItem(_PREFS_KEY, JSON.stringify({ ..._loadPrefs(), ...update })); } catch {} }
