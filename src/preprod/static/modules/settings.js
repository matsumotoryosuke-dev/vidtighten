import { S } from './state.js';
import { $, api, fmtDur, plur } from './utils.js';
import { renderWaveform } from './waveform.js';

let _previewTick = null;

export const SETTINGS_DEFAULTS = {
  threshold: '-40', minDur: '1', hangover: '300', padding: '200', model: 'turbo',
  fEnable: true, fJapanese: true, fEnglish: true, fCustom: '',
  tFps: '24', tRes: '3840x2160', tFont: 'Momochidori Heavy',
  tFontSize: '92', tFontColor: '#F3B500', tPosY: '-420', tLineSpacing: '-65',
};
const SK = 'vt_settings_v1';

export function bindSlider(id, valId, display) {
  const el = $(id), vl = $(valId);
  const up = () => { vl.textContent = display(parseFloat(el.value)); };
  el.addEventListener('input', up); up();
}

export function _readSettingsFromUI() {
  return {
    threshold:  $('s-threshold').value,
    minDur:     $('s-min-dur').value,
    hangover:   $('s-hangover').value,
    padding:    $('s-padding').value,
    model:      $('f-model').value,
    fEnable:    $('f-enable').checked,
    fJapanese:  $('f-japanese').checked,
    fEnglish:   $('f-english').checked,
    fCustom:    $('f-custom').value,
    tFps:       $('t-fps').value,
    tRes:       $('t-res').value,
    tFont:      $('t-font').value,
    tFontSize:  $('t-font-size').value,
    tFontColor:    $('t-font-color').value,
    tPosY:         $('t-pos-y').value,
    tLineSpacing:  $('t-line-spacing').value,
  };
}

export function _applySettingsToUI(s) {
  $('s-threshold').value  = s.threshold ?? SETTINGS_DEFAULTS.threshold;
  $('s-min-dur').value    = s.minDur    ?? SETTINGS_DEFAULTS.minDur;
  $('s-hangover').value   = s.hangover  ?? SETTINGS_DEFAULTS.hangover;
  $('s-padding').value    = s.padding   ?? SETTINGS_DEFAULTS.padding;
  $('f-model').value      = s.model     ?? SETTINGS_DEFAULTS.model;
  $('f-enable').checked   = s.fEnable   ?? SETTINGS_DEFAULTS.fEnable;
  $('f-japanese').checked = s.fJapanese ?? SETTINGS_DEFAULTS.fJapanese;
  $('f-english').checked  = s.fEnglish  ?? SETTINGS_DEFAULTS.fEnglish;
  $('f-custom').value     = s.fCustom   ?? SETTINGS_DEFAULTS.fCustom;
  $('t-fps').value        = s.tFps      ?? SETTINGS_DEFAULTS.tFps;
  $('t-res').value        = s.tRes      ?? SETTINGS_DEFAULTS.tRes;
  $('t-font').value       = s.tFont     ?? SETTINGS_DEFAULTS.tFont;
  $('t-font-size').value  = s.tFontSize ?? SETTINGS_DEFAULTS.tFontSize;
  $('t-font-color').value   = s.tFontColor   ?? SETTINGS_DEFAULTS.tFontColor;
  $('t-pos-y').value        = s.tPosY        ?? SETTINGS_DEFAULTS.tPosY;
  $('t-line-spacing').value = s.tLineSpacing ?? SETTINGS_DEFAULTS.tLineSpacing;
  // Refresh slider display labels
  ['s-threshold','s-min-dur','s-hangover','s-padding'].forEach(id => {
    $(id).dispatchEvent(new Event('input'));
  });
}

export function loadSavedSettings() {
  try {
    const raw = localStorage.getItem(SK);
    return raw ? JSON.parse(raw) : { ...SETTINGS_DEFAULTS };
  } catch { return { ...SETTINGS_DEFAULTS }; }
}

export function saveSettingsToStorage(s) {
  try { localStorage.setItem(SK, JSON.stringify(s)); } catch {}
}

export function cancelPreviewDebounce() {
  clearTimeout(_previewTick);
  _previewTick = null;
}

export function _updateSilPreview() {
  const badge = $('sil-preview-badge');
  if (!S.filePath || !S.waveformMaxAmp) {
    badge.style.display = 'none';
    if (S.silPreview !== null) { S.silPreview = null; renderWaveform(); }
    return;
  }
  badge.textContent = '…'; badge.style.display = '';
  clearTimeout(_previewTick);
  _previewTick = setTimeout(async () => {
    try {
      const r = await api('/api/analyze/redetect_silence', {
        path:         S.filePath,
        threshold_db: parseFloat($('s-threshold').value),
        min_duration: parseFloat($('s-min-dur').value),
        hangover_ms:  parseInt($('s-hangover').value),
      });
      if (!r.candidates) { badge.style.display = 'none'; S.silPreview = null; renderWaveform(); return; }
      S.silPreview = r.candidates;
      renderWaveform();
      const n = r.candidates.length;
      let dur = '';
      if (n > 0) {
        dur = ` · ${fmtDur(r.total_duration)}`;
        if (S.media?.duration > 0) {
          const pct = Math.round(r.total_duration / S.media.duration * 100);
          dur += ` (${pct}%)`;
        }
      }
      badge.textContent = plur(n, 'silence') + dur;
    } catch (_) { badge.style.display = 'none'; S.silPreview = null; renderWaveform(); }
  }, 300);
}
