import { S } from './state.js';

export function clampZoomOffset() {
  const z = S.zoom;
  z.offset = Math.max(0, Math.min(1 - 1/z.level, z.offset));
}

export function applyZoom(newLevel, pivotFrac) {
  // pivotFrac: canvas-x fraction (0-1) that should stay fixed during zoom
  const z = S.zoom;
  const pivotTimeFrac = pivotFrac / z.level + z.offset;
  z.level  = Math.max(1, Math.min(64, newLevel));
  z.offset = pivotTimeFrac - pivotFrac / z.level;
  clampZoomOffset();
  updateZoomLabel();
}

export function updateZoomLabel() {
  const lv = S.zoom.level;
  document.getElementById('zoom-label').textContent = lv === 1 ? '1×' : lv < 10 ? lv.toFixed(1).replace('.0','')+'×' : Math.round(lv)+'×';
  document.getElementById('waveform-wrap').classList.toggle('zoomed', lv > 1);
}

// Time <-> canvas-X conversions (zoom-aware)
export function timeToX(t, W) {
  const dur = S.media?.duration || 1;
  return ((t/dur - S.zoom.offset) * S.zoom.level) * W;
}

export function xToTime(x, W) {
  const dur = S.media?.duration || 1;
  return (x/W/S.zoom.level + S.zoom.offset) * dur;
}
