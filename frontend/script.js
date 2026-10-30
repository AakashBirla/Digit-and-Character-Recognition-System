/**
 * Digit and Character Recognition System — frontend controller (vanilla JS, no dependencies).
 *
 * Responsibilities
 *  - Drawing: a fixed 400×400 logical canvas rendered at devicePixelRatio for crisp strokes.
 *    Strokes are recorded as vectors, so resizing, DPR changes and brush changes never lose
 *    the drawing (and Undo is cheap). Pointer Events with pointer capture, coalesced events,
 *    pen pressure, the pen's eraser end and palm rejection give a good pen-tablet feel.
 *  - Export: the transparent drawing layer is composited onto an opaque black 400×400 PNG
 *    (white ink on black), which is what POST /api/predict expects.
 *  - API: polls GET /api/health every 15 s (online / degraded / offline) and renders the
 *    prediction response (top-5, all class probabilities, preprocessed 28×28 preview).
 *
 * Everything lives inside this IIFE: no globals are created.
 */
(function () {
  'use strict';

  /* ======================================================================
   * Configuration
   * ==================================================================== */

  const CONFIG = Object.freeze({
    LOGICAL_SIZE: 400,          // drawing coordinates are always 0..400 on both axes
    MAX_DPR: 3,                 // cap the backing-store resolution (memory / speed)
    BRUSH_MIN: 4,
    BRUSH_MAX: 48,
    BRUSH_DEFAULT: 22,
    BRUSH_STEP: 2,              // [ and ] change the brush by this amount
    MIN_POINT_DISTANCE: 0.6,    // logical px; closer points are skipped (noise)
    PRESSURE_SMOOTHING: 0.45,   // 0 = raw pressure, 1 = frozen width
    PALM_REJECT_MS: 600,        // ignore touch shortly after/while a pen is near the canvas
    MAX_UNDO: 100,
    INK_ALPHA_THRESHOLD: 40,    // mirrors the backend ink threshold (0-255)
    MIN_INK_PIXELS: 20,         // fewer inked pixels → treat the canvas as empty
    HEALTH_INTERVAL_MS: 15000,
    HEALTH_TIMEOUT_MS: 6000,
    PREDICT_TIMEOUT_MS: 30000,
    TOP_K: 5,
    STORAGE_PREFIX: 'hwr.',
  });

  /** Static facts about each mode (no metrics here — those come from /api/health only). */
  const MODES = Object.freeze({
    digits: Object.freeze({
      key: 'digits',
      datasetName: 'MNIST',
      modelName: 'MNIST CNN',
      numClasses: 10,
      trainCommand: 'python train_mnist.py',
      prompt: 'Draw a single digit (0–9). Large and centered works best.',
      emptyText: 'Draw a digit from 0 to 9 on the canvas, then press Predict.',
      emptyGlyph: '?',
    }),
    characters: Object.freeze({
      key: 'characters',
      datasetName: 'EMNIST ByClass',
      modelName: 'EMNIST CNN',
      numClasses: 62,
      trainCommand: 'python train_emnist.py',
      prompt: 'Draw a single character: 0–9, A–Z or a–z.',
      emptyText: 'Draw a digit or letter (0–9, A–Z, a–z) on the canvas, then press Predict.',
      emptyGlyph: '?',
    }),
  });

  const CATEGORY_NAMES = Object.freeze({
    digit: 'Digit',
    uppercase: 'Uppercase letter',
    lowercase: 'Lowercase letter',
  });

  /** Letters whose upper- and lowercase handwritten shapes are nearly identical in EMNIST. */
  const CASE_AMBIGUOUS = new Set('cCkKmMoOpPsSuUvVwWxXyYzZ'.split(''));
  /** Other well-known look-alike groups. */
  const LOOKALIKE_GROUPS = Object.freeze([
    ['0', 'O', 'o'],
    ['1', 'l', 'I', 'i'],
    ['2', 'Z', 'z'],
    ['5', 'S', 's'],
    ['9', 'g', 'q'],
  ]);

  const ERROR_TITLES = Object.freeze({
    blank_canvas: 'Nothing to recognize',
    invalid_image: 'The drawing could not be read',
    preprocessing_error: 'The drawing could not be processed',
    model_unavailable: 'Model not available',
    invalid_mode: 'Unknown recognition mode',
    payload_too_large: 'Drawing is too large to upload',
    timeout: 'The server took too long',
    network: 'Could not reach the server',
  });

  /* ======================================================================
   * State
   * ==================================================================== */

  /**
   * Single mutable state object for the whole page.
   * @type {{
   *   mode: 'digits'|'characters', brushSize: number, eraser: boolean,
   *   strokes: Array<Stroke>, undoStack: Array<'stroke'|{cleared: Array<Stroke>}>,
   *   activePointerId: (number|null), activePointerType: (string|null), activeStroke: (Stroke|null),
   *   lastPenTime: number, dpr: number, scale: number,
   *   backend: 'checking'|'online'|'degraded'|'offline', health: (object|null),
   *   healthTimer: (number|null), healthInFlight: boolean,
   *   predicting: boolean, predictController: (AbortController|null),
   *   result: (object|null), filter: string, topOnly: boolean,
   *   backendRenderPending: boolean, messageKind: (string|null)
   * }}
   */
  const state = {
    mode: 'digits',
    brushSize: CONFIG.BRUSH_DEFAULT,
    eraser: false,
    strokes: [],
    undoStack: [],
    activePointerId: null,
    activePointerType: null,
    activeStroke: null,
    lastPenTime: -Infinity,
    dpr: 1,
    scale: 1,
    backend: 'checking',
    health: null,
    healthTimer: null,
    healthInFlight: false,
    predicting: false,
    predictController: null,
    result: null,
    filter: '',
    topOnly: false,
    backendRenderPending: false, // a status re-render waiting for the current stroke to end
    messageKind: null,           // 'connection' while the result card shows a server-offline message
  };

  /**
   * @typedef {{x: number, y: number, w: number}} StrokePoint  logical coords + line width
   * @typedef {{erase: boolean, size: number, points: StrokePoint[], ended: boolean, sawPressure: boolean}} Stroke
   */

  /** DOM references, filled by cacheElements(). */
  const els = {};
  let ctx = null;
  let exportCanvas = null;

  /* ======================================================================
   * Small utilities
   * ==================================================================== */

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function isFiniteNumber(value) {
    return typeof value === 'number' && Number.isFinite(value);
  }

  /**
   * Format a probability (0..1) as a percentage without ever implying certainty:
   * values that would round to 100% are shown as ">99.9%".
   */
  function formatPercent(p, digits = 1) {
    if (!isFiniteNumber(p)) return '—';
    const pct = clamp(p, 0, 1) * 100;
    const limit = 100 - 0.5 * Math.pow(10, -digits);
    if (pct > 0 && pct < Math.pow(10, -digits)) return '<' + Math.pow(10, -digits).toFixed(digits) + '%';
    if (pct >= limit) return '>' + (100 - Math.pow(10, -digits)).toFixed(digits) + '%';
    return pct.toFixed(digits) + '%';
  }

  /** Accept accuracy as a 0..1 fraction (spec) but tolerate 0..100 percentages. */
  function formatAccuracy(value, digits = 2) {
    if (!isFiniteNumber(value)) return null;
    const fraction = value > 1 ? value / 100 : value;
    return (clamp(fraction, 0, 1) * 100).toFixed(digits) + '%';
  }

  function formatInteger(value) {
    return isFiniteNumber(value) ? Math.round(value).toLocaleString('en-US') : null;
  }

  function formatDate(iso) {
    if (typeof iso !== 'string' || !iso) return null;
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return null;
    return date.toLocaleString(undefined, { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  }

  function storageGet(key) {
    try {
      return window.localStorage.getItem(CONFIG.STORAGE_PREFIX + key);
    } catch (_err) {
      return null;
    }
  }

  function storageSet(key, value) {
    try {
      window.localStorage.setItem(CONFIG.STORAGE_PREFIX + key, String(value));
    } catch (_err) {
      /* storage unavailable (private mode, blocked) — preferences are optional */
    }
  }

  /** Create an element with optional class and text content (never innerHTML). */
  function h(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  /** Announce a message to screen readers via the polite live region. */
  let announceTimer = null;
  function announce(message) {
    if (!els.announcer) return;
    window.clearTimeout(announceTimer);
    els.announcer.textContent = '';
    announceTimer = window.setTimeout(() => {
      els.announcer.textContent = message;
    }, 60);
  }

  /** Derive a category from a single-character label when the API did not send one. */
  function categoryFromLabel(label) {
    if (/^[0-9]$/.test(label)) return 'digit';
    if (/^[A-Z]$/.test(label)) return 'uppercase';
    if (/^[a-z]$/.test(label)) return 'lowercase';
    return null;
  }

  function currentModeInfo() {
    return MODES[state.mode];
  }

  /** Model info for the selected mode from the last health response (or null). */
  function currentModelHealth() {
    const models = state.health && state.health.models;
    if (!models || typeof models !== 'object') return null;
    const info = models[state.mode];
    return info && typeof info === 'object' ? info : null;
  }

  /** True when we positively know the selected model cannot serve predictions. */
  function isModelUnavailable() {
    const info = currentModelHealth();
    return Boolean(info) && info.status !== 'loaded';
  }

  /* ======================================================================
   * Networking
   * ==================================================================== */

  /** Error raised for transport-level failures (timeout, offline, cancelled). */
  class RequestError extends Error {
    /** @param {'timeout'|'network'|'aborted'} kind */
    constructor(kind, message) {
      super(message);
      this.name = 'RequestError';
      this.kind = kind;
    }
  }

  /**
   * fetch() + timeout + optional external abort signal; parses JSON bodies defensively.
   * @returns {Promise<{response: Response, data: (object|null)}>}
   */
  async function fetchJson(url, { method = 'GET', body, timeoutMs, signal } = {}) {
    const controller = new AbortController();
    const forwardAbort = () => controller.abort();
    if (signal) {
      if (signal.aborted) controller.abort();
      else signal.addEventListener('abort', forwardAbort, { once: true });
    }
    let timedOut = false;
    const timer = window.setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, timeoutMs);

    try {
      const headers = { Accept: 'application/json' };
      if (body !== undefined) headers['Content-Type'] = 'application/json';
      const response = await fetch(url, {
        method,
        headers,
        body,
        signal: controller.signal,
        cache: 'no-store',
        credentials: 'same-origin',
      });
      let data = null;
      const contentType = response.headers.get('content-type') || '';
      if (contentType.includes('application/json')) {
        try {
          data = await response.json();
        } catch (_err) {
          data = null;
        }
      }
      if (controller.signal.aborted) throw new DOMException('Aborted', 'AbortError');
      return { response, data: data && typeof data === 'object' ? data : null };
    } catch (err) {
      if (timedOut) throw new RequestError('timeout', 'The server did not respond in time. Please try again.');
      if (signal && signal.aborted) throw new RequestError('aborted', 'Request cancelled.');
      throw new RequestError('network', 'Could not reach the recognition server. Is it running?');
    } finally {
      window.clearTimeout(timer);
      if (signal) signal.removeEventListener('abort', forwardAbort);
    }
  }

  /* ----------------------------- health ------------------------------ */

  function scheduleHealthCheck(delayMs = CONFIG.HEALTH_INTERVAL_MS) {
    window.clearTimeout(state.healthTimer);
    state.healthTimer = window.setTimeout(checkHealth, delayMs);
  }

  /** GET /api/health → update backend state, model info and availability. */
  async function checkHealth() {
    if (state.healthInFlight) return;
    state.healthInFlight = true;
    window.clearTimeout(state.healthTimer);
    const previous = state.backend;

    try {
      const { response, data } = await fetchJson('/api/health', { timeoutMs: CONFIG.HEALTH_TIMEOUT_MS });
      if (!response.ok || !data || data.success === false) {
        throw new RequestError('network', 'Health check failed (HTTP ' + response.status + ').');
      }
      state.health = data;
      state.backend = data.status === 'ok' ? 'online' : 'degraded';
    } catch (_err) {
      state.health = null;
      state.backend = 'offline';
    } finally {
      state.healthInFlight = false;
      renderBackendState();
      if (state.backend !== 'offline' && state.messageKind === 'connection') {
        showResultState('empty'); // the "server offline" advice no longer applies
      }
      if (previous !== state.backend && previous !== 'checking') {
        announce('Backend status changed: ' + state.backend + '.');
      }
      scheduleHealthCheck();
    }
  }

  /* ======================================================================
   * Canvas: setup and resolution handling
   * ==================================================================== */

  /**
   * Size the backing store to LOGICAL_SIZE × devicePixelRatio and re-render all strokes.
   * The element's CSS size is independent (it fills its square wrapper), so window
   * resizes do not touch the canvas at all; only real DPR changes come through here.
   */
  function setupBackingStore() {
    const dpr = clamp(window.devicePixelRatio || 1, 1, CONFIG.MAX_DPR);
    const pixels = Math.round(CONFIG.LOGICAL_SIZE * dpr);
    state.dpr = dpr;
    state.scale = pixels / CONFIG.LOGICAL_SIZE;
    if (els.canvas.width !== pixels || els.canvas.height !== pixels) {
      els.canvas.width = pixels;   // resets the bitmap and context state…
      els.canvas.height = pixels;
    }
    redrawAll();                    // …so replay the recorded strokes
  }

  /** Rebuild the backing store only if the (clamped) devicePixelRatio really changed. */
  function syncDevicePixelRatio() {
    const dpr = clamp(window.devicePixelRatio || 1, 1, CONFIG.MAX_DPR);
    if (Math.abs(dpr - state.dpr) > 0.001) setupBackingStore();
  }

  /**
   * Watch for DPR changes (browser zoom, moving the window to another monitor) through
   * several signals, because no single one fires reliably in every browser. None of
   * them clears the drawing: a plain resize only changes the CSS size of the element.
   */
  function watchDevicePixelRatio() {
    const onSizeChange = () => {
      syncDevicePixelRatio();
      updateBrushCursorSize();
    };

    // 1) Resolution media query, re-armed for the new DPR after each change.
    let mql = null;
    const onResolutionChange = () => {
      syncDevicePixelRatio();
      listenResolution();
    };
    const listenResolution = () => {
      if (mql && mql.removeEventListener) mql.removeEventListener('change', onResolutionChange);
      if (!window.matchMedia) return;
      mql = window.matchMedia('(resolution: ' + (window.devicePixelRatio || 1) + 'dppx)');
      if (mql.addEventListener) mql.addEventListener('change', onResolutionChange);
    };
    listenResolution();

    // 2) Device-pixel size of the canvas element (changes with DPR in Chromium).
    if (typeof ResizeObserver === 'function') {
      const observer = new ResizeObserver(onSizeChange);
      try {
        observer.observe(els.canvas, { box: 'device-pixel-content-box' });
      } catch (_err) {
        observer.observe(els.canvas);
      }
    }

    // 3) Plain window resize (browser zoom fires this too).
    window.addEventListener('resize', onSizeChange);
  }

  /** Clear the bitmap and replay every recorded stroke. */
  function redrawAll() {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalCompositeOperation = 'source-over';
    ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
    ctx.setTransform(state.scale, 0, 0, state.scale, 0, 0);
    for (const stroke of state.strokes) renderStroke(stroke);
    ctx.globalCompositeOperation = 'source-over';
  }

  /* ======================================================================
   * Canvas: stroke rendering (shared by live drawing and replay)
   * ==================================================================== */

  function applyStrokeStyle(stroke) {
    // destination-out removes alpha (true eraser) because the drawing layer is transparent.
    ctx.globalCompositeOperation = stroke.erase ? 'destination-out' : 'source-over';
    ctx.strokeStyle = '#ffffff';
    ctx.fillStyle = '#ffffff';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
  }

  function midpoint(a, b) {
    return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, w: (a.w + b.w) / 2 };
  }

  /** A single tap: a filled dot of the current width. */
  function drawDot(p) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.w / 2, 0, Math.PI * 2);
    ctx.fill();
  }

  /**
   * Draw the piece of the stroke introduced by point k (k >= 1).
   * Curves go from midpoint to midpoint with the real sample as the control point,
   * which yields smooth quadratic curves without lagging behind the pen.
   */
  function drawSegment(points, k) {
    const p0 = points[k - 1];
    const p1 = points[k];
    ctx.beginPath();
    if (k === 1) {
      const end = midpoint(p0, p1);
      ctx.lineWidth = p0.w;
      ctx.moveTo(p0.x, p0.y);
      ctx.lineTo(end.x, end.y);
    } else {
      const start = midpoint(points[k - 2], p0);
      const end = midpoint(p0, p1);
      ctx.lineWidth = p0.w;
      ctx.moveTo(start.x, start.y);
      ctx.quadraticCurveTo(p0.x, p0.y, end.x, end.y);
    }
    ctx.stroke();
  }

  /** Close the stroke: straight line from the last midpoint to the last sample. */
  function drawTail(points) {
    const n = points.length;
    if (n < 2) return;
    const start = midpoint(points[n - 2], points[n - 1]);
    const last = points[n - 1];
    ctx.beginPath();
    ctx.lineWidth = last.w;
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(last.x, last.y);
    ctx.stroke();
  }

  /** Replay a complete (or in-progress) stroke. */
  function renderStroke(stroke) {
    const pts = stroke.points;
    if (!pts.length) return;
    applyStrokeStyle(stroke);
    drawDot(pts[0]);
    for (let k = 1; k < pts.length; k += 1) drawSegment(pts, k);
    if (stroke.ended) drawTail(pts);
  }

  /* ======================================================================
   * Canvas: pointer input
   * ==================================================================== */

  /** Map a pointer event to logical 0..400 canvas coordinates. */
  function toLogical(event, rect) {
    const sx = CONFIG.LOGICAL_SIZE / (rect.width || 1);
    const sy = CONFIG.LOGICAL_SIZE / (rect.height || 1);
    return { x: (event.clientX - rect.left) * sx, y: (event.clientY - rect.top) * sy };
  }

  /**
   * Width for a sample: pressure-sensitive for pens, constant otherwise, never too thin.
   * Pens that support pressure report 0 at their lightest contact (first touch, lift-off), so once a
   * stroke has reported real pressure a 0 means "thinnest"; a pen that never reports pressure (and a
   * pen-down before any reading) is drawn at the full / minimum width instead of a blob.
   * @param {PointerEvent} event
   * @param {number} size  brush size
   * @param {Stroke|null} stroke  the stroke being drawn (null for the pen-down sample)
   */
  function widthForSample(event, size, stroke) {
    const minWidth = Math.max(2, 0.35 * size);
    let width = size;
    if (event.pointerType === 'pen') {
      if (event.pressure > 0) {
        if (stroke) stroke.sawPressure = true;
        width = size * (0.35 + 0.65 * event.pressure);
      } else if (!stroke || stroke.sawPressure) {
        width = minWidth;
      }
    }
    return Math.max(width, minWidth);
  }

  function isPenEraser(event) {
    return event.pointerType === 'pen' && (event.button === 5 || (event.buttons & 32) === 32);
  }

  /** Decide whether a pointerdown should start a stroke (palm/multi-touch/button filtering). */
  function shouldStartStroke(event) {
    if (!event.isPrimary) return false;                                   // second finger, palm
    if (event.pointerType === 'mouse' && event.button !== 0) return false; // right/middle click
    if (event.pointerType === 'pen' && event.button !== 0 && event.button !== 5) return false; // barrel button
    if (event.pointerType === 'touch' && performance.now() - state.lastPenTime < CONFIG.PALM_REJECT_MS) {
      return false;                                                       // palm while the pen is near
    }
    return true;
  }

  function onPointerDown(event) {
    if (event.pointerType === 'pen') state.lastPenTime = performance.now();

    // A pen arriving while a touch "stroke" is in progress means that touch was a palm: drop it.
    if (state.activePointerId !== null) {
      if (event.pointerType === 'pen' && state.activePointerType === 'touch') {
        discardActiveStroke();
      } else {
        return;
      }
    }
    if (!shouldStartStroke(event)) return;

    event.preventDefault();
    releaseButtonFocus();
    syncDevicePixelRatio(); // cheap safety net in case a DPR change went unnoticed
    try {
      els.canvas.setPointerCapture(event.pointerId);
    } catch (_err) {
      /* synthetic or already-released pointer: drawing still works without capture */
    }

    invalidateResult();

    const rect = els.canvas.getBoundingClientRect();
    const pos = toLogical(event, rect);
    const width = widthForSample(event, state.brushSize, null);
    /** @type {Stroke} */
    const stroke = {
      erase: state.eraser || isPenEraser(event),
      size: state.brushSize,
      points: [{ x: pos.x, y: pos.y, w: width }],
      ended: false,
      sawPressure: event.pointerType === 'pen' && event.pressure > 0,
    };

    state.activePointerId = event.pointerId;
    state.activePointerType = event.pointerType;
    state.activeStroke = stroke;
    state.strokes.push(stroke);

    applyStrokeStyle(stroke);
    drawDot(stroke.points[0]);
    updateCanvasChrome();
    updateBrushCursor(event, stroke.erase);
  }

  function onPointerMove(event) {
    if (event.pointerType === 'pen') state.lastPenTime = performance.now();

    if (event.pointerId !== state.activePointerId) {
      updateBrushCursor(event, state.eraser || isPenEraser(event));
      return;
    }
    event.preventDefault();

    const stroke = state.activeStroke;
    const rect = els.canvas.getBoundingClientRect();
    const coalesced = typeof event.getCoalescedEvents === 'function' ? event.getCoalescedEvents() : null;
    const samples = coalesced && coalesced.length ? coalesced : [event];

    applyStrokeStyle(stroke);
    for (const sample of samples) addSample(stroke, sample, rect);
    updateBrushCursor(event, stroke.erase);
  }

  /** Append one input sample to the active stroke and draw the new segment. */
  function addSample(stroke, sample, rect) {
    const pos = toLogical(sample, rect);
    const pts = stroke.points;
    const last = pts[pts.length - 1];
    if (Math.hypot(pos.x - last.x, pos.y - last.y) < CONFIG.MIN_POINT_DISTANCE) return;

    let width = widthForSample(sample, stroke.size, stroke);
    if (sample.pointerType === 'pen') {
      // Light smoothing avoids "beading" from noisy pressure readings.
      width = last.w * CONFIG.PRESSURE_SMOOTHING + width * (1 - CONFIG.PRESSURE_SMOOTHING);
    }
    pts.push({ x: pos.x, y: pos.y, w: width });
    drawSegment(pts, pts.length - 1);
  }

  /** pointerup / pointercancel / lostpointercapture all finish the stroke. */
  function onPointerEnd(event) {
    if (event.pointerId !== state.activePointerId) return;
    finishActiveStroke();
    try {
      if (els.canvas.hasPointerCapture && els.canvas.hasPointerCapture(event.pointerId)) {
        els.canvas.releasePointerCapture(event.pointerId);
      }
    } catch (_err) {
      /* ignore */
    }
    if (event.type !== 'pointerup' || event.pointerType === 'touch') hideBrushCursor();
  }

  function finishActiveStroke() {
    const stroke = state.activeStroke;
    state.activePointerId = null;
    state.activePointerType = null;
    state.activeStroke = null;
    if (!stroke) return;

    stroke.ended = true;
    applyStrokeStyle(stroke);
    drawTail(stroke.points);
    ctx.globalCompositeOperation = 'source-over';

    pushUndo('stroke');
    updateCanvasChrome();
    if (state.backendRenderPending) renderBackendState();
  }

  /** Remove the in-progress stroke entirely (used for palm rejection). */
  function discardActiveStroke() {
    const stroke = state.activeStroke;
    if (stroke) {
      const index = state.strokes.lastIndexOf(stroke);
      if (index !== -1) state.strokes.splice(index, 1);
    }
    try {
      if (state.activePointerId !== null) els.canvas.releasePointerCapture(state.activePointerId);
    } catch (_err) {
      /* ignore */
    }
    state.activePointerId = null;
    state.activePointerType = null;
    state.activeStroke = null;
    redrawAll();
    updateCanvasChrome();
    if (state.backendRenderPending) renderBackendState();
  }

  /**
   * Drawing should not leave keyboard focus on a toolbar button: otherwise pressing Enter
   * afterwards would "click" e.g. Clear instead of running the Predict shortcut.
   */
  function releaseButtonFocus() {
    const active = document.activeElement;
    if (active && active !== document.body && typeof active.blur === 'function') active.blur();
  }

  /* ----------------------------- brush cursor ------------------------- */

  function updateBrushCursorSize() {
    if (!els.canvas || !els.brushCursor) return;
    const rect = els.canvas.getBoundingClientRect();
    const cssSize = Math.max(4, state.brushSize * (rect.width / CONFIG.LOGICAL_SIZE));
    els.brushCursor.style.width = cssSize + 'px';
    els.brushCursor.style.height = cssSize + 'px';
    return cssSize;
  }

  /** Show a ring the size of the brush under a hovering mouse/pen (not for touch). */
  function updateBrushCursor(event, erasing) {
    if (event.pointerType === 'touch') {
      hideBrushCursor();
      return;
    }
    const rect = els.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    if (x < 0 || y < 0 || x > rect.width || y > rect.height) {
      hideBrushCursor();
      return;
    }
    const size = updateBrushCursorSize();
    els.brushCursor.style.transform = 'translate(' + (x - size / 2) + 'px, ' + (y - size / 2) + 'px)';
    els.brushCursor.classList.toggle('is-eraser', Boolean(erasing));
    els.brushCursor.classList.add('is-visible');
  }

  function hideBrushCursor() {
    if (els.brushCursor) els.brushCursor.classList.remove('is-visible');
  }

  /* ======================================================================
   * Canvas: tools (brush, eraser, undo, clear) and export
   * ==================================================================== */

  /** Any visible (non-eraser) stroke recorded since the last clear. */
  function hasInk() {
    return state.strokes.some((stroke) => !stroke.erase && stroke.points.length > 0);
  }

  function pushUndo(entry) {
    state.undoStack.push(entry);
    if (state.undoStack.length > CONFIG.MAX_UNDO) state.undoStack.shift();
  }

  function setBrushSize(size, { persist = true } = {}) {
    const value = clamp(Math.round(Number(size) || CONFIG.BRUSH_DEFAULT), CONFIG.BRUSH_MIN, CONFIG.BRUSH_MAX);
    state.brushSize = value;
    els.brushInput.value = String(value);
    els.brushValue.textContent = value + ' px';
    const fill = ((value - CONFIG.BRUSH_MIN) / (CONFIG.BRUSH_MAX - CONFIG.BRUSH_MIN)) * 100;
    els.brushInput.style.setProperty('--fill', fill.toFixed(1) + '%');
    updateBrushCursorSize();
    if (persist) storageSet('brush', value);
    // Note: the canvas is NOT touched — existing strokes keep their own recorded widths.
  }

  function setEraser(enabled) {
    state.eraser = Boolean(enabled);
    els.eraserBtn.setAttribute('aria-pressed', String(state.eraser));
    els.brushCursor.classList.toggle('is-eraser', state.eraser);
  }

  function undo() {
    if (state.activePointerId !== null) return;
    const entry = state.undoStack.pop();
    if (!entry) return;
    if (entry === 'stroke') {
      state.strokes.pop();
    } else if (entry && Array.isArray(entry.cleared)) {
      state.strokes = entry.cleared;
    }
    invalidateResult();
    redrawAll();
    updateCanvasChrome();
    announce('Undone.');
  }

  function clearCanvas({ announceIt = true } = {}) {
    if (state.activePointerId !== null) discardActiveStroke();
    if (state.strokes.length) pushUndo({ cleared: state.strokes });
    state.strokes = [];
    redrawAll();
    invalidateResult();
    updateCanvasChrome();
    if (announceIt) announce('Canvas cleared.');
  }

  /** Placeholder, Undo availability, etc. */
  function updateCanvasChrome() {
    // Erasing can remove all ink while strokes stay recorded, so probe the pixels in that case only.
    const erasedAll = !state.activeStroke && state.strokes.some((s) => s.erase) && canvasLooksEmpty();
    els.placeholder.hidden = state.strokes.length > 0 && !erasedAll;
    els.undoBtn.disabled = state.undoStack.length === 0;
  }

  /** True if no visible ink is left (cheap 50×50 alpha probe of the transparent drawing layer). */
  function canvasLooksEmpty() {
    const probe = document.createElement('canvas');
    probe.width = 50;
    probe.height = 50;
    const pctx = probe.getContext('2d', { willReadFrequently: true });
    pctx.drawImage(els.canvas, 0, 0, 50, 50);
    const alpha = pctx.getImageData(0, 0, 50, 50).data;
    for (let i = 3; i < alpha.length; i += 4) {
      if (alpha[i] > 0) return false;
    }
    return true;
  }

  /**
   * Composite the transparent drawing onto an opaque black 400×400 canvas.
   * @returns {{dataUrl: string, inkPixels: number}}
   */
  function exportDrawing() {
    const size = CONFIG.LOGICAL_SIZE;
    if (!exportCanvas) {
      exportCanvas = document.createElement('canvas');
      exportCanvas.width = size;
      exportCanvas.height = size;
    }
    const ectx = exportCanvas.getContext('2d', { willReadFrequently: true });
    ectx.globalCompositeOperation = 'source-over';
    ectx.clearRect(0, 0, size, size);
    ectx.imageSmoothingEnabled = true;
    ectx.imageSmoothingQuality = 'high';
    ectx.drawImage(els.canvas, 0, 0, size, size);

    // Count inked pixels on the transparent layer (catches drawings that were fully erased).
    const alpha = ectx.getImageData(0, 0, size, size).data;
    let inkPixels = 0;
    for (let i = 3; i < alpha.length; i += 4) {
      if (alpha[i] > CONFIG.INK_ALPHA_THRESHOLD) inkPixels += 1;
    }

    ectx.globalCompositeOperation = 'destination-over';
    ectx.fillStyle = '#000000';
    ectx.fillRect(0, 0, size, size);
    ectx.globalCompositeOperation = 'source-over';

    return { dataUrl: exportCanvas.toDataURL('image/png'), inkPixels };
  }

  /* ======================================================================
   * Mode selection
   * ==================================================================== */

  function setMode(mode, { focus = false, persist = true, announceIt = true } = {}) {
    if (!MODES[mode]) return;
    const changed = mode !== state.mode;
    state.mode = mode;

    els.modeOptions.forEach((option) => {
      const selected = option.dataset.mode === mode;
      option.setAttribute('aria-checked', String(selected));
      option.tabIndex = selected ? 0 : -1;
      if (selected && focus) option.focus();
    });

    const info = currentModeInfo();
    els.drawPrompt.textContent = info.prompt;
    els.emptyText.textContent = info.emptyText;
    els.emptyGlyph.textContent = info.emptyGlyph;
    els.resultCard.classList.toggle('is-characters', mode === 'characters');

    if (changed) {
      invalidateResult(); // keep the drawing, drop the stale prediction
      if (persist) storageSet('mode', mode);
      if (announceIt) announce('Mode: ' + (mode === 'digits' ? 'digit recognition, MNIST.' : 'character recognition, EMNIST.'));
    }
    renderBackendState();
  }

  /** Arrow-key navigation inside the radiogroup (roving tabindex). */
  function onModeKeyDown(event) {
    const keys = ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'];
    if (!keys.includes(event.key)) return;
    event.preventDefault();
    const order = els.modeOptions.map((option) => option.dataset.mode);
    let index = order.indexOf(state.mode);
    if (event.key === 'Home') index = 0;
    else if (event.key === 'End') index = order.length - 1;
    else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') index = (index - 1 + order.length) % order.length;
    else index = (index + 1) % order.length;
    setMode(order[index], { focus: true });
  }

  /* ======================================================================
   * Rendering: backend status, notice, model information
   * ==================================================================== */

  function renderBackendState() {
    // Showing/hiding the server notice above the canvas would move the canvas under a stroke in
    // progress (and bend the stroke), so wait until the stroke has ended.
    if (state.activePointerId !== null) {
      state.backendRenderPending = true;
      return;
    }
    state.backendRenderPending = false;
    renderStatusPill();
    renderModeBadges();
    renderModelChip();
    renderNotice();
    renderModelInfo();
    updatePredictButton();
  }

  function renderStatusPill() {
    const labels = { checking: 'Checking…', online: 'Online', degraded: 'Degraded', offline: 'Offline' };
    const titles = {
      checking: 'Contacting the backend…',
      online: 'Backend online — all models loaded',
      degraded: 'Backend online, but at least one model is not loaded',
      offline: 'Backend unreachable — retrying every 15 seconds',
    };
    els.statusPill.dataset.state = state.backend;
    els.statusPill.title = titles[state.backend];
    els.statusText.textContent = labels[state.backend];
  }

  function renderModeBadges() {
    const models = state.health && state.health.models;
    els.modeBadges.forEach((badge) => {
      const info = models && models[badge.dataset.badgeFor];
      if (!info || info.status === 'loaded') {
        badge.hidden = true;
        return;
      }
      badge.hidden = false;
      badge.dataset.status = info.status === 'error' ? 'error' : 'missing';
      badge.textContent = info.status === 'error' ? 'Failed to load' : 'Not trained yet';
    });
  }

  function renderModelChip() {
    const info = currentModelHealth();
    const fallback = currentModeInfo();
    const name = info && typeof info.display_name === 'string' && info.display_name
      ? info.display_name + ' CNN'
      : fallback.modelName;
    els.modelChipName.textContent = name;
    els.modelChip.dataset.status = info ? String(info.status || 'unknown') : 'unknown';
    els.modelChip.title = 'Selected model: ' + name + (info ? ' (' + (info.status || 'unknown') + ')' : '');
  }

  /** Banner for offline backend or an unavailable model (with the command to fix it). */
  function renderNotice() {
    const mode = currentModeInfo();
    const info = currentModelHealth();

    if (state.backend === 'offline') {
      showNotice({
        tone: 'danger',
        title: 'Recognition server offline',
        text: 'The page could not reach /api/health. Start the backend from the project folder; '
          + 'this page reconnects automatically (checked every 15 s).',
        command: 'python app.py',
      });
      return;
    }

    if (state.health && !info) {
      showNotice({
        tone: 'warning',
        title: mode.datasetName + ' model not reported by the server',
        text: 'The backend did not return information for this mode. Train the model, then restart the server.',
        command: mode.trainCommand,
      });
      return;
    }

    if (info && info.status !== 'loaded') {
      const failed = info.status === 'error';
      showNotice({
        tone: failed ? 'danger' : 'warning',
        title: mode.datasetName + (failed ? ' model failed to load' : ' model not available'),
        text: (typeof info.error === 'string' && info.error)
          || (failed ? 'The checkpoint exists but could not be loaded.' : 'No trained checkpoint was found.'),
        command: mode.trainCommand,
      });
      return;
    }

    els.notice.hidden = true;
  }

  function showNotice({ tone, title, text, command }) {
    els.notice.dataset.tone = tone;
    els.noticeTitle.textContent = title;
    els.noticeText.textContent = text;
    els.noticeCommand.hidden = !command;
    els.noticeCommandText.textContent = command || '';
    els.copyLabel.textContent = 'Copy';
    els.notice.hidden = false;
  }

  /** Fill a <dd> with plain text, or a muted placeholder when the value is missing. */
  function setInfoText(dd, value, placeholder = '—') {
    dd.replaceChildren();
    if (value === null || value === undefined || value === '') {
      dd.appendChild(h('span', 'muted', placeholder));
    } else {
      dd.textContent = String(value);
    }
  }

  function renderModelInfo() {
    const mode = currentModeInfo();
    const info = currentModelHealth();
    const health = state.health;

    // Status badge
    const status = info ? String(info.status || 'unknown') : 'unknown';
    const statusLabels = { loaded: 'Loaded', missing: 'Not trained', error: 'Load error', unknown: 'Unknown' };
    els.infoStatus.dataset.status = statusLabels[status] ? status : 'unknown';
    els.infoStatus.textContent = statusLabels[status] || status;

    // Dataset + classes: static facts are fine to show even when offline.
    setInfoText(els.infoDataset, (info && info.display_name) || mode.datasetName);
    setInfoText(els.infoClasses, isFiniteNumber(info && info.num_classes) ? info.num_classes : mode.numClasses);

    renderTestAccuracy(info);

    // Validation accuracy (held-out part of the TRAIN split) — only if the checkpoint recorded it.
    const valAcc = info ? formatAccuracy(info.val_accuracy) : null;
    els.infoValAcc.replaceChildren();
    if (valAcc) {
      els.infoValAcc.appendChild(h('span', null, valAcc));
      const epoch = isFiniteNumber(info.best_epoch) ? ' · best epoch ' + info.best_epoch : '';
      els.infoValAcc.appendChild(h('span', 'subtle', 'held-out train subset' + epoch));
    } else {
      els.infoValAcc.appendChild(h('span', 'muted', '—'));
    }

    setInfoText(els.infoParams, info ? formatInteger(info.parameters) : null);
    setInfoText(els.infoArch, info && typeof info.architecture === 'string' ? info.architecture : null);

    const device = health && (health.device_name || health.device);
    renderDevice(els.infoDevice, device, state.backend === 'offline' ? 'Unknown (offline)' : '—');
    renderDevice(els.infoTrained, info && info.trained_on, '—');

    // Checkpoint path + creation date
    els.infoCheckpoint.replaceChildren();
    if (info && info.checkpoint) {
      els.infoCheckpoint.appendChild(h('span', null, info.checkpoint));
      const created = formatDate(info.created_at);
      if (created) els.infoCheckpoint.appendChild(h('span', 'subtle', 'created ' + created));
    } else {
      els.infoCheckpoint.appendChild(h('span', 'muted', '—'));
    }
  }

  /**
   * Show a device string such as "cuda (NVIDIA GeForce RTX 3050 Laptop GPU)" as the
   * readable name with the backend ("CUDA") underneath; plain strings are shown as-is.
   */
  function renderDevice(dd, value, placeholder) {
    if (typeof value !== 'string' || !value) {
      setInfoText(dd, null, placeholder);
      return;
    }
    const match = /^([^()]+?)\s*\((.+)\)\s*$/.exec(value);
    dd.replaceChildren();
    if (match) {
      dd.appendChild(h('span', null, match[2]));
      dd.appendChild(h('span', 'subtle', match[1].toUpperCase()));
    } else {
      dd.textContent = /^(cpu|cuda|mps)(:\d+)?$/i.test(value) ? value.toUpperCase() : value;
    }
  }

  /** Test accuracy is shown ONLY when the checkpoint carries a measured value. */
  function renderTestAccuracy(info) {
    const dd = els.infoTestAcc;
    dd.replaceChildren();
    const metrics = info && info.test_metrics && typeof info.test_metrics === 'object' ? info.test_metrics : null;
    const accuracy = metrics ? formatAccuracy(metrics.accuracy) : null;

    if (accuracy) {
      dd.appendChild(h('span', 'metric', accuracy));
      const details = [];
      const samples = formatInteger(metrics.num_samples);
      details.push('official test split' + (samples ? ' · ' + samples + ' images' : ''));
      const macroF1 = isFiniteNumber(metrics.macro_f1) ? (clamp(metrics.macro_f1, 0, 1)).toFixed(3) : null;
      if (macroF1) details.push('macro F1 ' + macroF1);
      dd.appendChild(h('span', 'subtle', details.join(' · ')));
      return;
    }

    if (info && info.status === 'loaded') {
      dd.appendChild(h('span', 'muted', 'Not evaluated yet — run '));
      dd.appendChild(h('code', null, 'python evaluate.py'));
    } else if (info) {
      dd.appendChild(h('span', 'muted', 'Not available — train the model first'));
    } else {
      dd.appendChild(h('span', 'muted', '—'));
    }
  }

  /** Enable/disable Predict and explain why it is disabled. */
  function updatePredictButton() {
    const offline = state.backend === 'offline';
    const unavailable = isModelUnavailable();
    // While a request is in flight the button stays focusable (aria-disabled instead of disabled), so
    // keyboard and screen-reader focus is not dropped to <body>; predict() ignores repeated presses.
    els.predictBtn.disabled = offline || unavailable;
    els.predictBtn.setAttribute('aria-disabled', String(state.predicting || offline || unavailable));
    els.predictBtn.classList.toggle('is-busy', state.predicting);
    els.predictBtn.setAttribute('aria-busy', String(state.predicting));
    els.predictLabel.textContent = state.predicting ? 'Recognizing…' : 'Predict';

    let hint = '';
    if (!state.predicting) {
      if (offline) hint = 'Predict is disabled while the server is offline.';
      else if (unavailable) hint = 'Predict is disabled: the ' + currentModeInfo().datasetName + ' model is not loaded.';
    }
    els.predictHint.textContent = hint;
    els.predictHint.hidden = !hint;
  }

  /* ======================================================================
   * Prediction
   * ==================================================================== */

  /** Abort any in-flight request and return the result panel to its empty state. */
  function invalidateResult() {
    if (state.predictController) {
      state.predictController.abort();
      state.predictController = null;
    }
    const showingSomething = state.predicting || state.result !== null || els.resultEmpty.hidden;
    state.predicting = false;
    state.result = null;
    if (showingSomething) showResultState('empty');
    updatePredictButton();
  }

  /** Switch the result card between its four views. */
  function showResultState(view) {
    els.resultEmpty.hidden = view !== 'empty';
    els.resultLoading.hidden = view !== 'loading';
    els.resultMessage.hidden = view !== 'message';
    els.resultContent.hidden = view !== 'result';
    els.resultMeta.hidden = view !== 'result';
    els.resultCard.setAttribute('aria-busy', String(view === 'loading'));
    if (view !== 'message') state.messageKind = null;
  }

  /** Show a friendly/error message in the result card. */
  function showMessage({ tone = 'danger', title, text, command, kind = null }) {
    els.resultMessage.dataset.tone = tone;
    els.messageTitle.textContent = title;
    els.messageText.textContent = text;
    els.messageCommand.hidden = !command;
    els.messageCommandText.textContent = command || '';
    els.messageIcon.setAttribute('href', tone === 'info' ? '#icon-info' : '#icon-alert');
    showResultState('message');
    state.messageKind = kind;
    announce(title + '. ' + text);
  }

  async function predict() {
    if (state.predicting || state.activePointerId !== null) return;

    const mode = currentModeInfo();
    if (state.backend === 'offline') {
      showMessage({
        tone: 'danger',
        title: 'Server offline',
        text: 'Start the backend and wait for the status to turn online.',
        command: 'python app.py',
        kind: 'connection',
      });
      return;
    }
    if (isModelUnavailable()) {
      const info = currentModelHealth();
      showMessage({
        tone: 'warning',
        title: mode.datasetName + ' model not available',
        text: (info && info.error) || 'Train the model first, then restart the server.',
        command: mode.trainCommand,
      });
      return;
    }

    if (!hasInk()) {
      showMessage({ tone: 'info', title: 'The canvas is empty', text: 'Draw a ' + (state.mode === 'digits' ? 'digit' : 'character') + ' first, then press Predict.' });
      return;
    }
    const drawing = exportDrawing();
    if (drawing.inkPixels < CONFIG.MIN_INK_PIXELS) {
      showMessage(drawing.inkPixels === 0
        ? { tone: 'info', title: 'The canvas looks empty', text: 'Everything seems to be erased. Draw something, then press Predict.' }
        : { tone: 'info', title: 'Too little ink to recognize', text: 'Draw the character larger or use a bigger brush, then press Predict.' });
      return;
    }

    const controller = new AbortController();
    const requestMode = state.mode;
    const startedAt = performance.now();
    state.predictController = controller;
    state.predicting = true;
    showResultState('loading');
    updatePredictButton();

    try {
      const { response, data } = await fetchJson('/api/predict', {
        method: 'POST',
        body: JSON.stringify({ image: drawing.dataUrl, mode: requestMode }),
        timeoutMs: CONFIG.PREDICT_TIMEOUT_MS,
        signal: controller.signal,
      });
      if (controller.signal.aborted) return; // superseded by a new stroke / clear / mode switch

      if (!response.ok || !data || data.success !== true) {
        handleApiError(response, data);
        return;
      }
      renderResult(data, performance.now() - startedAt);
    } catch (err) {
      if (controller.signal.aborted || (err && err.kind === 'aborted')) return;
      const kind = err && err.kind ? err.kind : 'network';
      if (kind === 'network') {
        state.backend = 'offline';
        state.health = null;
        renderBackendState();
        scheduleHealthCheck(3000);
      }
      showMessage({
        tone: 'danger',
        title: ERROR_TITLES[kind] || 'Prediction failed',
        text: (err && err.message) || 'Something went wrong. Please try again.',
        command: kind === 'network' ? 'python app.py' : null,
        kind: kind === 'network' ? 'connection' : null,
      });
    } finally {
      if (state.predictController === controller) {
        state.predictController = null;
        state.predicting = false;
        updatePredictButton();
      }
    }
  }

  /** Render {success:false, error, code} (or a non-JSON failure) in the result card. */
  function handleApiError(response, data) {
    const code = data && typeof data.code === 'string' ? data.code : null;
    const serverMessage = data && typeof data.error === 'string' && data.error ? data.error : null;
    const text = serverMessage || 'The server returned an error (HTTP ' + response.status + '). Please try again.';
    const isBlank = code === 'blank_canvas';
    showMessage({
      tone: isBlank ? 'info' : (code === 'model_unavailable' ? 'warning' : 'danger'),
      title: ERROR_TITLES[code] || 'Prediction failed',
      text,
      command: code === 'model_unavailable' ? currentModeInfo().trainCommand : null,
    });
    if (code === 'model_unavailable') checkHealth();
  }

  /**
   * Normalise the probabilities array defensively.
   * @returns {Array<{classIndex: number, label: string, probability: number}>}
   */
  function normalizeProbabilities(list) {
    if (!Array.isArray(list)) return [];
    return list
      .filter((item) => item && typeof item === 'object')
      .map((item, i) => ({
        classIndex: Number.isInteger(item.class_index) ? item.class_index : i,
        label: item.label === undefined || item.label === null ? String(i) : String(item.label),
        probability: isFiniteNumber(item.probability) ? clamp(item.probability, 0, 1) : 0,
      }));
  }

  /** Turn an API response into a render-friendly result object. */
  function buildResult(data, roundTripMs) {
    let probabilities = normalizeProbabilities(data.probabilities).sort((a, b) => a.classIndex - b.classIndex);
    let topK = normalizeProbabilities(data.top_k).sort((a, b) => b.probability - a.probability);
    if (!probabilities.length) probabilities = topK.slice().sort((a, b) => a.classIndex - b.classIndex);
    if (!topK.length) topK = probabilities.slice().sort((a, b) => b.probability - a.probability);
    topK = topK.slice(0, CONFIG.TOP_K);

    const best = topK[0] || null;
    const label = data.prediction !== undefined && data.prediction !== null
      ? String(data.prediction)
      : (best ? best.label : '?');
    const classIndex = Number.isInteger(data.class_index) ? data.class_index : (best ? best.classIndex : null);
    const confidence = isFiniteNumber(data.confidence) ? clamp(data.confidence, 0, 1) : (best ? best.probability : null);
    const category = typeof data.category === 'string' && data.category ? data.category : categoryFromLabel(label);

    return {
      mode: typeof data.mode === 'string' ? data.mode : state.mode,
      label,
      classIndex,
      confidence,
      category,
      probabilities,
      topK,
      preview: typeof data.preprocessed_image === 'string' && data.preprocessed_image.startsWith('data:image/')
        ? data.preprocessed_image
        : null,
      processingMs: isFiniteNumber(data.processing_time_ms) ? data.processing_time_ms : null,
      roundTripMs,
      device: typeof data.device === 'string' ? data.device : null,
    };
  }

  function confidenceLevel(p) {
    if (!isFiniteNumber(p)) return { level: 'low', text: 'Unknown' };
    if (p >= 0.9) return { level: 'high', text: 'High' };
    if (p >= 0.6) return { level: 'moderate', text: 'Moderate' };
    return { level: 'low', text: 'Low' };
  }

  function renderResult(data, roundTripMs) {
    const result = buildResult(data, roundTripMs);
    state.result = result;

    // Hero: glyph, category, confidence
    els.resultGlyph.textContent = result.label;
    const categoryName = CATEGORY_NAMES[result.category] || 'Prediction';
    els.resultCategory.textContent = categoryName;

    const classBits = [];
    if (result.classIndex !== null) classBits.push('Class index ' + result.classIndex);
    if (result.label.length === 1) classBits.push('ASCII ' + result.label.charCodeAt(0));
    els.resultClass.textContent = classBits.join(' · ');
    els.resultClass.hidden = classBits.length === 0;

    const level = confidenceLevel(result.confidence);
    els.confidenceValue.textContent = formatPercent(result.confidence);
    els.confidenceLevel.dataset.level = level.level;
    els.confidenceLevel.textContent = level.text;
    els.confidenceFill.style.width = (isFiniteNumber(result.confidence) ? result.confidence * 100 : 0).toFixed(1) + '%';

    // Preprocessed 28×28 preview
    if (result.preview) {
      els.previewImg.src = result.preview;
      els.preview.hidden = false;
    } else {
      els.previewImg.removeAttribute('src');
      els.preview.hidden = true;
    }

    renderAmbiguityNote(result);
    renderTopK(result);
    els.listControls.hidden = state.mode !== 'characters';
    renderProbabilityList();

    // Meta line: server time · round trip · device
    const meta = [];
    if (result.processingMs !== null) meta.push('Server ' + result.processingMs.toFixed(1) + ' ms');
    meta.push('round trip ' + Math.round(result.roundTripMs) + ' ms');
    if (result.device) meta.push(result.device);
    els.resultMeta.textContent = meta.join(' · ');

    showResultState('result');
    els.probScroll.scrollTop = 0;

    const second = result.topK[1];
    announce(
      'Prediction: ' + result.label + ', ' + categoryName.toLowerCase() + '. Model confidence '
      + formatPercent(result.confidence) + '.'
      + (second ? ' Next most likely: ' + second.label + ' at ' + formatPercent(second.probability) + '.' : ''),
    );
  }

  /** Educational hint for classes EMNIST ByClass cannot reliably tell apart. */
  function renderAmbiguityNote(result) {
    const label = result.label;
    let text = '';
    if (result.mode === 'characters' && label.length === 1) {
      if (CASE_AMBIGUOUS.has(label)) {
        text = 'Handwritten “' + label.toLowerCase() + '” and “' + label.toUpperCase()
          + '” look almost the same; EMNIST ByClass models often mix up their case. Check the alternatives below.';
      } else {
        const group = LOOKALIKE_GROUPS.find((g) => g.includes(label));
        if (group) {
          text = 'Look-alike characters (' + group.join(' · ') + ') are easy to confuse in handwriting. '
            + 'Check the alternatives below.';
        }
      }
    }
    els.ambiguityText.textContent = text;
    els.ambiguityNote.hidden = !text;
  }

  function renderTopK(result) {
    const items = result.topK.map((item, rank) => {
      const li = h('li', 'topk-item' + (rank === 0 ? ' is-top' : ''));
      li.appendChild(h('span', 'topk-rank', '#' + (rank + 1)));
      li.appendChild(h('span', 'topk-label', item.label));
      li.appendChild(h('span', 'topk-value', formatPercent(item.probability)));
      const bar = h('span', 'topk-bar');
      const fill = h('span');
      fill.style.width = (item.probability * 100).toFixed(1) + '%';
      bar.appendChild(fill);
      li.appendChild(bar);
      li.setAttribute('aria-label', 'Rank ' + (rank + 1) + ': ' + item.label + ', ' + formatPercent(item.probability));
      return li;
    });
    els.topkList.replaceChildren(...items);
  }

  /** Does a probability item match the search query? (label, or category word with >= 3 chars) */
  function matchesQuery(item, query) {
    const q = query.toLowerCase();
    if (item.label.toLowerCase().includes(q)) return true;
    if (q.length >= 3) {
      const category = categoryFromLabel(item.label);
      const name = category ? CATEGORY_NAMES[category].toLowerCase() : '';
      return name.startsWith(q) || (category || '').startsWith(q);
    }
    return false;
  }

  function createProbRow(item, predictedIndex) {
    const isTop = item.classIndex === predictedIndex;
    const li = h('li', 'prob-row' + (isTop ? ' is-top' : ''));
    li.appendChild(h('span', 'prob-label', item.label));
    const track = h('span', 'prob-track');
    track.setAttribute('aria-hidden', 'true');
    const fill = h('span', 'prob-fill');
    fill.style.width = (item.probability * 100).toFixed(2) + '%';
    track.appendChild(fill);
    li.appendChild(track);
    li.appendChild(h('span', 'prob-value', formatPercent(item.probability)));
    if (isTop) li.appendChild(h('span', 'visually-hidden', '(predicted)'));
    return li;
  }

  /** All-class list: class-index order; characters mode adds search and "Top 5 only". */
  function renderProbabilityList() {
    const result = state.result;
    if (!result) return;
    const characters = state.mode === 'characters';
    let items = characters && state.topOnly ? result.topK : result.probabilities;
    const query = characters ? state.filter.trim() : '';
    if (query) items = items.filter((item) => matchesQuery(item, query));

    const predictedIndex = result.classIndex !== null ? result.classIndex : (result.topK[0] && result.topK[0].classIndex);
    els.probList.replaceChildren(...items.map((item) => createProbRow(item, predictedIndex)));
    els.probList.classList.toggle('is-multicol', items.length > 12);

    els.probEmpty.hidden = items.length > 0;
    els.probEmpty.textContent = items.length ? '' : 'No class matches “' + query + '”.';
    els.probsHeading.textContent = characters && state.topOnly ? 'Top 5 class probabilities' : 'All class probabilities';
  }

  /* ======================================================================
   * Keyboard shortcuts
   * ==================================================================== */

  function isTextEntry(target) {
    if (!target || !(target instanceof Element)) return false;
    if (target.isContentEditable) return true;
    if (target.tagName === 'TEXTAREA' || target.tagName === 'SELECT') return true;
    if (target.tagName === 'INPUT') {
      const type = (target.getAttribute('type') || 'text').toLowerCase();
      return !['range', 'checkbox', 'radio', 'button', 'submit', 'reset'].includes(type);
    }
    return false;
  }

  function onKeyDown(event) {
    if (event.defaultPrevented || event.isComposing) return;
    const target = event.target;
    if (isTextEntry(target)) return;

    const key = event.key;
    const ctrlOrMeta = event.ctrlKey || event.metaKey;

    if (ctrlOrMeta && !event.altKey && !event.shiftKey && (key === 'z' || key === 'Z')) {
      event.preventDefault();
      undo();
      return;
    }
    if (ctrlOrMeta || event.altKey) return;

    // Holding a key must not toggle the eraser back and forth or fire repeated predictions/clears.
    if (event.repeat && ['Enter', 'Escape', 'Delete', 'e', 'E'].includes(key)) return;

    // Let Enter/Space activate focused buttons, links and toggles natively.
    const interactive = target instanceof Element && target.closest('button, a, [role="radio"], label, summary');

    switch (key) {
      case 'Enter':
        if (interactive) return;
        event.preventDefault();
        predict();
        break;
      case 'Escape':
      case 'Delete':
        event.preventDefault();
        clearCanvas();
        break;
      case 'e':
      case 'E':
        event.preventDefault();
        setEraser(!state.eraser);
        announce(state.eraser ? 'Eraser on.' : 'Eraser off. Brush on.');
        break;
      case '[':
        event.preventDefault();
        setBrushSize(state.brushSize - CONFIG.BRUSH_STEP);
        announce('Brush ' + state.brushSize + ' pixels.');
        break;
      case ']':
        event.preventDefault();
        setBrushSize(state.brushSize + CONFIG.BRUSH_STEP);
        announce('Brush ' + state.brushSize + ' pixels.');
        break;
      default:
        break;
    }
  }

  /* ======================================================================
   * Misc UI handlers
   * ==================================================================== */

  async function copyCommand() {
    const text = els.noticeCommandText.textContent;
    let copied = false;
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        copied = true;
      }
    } catch (_err) {
      copied = false;
    }
    if (!copied) {
      // Fallback: select the text so the user can copy it manually.
      const range = document.createRange();
      range.selectNodeContents(els.noticeCommandText);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
    }
    els.copyLabel.textContent = copied ? 'Copied' : 'Selected';
    announce(copied ? 'Command copied to clipboard.' : 'Command selected; press Ctrl+C to copy.');
    window.setTimeout(() => {
      els.copyLabel.textContent = 'Copy';
    }, 1800);
  }

  /* ======================================================================
   * Initialisation
   * ==================================================================== */

  function cacheElements() {
    const byId = (id) => {
      const node = document.getElementById(id);
      if (!node) throw new Error('Missing element #' + id);
      return node;
    };
    Object.assign(els, {
      canvas: byId('draw-canvas'),
      canvasWrap: byId('canvas-wrap'),
      placeholder: byId('canvas-placeholder'),
      brushCursor: byId('brush-cursor'),
      brushInput: byId('brush-size'),
      brushValue: byId('brush-size-value'),
      eraserBtn: byId('eraser-btn'),
      undoBtn: byId('undo-btn'),
      clearBtn: byId('clear-btn'),
      predictBtn: byId('predict-btn'),
      predictLabel: byId('predict-label'),
      predictHint: byId('predict-hint'),
      drawPrompt: byId('draw-prompt'),

      modeSwitch: byId('mode-switch'),
      modeOptions: Array.from(document.querySelectorAll('.mode-option[data-mode]')),
      modeBadges: Array.from(document.querySelectorAll('.mode-badge[data-badge-for]')),

      statusPill: byId('status-pill'),
      statusText: byId('status-text'),
      modelChip: byId('model-chip'),
      modelChipName: byId('model-chip-name'),

      notice: byId('notice'),
      noticeTitle: byId('notice-title'),
      noticeText: byId('notice-text'),
      noticeCommand: byId('notice-command'),
      noticeCommandText: byId('notice-command-text'),
      copyBtn: byId('copy-command'),
      copyLabel: byId('copy-command-label'),
      retryBtn: byId('retry-health'),

      resultCard: byId('result-card'),
      resultMeta: byId('result-meta'),
      resultEmpty: byId('result-empty'),
      emptyText: byId('empty-text'),
      emptyGlyph: byId('empty-glyph'),
      resultLoading: byId('result-loading'),
      resultMessage: byId('result-message'),
      messageTitle: byId('message-title'),
      messageText: byId('message-text'),
      messageCommand: byId('message-command'),
      messageCommandText: byId('message-command-text'),
      messageIcon: document.querySelector('#result-message .message-icon use'),
      resultContent: byId('result-content'),
      resultGlyph: byId('result-glyph'),
      resultCategory: byId('result-category'),
      resultClass: byId('result-class'),
      confidenceValue: byId('confidence-value'),
      confidenceLevel: byId('confidence-level'),
      confidenceFill: byId('confidence-fill'),
      preview: byId('preview'),
      previewImg: byId('preview-img'),
      ambiguityNote: byId('ambiguity-note'),
      ambiguityText: byId('ambiguity-text'),
      topkList: byId('topk-list'),
      listControls: byId('list-controls'),
      classSearch: byId('class-search'),
      topOnly: byId('top-only'),
      probsHeading: byId('probs-heading'),
      probScroll: byId('prob-scroll'),
      probList: byId('prob-list'),
      probEmpty: byId('prob-empty'),

      infoStatus: byId('info-status'),
      infoDataset: byId('info-dataset'),
      infoClasses: byId('info-classes'),
      infoTestAcc: byId('info-test-acc'),
      infoValAcc: byId('info-val-acc'),
      infoParams: byId('info-params'),
      infoArch: byId('info-arch'),
      infoDevice: byId('info-device'),
      infoTrained: byId('info-trained'),
      infoCheckpoint: byId('info-checkpoint'),

      announcer: byId('announcer'),
    });
  }

  function bindCanvasEvents() {
    const canvas = els.canvas;
    canvas.addEventListener('pointerdown', onPointerDown);
    canvas.addEventListener('pointermove', onPointerMove);
    canvas.addEventListener('pointerup', onPointerEnd);
    canvas.addEventListener('pointercancel', onPointerEnd);
    canvas.addEventListener('lostpointercapture', onPointerEnd);
    canvas.addEventListener('pointerleave', (event) => {
      if (event.pointerId !== state.activePointerId) hideBrushCursor();
    });
    canvas.addEventListener('contextmenu', (event) => event.preventDefault());
    // Belt and braces for browsers that ignore touch-action: never scroll/zoom while drawing.
    const stopTouch = (event) => {
      if (event.cancelable) event.preventDefault();
    };
    canvas.addEventListener('touchstart', stopTouch, { passive: false });
    canvas.addEventListener('touchmove', stopTouch, { passive: false });
    canvas.addEventListener('dragstart', (event) => event.preventDefault());
  }

  function bindControls() {
    els.brushInput.addEventListener('input', () => setBrushSize(els.brushInput.value));
    els.eraserBtn.addEventListener('click', () => setEraser(!state.eraser));
    els.undoBtn.addEventListener('click', undo);
    els.clearBtn.addEventListener('click', () => clearCanvas());
    els.predictBtn.addEventListener('click', predict);

    els.modeOptions.forEach((option) => {
      option.addEventListener('click', () => setMode(option.dataset.mode));
    });
    els.modeSwitch.addEventListener('keydown', onModeKeyDown);

    els.classSearch.addEventListener('input', () => {
      state.filter = els.classSearch.value;
      renderProbabilityList();
    });
    els.topOnly.addEventListener('change', () => {
      state.topOnly = els.topOnly.checked;
      renderProbabilityList();
    });

    els.copyBtn.addEventListener('click', copyCommand);
    els.retryBtn.addEventListener('click', () => {
      state.backend = state.health ? state.backend : 'checking';
      renderStatusPill();
      checkHealth();
    });

    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') checkHealth();
    });
    window.addEventListener('online', () => checkHealth());
  }

  function restorePreferences() {
    const savedMode = storageGet('mode');
    const savedBrush = Number(storageGet('brush'));
    setBrushSize(Number.isFinite(savedBrush) && savedBrush > 0 ? savedBrush : CONFIG.BRUSH_DEFAULT, { persist: false });
    setMode(savedMode && MODES[savedMode] ? savedMode : 'digits', { persist: false, announceIt: false });
  }

  function init() {
    cacheElements();
    ctx = els.canvas.getContext('2d');
    if (!ctx) {
      els.drawPrompt.textContent = 'Your browser does not support the canvas element.';
      return;
    }
    setupBackingStore();
    watchDevicePixelRatio();
    bindCanvasEvents();
    bindControls();
    restorePreferences();
    setEraser(false);
    updateCanvasChrome();
    showResultState('empty');
    renderBackendState();
    checkHealth();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();
