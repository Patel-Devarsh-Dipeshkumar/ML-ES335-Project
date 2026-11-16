/* ==========================================================================
   Interaction 1: Model Architecture & Mathematical Mechanism Lab
   Interactive 2D decision fields, pointer-draggable eigenvectors,
   ensemble stair-step boundaries, animated deep residual highways,
   and synthetic H&E spatial pooling microscopy for all 11 architectures.
   ========================================================================== */
(function () {
  'use strict';

  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };

  /* ------------------------------------------------------------------ */
  /* KaTeX Safe Renderer with Plain-Text Fallback                        */
  /* ------------------------------------------------------------------ */
  function renderLatex(container) {
    if (!container) return;
    $$('.tex', container).forEach(function (el) {
      var src = el.dataset.tex || el.textContent;
      el.dataset.tex = src;
      if (window.katex) {
        try {
          window.katex.render(src, el, {
            displayMode: el.classList.contains('display'),
            throwOnError: false
          });
          return;
        } catch (e) { /* fall through to text */ }
      }
      el.textContent = src;
    });
  }

  /* ------------------------------------------------------------------ */
  /* Explicit Canvas 2D Theme Palette (Zero CSS Variable Leakage)        */
  /* ------------------------------------------------------------------ */
  function getTheme() {
    var isDark = document.documentElement.dataset.theme === 'dark';
    return {
      isDark: isDark,
      bg: isDark ? '#0e1320' : '#ffffff',
      surface: isDark ? '#151b2b' : '#f8fafc',
      surface2: isDark ? '#1c2438' : '#f1f5f9',
      text: isDark ? '#e6eaf2' : '#0f172a',
      muted: isDark ? '#aeb8cc' : '#475569',
      faint: isDark ? '#64748b' : '#94a3b8',
      axis: isDark ? 'rgba(226, 232, 240, 0.22)' : 'rgba(15, 23, 42, 0.20)',
      grid: isDark ? 'rgba(226, 232, 240, 0.08)' : 'rgba(15, 23, 42, 0.07)',
      corridor: isDark ? 'rgba(169, 154, 232, 0.12)' : 'rgba(76, 58, 142, 0.08)',
      accent: isDark ? '#a99ae8' : '#4c3a8e',
      accentStrong: isDark ? '#c6bbf3' : '#3b2c73',
      tumor: isDark ? '#f43f5e' : '#e11d48',
      tumorSoft: isDark ? 'rgba(244, 63, 94, 0.24)' : 'rgba(225, 29, 72, 0.16)',
      normal: isDark ? '#38bdf8' : '#0284c7',
      normalSoft: isDark ? 'rgba(56, 189, 248, 0.24)' : 'rgba(2, 132, 199, 0.16)',
      gold: isDark ? '#facc15' : '#d97706',
      emerald: isDark ? '#34d399' : '#059669',
      purple: isDark ? '#c084fc' : '#7c3aed',
      boundary: isDark ? '#f8fafc' : '#0f172a'
    };
  }

  /* ------------------------------------------------------------------ */
  /* Deterministic Pseudo-Random Generator & Synthetic Pathology Data    */
  /* ------------------------------------------------------------------ */
  function createRnd(seed) {
    var s = seed % 2147483647;
    if (s <= 0) s += 2147483646;
    return function () {
      s = (s * 16807) % 2147483647;
      return (s - 1) / 2147483646;
    };
  }

  function gaussRnd(rng) {
    var u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  // 1. PCA Point Cloud (56 points aligned along principal axis theta_1 = 22 deg)
  var rndPca = createRnd(42);
  var PCA_POINTS = [];
  var PCA_THETA1 = 22 * Math.PI / 180;
  var PCA_L1 = 2.354, PCA_L2 = 0.370;
  (function () {
    for (var i = 0; i < 56; i++) {
      var u1 = gaussRnd(rndPca) * 58;
      var u2 = gaussRnd(rndPca) * 21;
      var px = u1 * Math.cos(PCA_THETA1) - u2 * Math.sin(PCA_THETA1);
      var py = u1 * Math.sin(PCA_THETA1) + u2 * Math.cos(PCA_THETA1);
      PCA_POINTS.push({
        x: Math.max(-135, Math.min(135, px)),
        y: Math.max(-115, Math.min(115, py))
      });
    }
  })();

  // 2. 2D Feature Space Pathology Cohort (48 patches: 24 Normal, 24 Tumour)
  // Coordinates centered around (0,0) in range [-165..165, -115..115]
  var rndClf = createRnd(1337);
  var CLF_POINTS = [];
  (function () {
    // Normal tissue cluster (bottom-left + upper-left lobe)
    for (var i = 0; i < 24; i++) {
      var lobe = i < 17 ? { cx: -52, cy: 32, sx: 42, sy: 34 } : { cx: -28, cy: -52, sx: 32, sy: 25 };
      var nx = lobe.cx + gaussRnd(rndClf) * lobe.sx;
      var ny = lobe.cy + gaussRnd(rndClf) * lobe.sy;
      CLF_POINTS.push({
        x: Math.max(-185, Math.min(185, nx)),
        y: Math.max(-120, Math.min(120, ny)),
        label: 0
      });
    }
    // Tumour tissue cluster (top-right + lower-right infiltrating nest)
    for (var j = 0; j < 24; j++) {
      var tlobe = j < 17 ? { cx: 54, cy: -28, sx: 42, sy: 35 } : { cx: 32, cy: 52, sx: 34, sy: 26 };
      var tx = tlobe.cx + gaussRnd(rndClf) * tlobe.sx;
      var ty = tlobe.cy + gaussRnd(rndClf) * tlobe.sy;
      CLF_POINTS.push({
        x: Math.max(-185, Math.min(185, tx)),
        y: Math.max(-120, Math.min(120, ty)),
        label: 1
      });
    }
  })();

  // Pre-build deterministic weak decision trees for Random Forest simulation
  var RF_TREES = [];
  (function () {
    var rTree = createRnd(2026);
    for (var t = 0; t < 25; t++) {
      RF_TREES.push({
        ax1: (rTree() - 0.5) * 75,
        ay1: (rTree() - 0.5) * 65,
        ax2: -35 + (rTree() - 0.5) * 55,
        ay2: 28 + (rTree() - 0.5) * 55,
        bx2: 38 + (rTree() - 0.5) * 55,
        by2: -28 + (rTree() - 0.5) * 55,
        cx3: (rTree() - 0.5) * 90,
        cy3: (rTree() - 0.5) * 80,
        flip: rTree() > 0.35 ? 1 : 0,
        colWeight: 0.35 + rTree() * 0.65
      });
    }
  })();

  function evalSingleTree(tree, x, y, depth, subspace) {
    // Root split (Depth 1)
    var useX = tree.flip ? (x > tree.ax1 * subspace) : (y < tree.ay1 * subspace);
    var p = useX ? 0.74 : 0.26;
    if (depth >= 2) {
      if (useX) {
        p = (y < tree.by2) ? 0.88 : ((x > tree.bx2) ? 0.68 : 0.36);
      } else {
        p = (x > tree.ax2 && y < tree.ay2) ? 0.64 : 0.12;
      }
    }
    if (depth >= 3) {
      var boxHit = Math.abs(x - tree.bx2) < 42 && Math.abs(y - tree.by2) < 36;
      if (boxHit) p = Math.min(0.96, p + 0.16);
      var benignPocket = Math.abs(x - tree.ax2) < 38 && Math.abs(y - tree.ay2) < 34;
      if (benignPocket) p = Math.max(0.05, p - 0.18);
    }
    if (depth >= 4) {
      p += 0.08 * Math.sign(Math.sin((x - tree.cx3) * 0.09) * Math.cos((y - tree.cy3) * 0.09));
    }
    if (depth >= 5) {
      p += 0.07 * Math.sign(Math.cos((x + y) * 0.14 + tree.ax1));
    }
    return Math.max(0.02, Math.min(0.98, p));
  }

  function evalForestProb(x, y, numTrees, depth, subspace) {
    var sum = 0;
    for (var b = 0; b < numTrees; b++) {
      sum += evalSingleTree(RF_TREES[b], x, y, depth, subspace);
    }
    return sum / numTrees;
  }

  // Boosting stages (Stage 1 coarse split + 9 residual correction boxes)
  var GBM_STAGES = [
    { x0: -5, y0: -140, x1: 220, y1: 35, val: 1.35, label: 'Coarse nuclear density split' },
    { x0: 5, y0: 15, x1: 115, y1: 98, val: 1.25, label: 'Lower-right tumour nest residual' },
    { x0: -92, y0: -95, x1: 5, y1: -15, val: -1.30, label: 'Suppress reactive follicle FP' },
    { x0: -32, y0: -48, x1: 42, y1: 24, val: 0.95, label: 'Central micrometastasis core' },
    { x0: -145, y0: -15, x1: -15, y1: 110, val: -1.15, label: 'Clear benign sinusoid zone' },
    { x0: 42, y0: -105, x1: 155, y1: -15, val: 1.10, label: 'Reinforce high-grade carcinoma' },
    { x0: -60, y0: 18, x1: 12, y1: 75, val: -0.90, label: 'Prune stromal border noise' },
    { x0: 12, y0: -20, x1: 88, y1: 48, val: 0.85, label: 'Capture infiltrating margin' },
    { x0: -42, y0: -88, x1: 25, y1: -32, val: -0.80, label: 'Dampen germinal centre halo' },
    { x0: 62, y0: 22, x1: 145, y1: 92, val: 0.90, label: 'Final residual pocket refinement' }
  ];

  function evalHistGbmProb(x, y, stages, lr) {
    // Quantize to 256-bin equivalent grid
    var qx = Math.round(x / 6) * 6;
    var qy = Math.round(y / 6) * 6;
    var f = -0.15; // Prior log-odds
    var effEta = lr * 2.2;
    for (var m = 0; m < stages; m++) {
      var st = GBM_STAGES[m];
      if (qx >= st.x0 && qx <= st.x1 && qy >= st.y0 && qy <= st.y1) {
        f += effEta * st.val;
      } else if (m === 0) {
        f -= effEta * 0.85;
      }
    }
    return 1 / (1 + Math.exp(-Math.max(-6, Math.min(6, f))));
  }

  function evalRbfSvmProb(x, y, gamma, C) {
    var pot = 0;
    var scale = 0.00038 * gamma;
    var cBoost = 0.65 + 0.22 * Math.sqrt(C);
    for (var i = 0; i < CLF_POINTS.length; i++) {
      var pt = CLF_POINTS[i];
      var dx = x - pt.x, dy = y - pt.y;
      var kVal = Math.exp(-(dx * dx + dy * dy) * scale);
      pot += (pt.label === 1 ? 1 : -1) * kVal;
    }
    return 1 / (1 + Math.exp(-Math.max(-7, Math.min(7, pot * cBoost))));
  }

  // Pre-generate realistic H&E cells for Model 11 (Spatial Scope)
  var SPATIAL_CELLS = [];
  (function () {
    var rSp = createRnd(909);
    // Benign stromal cells & lymphocytes
    for (var i = 0; i < 95; i++) {
      var bx = (rSp() - 0.5) * 180;
      var by = (rSp() - 0.5) * 180;
      var inCore = Math.abs(bx) < 30 && Math.abs(by) < 30;
      SPATIAL_CELLS.push({
        x: bx,
        y: by,
        r: inCore ? (1.8 + rSp() * 1.4) : (2.0 + rSp() * 1.8),
        type: 'benign'
      });
    }
    // Peripheral margin tumour cells (outside central 32x32 window, i.e., |x|>34 or |y|>34)
    for (var j = 0; j < 42; j++) {
      var angle = rSp() * Math.PI * 2;
      var dist = 46 + rSp() * 44;
      SPATIAL_CELLS.push({
        x: Math.cos(angle) * dist,
        y: Math.sin(angle) * dist,
        r: 3.6 + rSp() * 2.4,
        type: 'margin_tumor'
      });
    }
    // Adipose fat vacuoles
    for (var k = 0; k < 10; k++) {
      var fa = (k / 10) * Math.PI * 2 + 0.2;
      var fd = 62 + rSp() * 24;
      SPATIAL_CELLS.push({
        x: Math.cos(fa) * fd,
        y: Math.sin(fa) * fd,
        r: 6.5 + rSp() * 3.5,
        type: 'vacuole'
      });
    }
  })();

  /* ------------------------------------------------------------------ */
  /* Canvas Drawing Helpers                                              */
  /* ------------------------------------------------------------------ */
  function roundRect(ctx, x, y, w, h, r) {
    var rad = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + rad, y);
    ctx.arcTo(x + w, y, x + w, y + h, rad);
    ctx.arcTo(x + w, y + h, x, y + h, rad);
    ctx.arcTo(x, y + h, x, y, rad);
    ctx.arcTo(x, y, x + w, y, rad);
    ctx.closePath();
  }

  function drawScatterPoints(ctx, cx, cy, th, predFn, threshold) {
    var mis = 0;
    CLF_POINTS.forEach(function (pt) {
      var sx = cx + pt.x, sy = cy + pt.y;
      var prob = predFn ? predFn(pt.x, pt.y) : (pt.label ? 1 : 0);
      var call = prob >= (threshold != null ? threshold : 0.5) ? 1 : 0;
      var isErr = predFn ? (call !== pt.label) : false;
      if (isErr) mis++;

      if (isErr) {
        ctx.save();
        ctx.strokeStyle = th.gold;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(sx, sy, 6.2, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }

      ctx.fillStyle = pt.label === 1 ? th.tumor : th.normal;
      ctx.strokeStyle = th.bg;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(sx, sy, 3.8, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });
    return mis;
  }

  function drawAxesGrid(ctx, w, h, cx, cy, th) {
    ctx.save();
    ctx.strokeStyle = th.grid;
    ctx.lineWidth = 1;
    for (var gx = cx % 40; gx < w; gx += 40) {
      ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke();
    }
    for (var gy = cy % 40; gy < h; gy += 40) {
      ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
    }
    ctx.strokeStyle = th.axis;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(12, cy); ctx.lineTo(w - 12, cy);
    ctx.moveTo(cx, 12); ctx.lineTo(cx, h - 12);
    ctx.stroke();
    ctx.restore();
  }

  function drawProbabilityField(ctx, w, h, cx, cy, th, probFn, tau, step) {
    var s = step || 6;
    var cols = Math.ceil(w / s), rows = Math.ceil(h / s);
    var grid = new Float32Array((cols + 1) * (rows + 1));

    for (var r = 0; r <= rows; r++) {
      var py = r * s - cy;
      for (var c = 0; c <= cols; c++) {
        var px = c * s - cx;
        var p = probFn(px, py);
        grid[r * (cols + 1) + c] = p;

        if (r < rows && c < cols) {
          var delta = p - tau;
          if (delta >= 0) {
            var aPos = Math.min(0.30, 0.04 + Math.abs(delta) * 0.42);
            ctx.fillStyle = th.isDark
              ? 'rgba(244, 63, 94, ' + aPos.toFixed(3) + ')'
              : 'rgba(225, 29, 72, ' + aPos.toFixed(3) + ')';
          } else {
            var aNeg = Math.min(0.30, 0.04 + Math.abs(delta) * 0.42);
            ctx.fillStyle = th.isDark
              ? 'rgba(56, 189, 248, ' + aNeg.toFixed(3) + ')'
              : 'rgba(2, 132, 199, ' + aNeg.toFixed(3) + ')';
          }
          ctx.fillRect(c * s, r * s, s, s);
        }
      }
    }

    // Marching squares contour for P(y=1) = tau
    ctx.save();
    ctx.strokeStyle = th.boundary;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    for (var ri = 0; ri < rows; ri++) {
      for (var ci = 0; ci < cols; ci++) {
        var v00 = grid[ri * (cols + 1) + ci] - tau;
        var v10 = grid[ri * (cols + 1) + ci + 1] - tau;
        var v01 = grid[(ri + 1) * (cols + 1) + ci] - tau;
        var x0 = ci * s, y0 = ri * s;
        if ((v00 >= 0) !== (v10 >= 0)) {
          var tH = v00 / (v00 - v10);
          ctx.moveTo(x0 + tH * s, y0);
          ctx.lineTo(x0 + tH * s, y0 + s);
        }
        if ((v00 >= 0) !== (v01 >= 0)) {
          var tV = v00 / (v00 - v01);
          ctx.moveTo(x0, y0 + tV * s);
          ctx.lineTo(x0 + s, y0 + tV * s);
        }
      }
    }
    ctx.stroke();
    ctx.restore();
  }

  /* ------------------------------------------------------------------ */
  /* Animation Loop Controller (for ResNet-18, EffNet, ResNet-50)        */
  /* ------------------------------------------------------------------ */
  var animFrameId = null;
  var animPhase = 0;

  function isAnimatedModel(key) {
    return key === 'resnet18' || key === 'effnet' || key === 'resnet50';
  }

  function syncAnimationLoop() {
    var sel = $('#exploreModelSelect');
    var key = sel ? sel.value : 'pca';
    if (isAnimatedModel(key)) {
      if (!animFrameId) {
        var tick = function () {
          animPhase = (animPhase + 0.028) % (Math.PI * 20);
          updateExplorer();
          if (sel && isAnimatedModel(sel.value)) {
            animFrameId = requestAnimationFrame(tick);
          } else {
            animFrameId = null;
          }
        };
        animFrameId = requestAnimationFrame(tick);
      }
    } else if (animFrameId) {
      cancelAnimationFrame(animFrameId);
      animFrameId = null;
    }
  }

  /* ------------------------------------------------------------------ */
  /* 11 Model Laboratory Definitions                                     */
  /* ------------------------------------------------------------------ */
  var MODELS_CFG = {
    /* ================================================================
       1. Bilinear Pixel PCA (Eigenvector Decomposition)
       ================================================================ */
    pca: {
      badge: 'Unsupervised Feature Reduction (20 PCs)',
      badgeClass: 'badge-normal',
      state: { angle: 105, dragging: false },
      controls: function () {
        var a = Math.round(MODELS_CFG.pca.state.angle);
        return '<div class="ctrl-row">' +
          '<div class="ctrl-label"><span>Projection Axis Angle &theta; (Drag canvas or slider)</span><span id="expVal1">' + a + '&deg;</span></div>' +
          '<input type="range" id="expSlider1" min="0" max="180" step="1" value="' + a + '">' +
          '</div>' +
          '<div class="explorebtnrow">' +
          '<button type="button" class="btn primary" id="btnSnap1">Snap 1st Eigenvector (&theta;₁ = 22&deg;, &lambda;₁ = 2.354)</button>' +
          '<button type="button" class="btn" id="btnSnap2">Snap 2nd Eigenvector (&theta;₂ = 112&deg;, &lambda;₂ = 0.370)</button>' +
          '</div>';
      },
      init: function (update) {
        var s1 = $('#expSlider1');
        var setAngle = function (deg) {
          var clamped = Math.max(0, Math.min(180, Math.round(deg)));
          MODELS_CFG.pca.state.angle = clamped;
          if (s1) s1.value = clamped;
          var lbl = $('#expVal1');
          if (lbl) lbl.textContent = clamped + '°';
          update();
        };
        if (s1) s1.addEventListener('input', function () { setAngle(parseFloat(s1.value)); });
        var b1 = $('#btnSnap1'), b2 = $('#btnSnap2');
        if (b1) b1.addEventListener('click', function () { setAngle(22); });
        if (b2) b2.addEventListener('click', function () { setAngle(112); });
      },
      render: function (ctx, w, h) {
        var th = getTheme();
        var cx = 165, cy = h / 2;
        var angleDeg = MODELS_CFG.pca.state.angle;
        var theta = angleDeg * Math.PI / 180;
        var ux = Math.cos(theta), uy = -Math.sin(theta);

        // Left scatter panel background
        drawAxesGrid(ctx, 325, h, cx, cy, th);

        // Reference ghost axis for 1st Eigenvector (22 deg)
        var e1x = Math.cos(PCA_THETA1), e1y = -Math.sin(PCA_THETA1);
        ctx.save();
        ctx.strokeStyle = th.grid;
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1.3;
        ctx.beginPath();
        ctx.moveTo(cx - e1x * 135, cy - e1y * 135);
        ctx.lineTo(cx + e1x * 135, cy + e1y * 135);
        ctx.stroke();
        ctx.restore();

        // Projection rays & projected coordinates on u
        PCA_POINTS.forEach(function (p) {
          var sx = cx + p.x, sy = cy - p.y;
          var dot = p.x * ux + (-p.y) * uy;
          var projX = cx + dot * ux;
          var projY = cy + dot * uy;

          ctx.strokeStyle = th.isDark ? 'rgba(148, 163, 184, 0.25)' : 'rgba(100, 116, 139, 0.28)';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(sx, sy);
          ctx.lineTo(projX, projY);
          ctx.stroke();

          // Data point
          ctx.fillStyle = th.normal;
          ctx.beginPath();
          ctx.arc(sx, sy, 3.2, 0, Math.PI * 2);
          ctx.fill();

          // Projected dot on u
          ctx.fillStyle = th.gold;
          ctx.beginPath();
          ctx.arc(projX, projY, 2.8, 0, Math.PI * 2);
          ctx.fill();
        });

        // Active rotatable vector u
        var len = 138;
        ctx.strokeStyle = th.emerald;
        ctx.lineWidth = 2.8;
        ctx.beginPath();
        ctx.moveTo(cx - ux * len, cy - uy * len);
        ctx.lineTo(cx + ux * len, cy + uy * len);
        ctx.stroke();

        // Interactive drag handle at vector tip
        var tipX = cx + ux * 118, tipY = cy + uy * 118;
        ctx.fillStyle = th.emerald;
        ctx.beginPath();
        ctx.arc(tipX, tipY, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = th.bg;
        ctx.beginPath();
        ctx.arc(tipX, tipY, 2.3, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = th.text;
        ctx.font = '600 11px sans-serif';
        ctx.fillText('u (' + Math.round(angleDeg) + '°)', tipX + 8, tipY - 6);

        // Exact Rayleigh quotient variance u^T S u
        var diff = theta - PCA_THETA1;
        var rayleigh = PCA_L1 * Math.cos(diff) * Math.cos(diff) + PCA_L2 * Math.sin(diff) * Math.sin(diff);
        var pctVar = (rayleigh / (PCA_L1 + PCA_L2) * 100).toFixed(1);

        // Right mini-chart: Rayleigh Quotient Curve
        var cX = 342, cY = 38, cW = 104, cH = h - 76;
        roundRect(ctx, cX - 8, cY - 22, cW + 16, cH + 44, 6);
        ctx.fillStyle = th.surface;
        ctx.fill();
        ctx.strokeStyle = th.axis;
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.fillStyle = th.text;
        ctx.font = '600 10.5px sans-serif';
        ctx.fillText('Rayleigh u\u1D40Su', cX, cY - 7);

        // Axes of mini-chart
        ctx.strokeStyle = th.axis;
        ctx.beginPath();
        ctx.moveTo(cX, cY);
        ctx.lineTo(cX, cY + cH);
        ctx.lineTo(cX + cW, cY + cH);
        ctx.stroke();

        // Curve fill & stroke
        ctx.beginPath();
        for (var a = 0; a <= 180; a += 2) {
          var r = a * Math.PI / 180;
          var d = r - PCA_THETA1;
          var v = PCA_L1 * Math.cos(d) * Math.cos(d) + PCA_L2 * Math.sin(d) * Math.sin(d);
          var px = cX + (a / 180) * cW;
          var py = cY + cH - ((v - 0.25) / 2.25) * (cH - 14) - 4;
          if (a === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.strokeStyle = th.accent;
        ctx.lineWidth = 2.2;
        ctx.stroke();

        // Current angle vertical line & glowing indicator
        var curPx = cX + (angleDeg / 180) * cW;
        var curPy = cY + cH - ((rayleigh - 0.25) / 2.25) * (cH - 14) - 4;

        ctx.strokeStyle = th.tumor;
        ctx.lineWidth = 1.2;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(curPx, cY + 4);
        ctx.lineTo(curPx, cY + cH);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = th.tumorSoft;
        ctx.beginPath();
        ctx.arc(curPx, curPy, 7.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = th.tumor;
        ctx.beginPath();
        ctx.arc(curPx, curPy, 4, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = th.muted;
        ctx.font = '10px monospace';
        ctx.fillText('0°', cX, cY + cH + 13);
        ctx.fillText('90°', cX + cW * 0.43, cY + cH + 13);
        ctx.fillText('180°', cX + cW - 24, cY + cH + 13);

        return {
          metrics: '<span>u<sup>T</sup>Su = ' + rayleigh.toFixed(3) + ' (' + pctVar + '% Var)</span>' +
            '<span>&lambda;<sub>1</sub> = 2.354 (22&deg;)</span>' +
            '<span>&lambda;<sub>2</sub> = 0.370 (112&deg;)</span>' +
            '<span>Tr(S) = 2.724</span>',
          caption: 'Drag directly on the scatter plot or move the slider to rotate u. Variance peaks at the 1st eigenvector (22°) where orange projections spread widest.'
        };
      },
      eqs: '<div class="tex display">\\max_{\\|\\mathbf{u}\\|_2=1} \\mathbf{u}^T S \\mathbf{u} \\quad \\Longrightarrow \\quad S\\mathbf{u}_1 = \\lambda_1 \\mathbf{u}_1, \\quad \\det(S - \\lambda \\mathbf{I}) = 0</div>' +
        '<div class="tex display">\\text{Var}(\\mathbf{z}) = \\mathbf{u}(\\theta)^T S \\mathbf{u}(\\theta) = \\lambda_1 \\cos^2(\\theta - \\theta_1) + \\lambda_2 \\sin^2(\\theta - \\theta_1)</div>',
      theory: '<h4>Bilinear Pixel PCA Mechanics</h4>' +
        '<ul>' +
        '<li><strong>Spatial Compression:</strong> Downsamples <span class="tex">96\\times 96\\times 3</span> patches bilinearly to <span class="tex">24\\times 24\\times 3</span> and projects onto the top 20 eigenvectors of the training covariance matrix <span class="tex">S</span>.</li>' +
        '<li><strong>Spectral Invariance:</strong> The trace <span class="tex">\\lambda_1 + \\lambda_2 = 2.724</span> is constant under orthogonal rotation; aligning with <span class="tex">\\mathbf{u}_1</span> retains 86.4% of the 2D energy in a single coordinate.</li>' +
        '</ul>'
    },

    /* ================================================================
       2. Logistic Regression (Linear Sigmoid & L2 Penalty)
       ================================================================ */
    logreg: {
      badge: 'Rank #10: Linear Baseline (75.84% Test Acc)',
      badgeClass: 'badge-normal',
      state: { lambda: 1.0, threshold: 0.50, tilt: 0 },
      controls: function () {
        var s = MODELS_CFG.logreg.state;
        return '<div class="ctrl-row">' +
          '<div class="ctrl-label"><span>L<sub>2</sub> Regularisation Penalty (&lambda;)</span><span id="lrVal1">' + s.lambda.toFixed(2) + '</span></div>' +
          '<input type="range" id="lrSlider1" min="0.05" max="5.00" step="0.05" value="' + s.lambda + '">' +
          '</div>' +
          '<div class="ctrl-row">' +
          '<div class="ctrl-label"><span>Operating Decision Threshold (P<sub>0</sub> / &tau;)</span><span id="lrVal2">' + s.threshold.toFixed(2) + '</span></div>' +
          '<input type="range" id="lrSlider2" min="0.10" max="0.90" step="0.02" value="' + s.threshold + '">' +
          '</div>' +
          '<div class="ctrl-row">' +
          '<div class="ctrl-label"><span>Hyperplane Rotation Bias</span><span id="lrVal3">' + s.tilt + '&deg;</span></div>' +
          '<input type="range" id="lrSlider3" min="-35" max="35" step="1" value="' + s.tilt + '">' +
          '</div>';
      },
      init: function (update) {
        var s1 = $('#lrSlider1'), s2 = $('#lrSlider2'), s3 = $('#lrSlider3');
        if (s1) s1.addEventListener('input', function () {
          MODELS_CFG.logreg.state.lambda = parseFloat(s1.value);
          $('#lrVal1').textContent = MODELS_CFG.logreg.state.lambda.toFixed(2);
          update();
        });
        if (s2) s2.addEventListener('input', function () {
          MODELS_CFG.logreg.state.threshold = parseFloat(s2.value);
          $('#lrVal2').textContent = MODELS_CFG.logreg.state.threshold.toFixed(2);
          update();
        });
        if (s3) s3.addEventListener('input', function () {
          MODELS_CFG.logreg.state.tilt = parseFloat(s3.value);
          $('#lrVal3').textContent = s3.value + '°';
          update();
        });
      },
      render: function (ctx, w, h) {
        var th = getTheme();
        var cx = w / 2, cy = h / 2;
        var p = MODELS_CFG.logreg.state;
        var rad = (-32 + p.tilt) * Math.PI / 180;
        var nx = Math.cos(rad), ny = Math.sin(rad);
        var normW = 1.65 / (1.0 + 0.42 * p.lambda);

        var probFn = function (x, y) {
          var z = (x * nx + y * ny) * 0.032 * normW;
          return 1.0 / (1.0 + Math.exp(-z));
        };

        drawProbabilityField(ctx, w, h, cx, cy, th, probFn, p.threshold, 6);
        drawAxesGrid(ctx, w, h, cx, cy, th);

        // Draw iso-probability lines for P=0.25, P=P0, P=0.75
        var drawIsoLine = function (probLevel, color, dashed, labelTxt) {
          var lVal = Math.log(probLevel / (1.0 - probLevel));
          var dist = lVal / (0.032 * normW);
          var ox = cx + nx * dist, oy = cy + ny * dist;
          var tx = -ny * 280, ty = nx * 280;
          ctx.save();
          ctx.strokeStyle = color;
          ctx.lineWidth = dashed ? 1.5 : 2.5;
          if (dashed) ctx.setLineDash([5, 5]);
          ctx.beginPath();
          ctx.moveTo(ox - tx, oy - ty);
          ctx.lineTo(ox + tx, oy + ty);
          ctx.stroke();
          ctx.fillStyle = color;
          ctx.font = '600 10.5px monospace';
          ctx.fillText(labelTxt, ox + tx * 0.32 + 6, oy + ty * 0.32 - 6);
          ctx.restore();
        };

        drawIsoLine(0.25, th.normal, true, 'P=0.25');
        drawIsoLine(p.threshold, th.boundary, false, 'P=' + p.threshold.toFixed(2));
        drawIsoLine(0.75, th.tumor, true, 'P=0.75');

        var mis = drawScatterPoints(ctx, cx, cy, th, probFn, p.threshold);

        return {
          metrics: '<span>&lambda; = ' + p.lambda.toFixed(2) + '</span>' +
            '<span>||w||<sub>2</sub> = ' + normW.toFixed(2) + '</span>' +
            '<span>Misclassified = ' + mis + ' / ' + CLF_POINTS.length + '</span>' +
            '<span>Sealed Test Acc = 75.84%</span>',
          caption: 'Raising L2 penalty λ shrinks ||w||₂ and widens the P=0.25..0.75 transition band. Gold rings highlight misclassified patches.'
        };
      },
      eqs: '<div class="tex display">P(y=1 \\mid \\mathbf{x}) = \\sigma(\\mathbf{w}^T \\mathbf{x} + b) = \\frac{1}{1 + \\exp(-(\\mathbf{w}^T \\mathbf{x} + b))}</div>' +
        '<div class="tex display">\\mathcal{L}(\\mathbf{w}, b) = -\\frac{1}{N}\\sum_{i=1}^N \\left[ y_i \\ln p_i + (1 - y_i)\\ln(1 - p_i) \\right] + \\frac{\\lambda}{2}\\|\\mathbf{w}\\|_2^2</div>',
      theory: '<h4>Logistic Regression &amp; The Linear Ceiling</h4>' +
        '<ul>' +
        '<li><strong>Why It Plateaus at 75.84%:</strong> A single flat hyperplane <span class="tex">\\mathbf{w}^T\\mathbf{x} + b = \\ln(P_0/(1-P_0))</span> cannot curve around reactive lymphoid follicles that overlap with tumour clusters.</li>' +
        '<li><strong>Calibration Role:</strong> Produces smooth log-odds posterior gradients, regulated by <span class="tex">\\lambda</span> to prevent weight explosion on collinear colour moments.</li>' +
        '</ul>'
    },

    /* ================================================================
       3. Linear SVM (Maximum Margin Hyperplane)
       ================================================================ */
    linsvm: {
      badge: 'Rank #11: Linear Margin Baseline (75.46% Test Acc)',
      badgeClass: 'badge-normal',
      state: { C: 1.0, angle: 0, bias: 0 },
      controls: function () {
        var s = MODELS_CFG.linsvm.state;
        return '<div class="ctrl-row">' +
          '<div class="ctrl-label"><span>Soft-Margin Penalty (C)</span><span id="lsvmVal1">' + s.C.toFixed(1) + '</span></div>' +
          '<input type="range" id="lsvmSlider1" min="0.1" max="10.0" step="0.1" value="' + s.C + '">' +
          '</div>' +
          '<div class="ctrl-row">' +
          '<div class="ctrl-label"><span>Hyperplane Tilt Angle</span><span id="lsvmVal2">' + s.angle + '&deg;</span></div>' +
          '<input type="range" id="lsvmSlider2" min="-30" max="30" step="1" value="' + s.angle + '">' +
          '</div>' +
          '<div class="ctrl-row">' +
          '<div class="ctrl-label"><span>Bias Offset (b)</span><span id="lsvmVal3">' + s.bias + ' px</span></div>' +
          '<input type="range" id="lsvmSlider3" min="-40" max="40" step="1" value="' + s.bias + '">' +
          '</div>';
      },
      init: function (update) {
        var s1 = $('#lsvmSlider1'), s2 = $('#lsvmSlider2'), s3 = $('#lsvmSlider3');
        if (s1) s1.addEventListener('input', function () {
          MODELS_CFG.linsvm.state.C = parseFloat(s1.value);
          $('#lsvmVal1').textContent = MODELS_CFG.linsvm.state.C.toFixed(1);
          update();
        });
        if (s2) s2.addEventListener('input', function () {
          MODELS_CFG.linsvm.state.angle = parseFloat(s2.value);
          $('#lsvmVal2').textContent = s2.value + '°';
          update();
        });
        if (s3) s3.addEventListener('input', function () {
          MODELS_CFG.linsvm.state.bias = parseFloat(s3.value);
          $('#lsvmVal3').textContent = s3.value + ' px';
          update();
        });
      },
      render: function (ctx, w, h) {
        var th = getTheme();
        var cx = w / 2, cy = h / 2;
        var p = MODELS_CFG.linsvm.state;
        var rad = (-32 + p.angle) * Math.PI / 180;
        var nx = Math.cos(rad), ny = Math.sin(rad);
        var halfMargin = Math.max(11, 42 / Math.pow(p.C, 0.42));

        drawAxesGrid(ctx, w, h, cx, cy, th);

        var ox = cx + nx * p.bias, oy = cy + ny * p.bias;
        var dx = -ny * 300, dy = nx * 300;

        // Shaded geometric margin corridor (width = 2 / ||w||)
        ctx.save();
        ctx.fillStyle = th.corridor;
        ctx.beginPath();
        ctx.moveTo(ox - nx * halfMargin - dx, oy - ny * halfMargin - dy);
        ctx.lineTo(ox - nx * halfMargin + dx, oy - ny * halfMargin + dy);
        ctx.lineTo(ox + nx * halfMargin + dx, oy + ny * halfMargin + dy);
        ctx.lineTo(ox + nx * halfMargin - dx, oy + ny * halfMargin - dy);
        ctx.closePath();
        ctx.fill();
        ctx.restore();

        var drawPlane = function (offset, col, dashed) {
          ctx.save();
          ctx.strokeStyle = col;
          ctx.lineWidth = dashed ? 1.5 : 2.5;
          if (dashed) ctx.setLineDash([5, 5]);
          ctx.beginPath();
          ctx.moveTo(ox + nx * offset - dx, oy + ny * offset - dy);
          ctx.lineTo(ox + nx * offset + dx, oy + ny * offset + dy);
          ctx.stroke();
          ctx.restore();
        };

        drawPlane(-halfMargin, th.normal, true);
        drawPlane(0, th.boundary, false);
        drawPlane(halfMargin, th.tumor, true);

        var svCount = 0, slackSum = 0;
        CLF_POINTS.forEach(function (pt) {
          var sx = cx + pt.x, sy = cy + pt.y;
          var signedDist = (pt.x * nx + pt.y * ny - p.bias);
          var ySign = pt.label === 1 ? 1 : -1;
          var functionalMargin = ySign * (signedDist / halfMargin);
          var isSV = functionalMargin <= 1.02;

          if (isSV) {
            svCount++;
            var xi = Math.max(0, 1.0 - functionalMargin);
            slackSum += xi;

            // Slack vector line to margin boundary
            var targetOffset = ySign * halfMargin;
            var projDist = signedDist - targetOffset;
            ctx.save();
            ctx.strokeStyle = th.gold;
            ctx.lineWidth = 1;
            ctx.setLineDash([2, 2]);
            ctx.beginPath();
            ctx.moveTo(sx, sy);
            ctx.lineTo(sx - nx * projDist, sy - ny * projDist);
            ctx.stroke();
            ctx.restore();

            // Glowing Support Vector ring
            ctx.save();
            ctx.fillStyle = 'rgba(234, 179, 8, 0.22)';
            ctx.beginPath();
            ctx.arc(sx, sy, 8.2, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = th.gold;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(sx, sy, 6.2, 0, Math.PI * 2);
            ctx.stroke();
            ctx.restore();
          }

          ctx.fillStyle = pt.label === 1 ? th.tumor : th.normal;
          ctx.strokeStyle = th.bg;
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.arc(sx, sy, 3.8, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        });

        return {
          metrics: '<span>Margin &Delta; = ' + (halfMargin * 2).toFixed(1) + ' px</span>' +
            '<span>Active SVs (&alpha;<sub>i</sub>&gt;0) = ' + svCount + '</span>' +
            '<span>&Sigma;&xi;<sub>i</sub> = ' + slackSum.toFixed(1) + '</span>' +
            '<span>Test Acc = 75.46%</span>',
          caption: 'Gold halos mark Support Vectors (αᵢ > 0) inside or violating the margin corridor Δ = 2/||w||₂. Increasing C narrows Δ.'
        };
      },
      eqs: '<div class="tex display">\\min_{\\mathbf{w}, b, \\boldsymbol{\\xi}} \\; \\frac{1}{2}\\|\\mathbf{w}\\|_2^2 + C \\sum_{i=1}^N \\xi_i \\quad \\text{s.t.} \\quad y_i(\\mathbf{w}^T \\mathbf{x}_i + b) \\ge 1 - \\xi_i, \\; \\xi_i \\ge 0</div>' +
        '<div class="tex display">\\text{Geometric Margin Width: } \\Delta = \\frac{2}{\\|\\mathbf{w}\\|_2}, \\quad \\mathbf{w} = \\sum_{i \\in \\text{SV}} \\alpha_i y_i \\mathbf{x}_i</div>',
      theory: '<h4>Linear Support Vector Machine Mechanics</h4>' +
        '<ul>' +
        '<li><strong>Support Vector Sparsity:</strong> By the Karush-Kuhn-Tucker (KKT) conditions, only patches on or inside the margin corridor have <span class="tex">\\alpha_i &gt; 0</span>.</li>' +
        '<li><strong>Confirming the Linear Ceiling:</strong> Finishes within 0.38% of Logistic Regression (75.46% vs 75.84%), proving the bottleneck is geometric linearity rather than loss function choice.</li>' +
        '</ul>'
    },

    /* ================================================================
       4. RBF Kernel SVM (Non-linear Gaussian Hilbert Space)
       ================================================================ */
    rbfsvm: {
      badge: 'Rank #7: Non-linear Kernel Machine (78.80% Test Acc)',
      badgeClass: 'badge-normal',
      state: { gamma: 1.2, C: 3.0 },
      controls: function () {
        var s = MODELS_CFG.rbfsvm.state;
        return '<div class="ctrl-row">' +
          '<div class="ctrl-label"><span>RBF Kernel Bandwidth (&gamma;)</span><span id="rbfVal1">' + s.gamma.toFixed(1) + '</span></div>' +
          '<input type="range" id="rbfSlider1" min="0.2" max="5.0" step="0.2" value="' + s.gamma + '">' +
          '</div>' +
          '<div class="ctrl-row">' +
          '<div class="ctrl-label"><span>Regularisation Penalty (C)</span><span id="rbfVal2">' + s.C.toFixed(1) + '</span></div>' +
          '<input type="range" id="rbfSlider2" min="0.5" max="8.0" step="0.5" value="' + s.C + '">' +
          '</div>';
      },
      init: function (update) {
        var s1 = $('#rbfSlider1'), s2 = $('#rbfSlider2');
        if (s1) s1.addEventListener('input', function () {
          MODELS_CFG.rbfsvm.state.gamma = parseFloat(s1.value);
          $('#rbfVal1').textContent = MODELS_CFG.rbfsvm.state.gamma.toFixed(1);
          update();
        });
        if (s2) s2.addEventListener('input', function () {
          MODELS_CFG.rbfsvm.state.C = parseFloat(s2.value);
          $('#rbfVal2').textContent = MODELS_CFG.rbfsvm.state.C.toFixed(1);
          update();
        });
      },
      render: function (ctx, w, h) {
        var th = getTheme();
        var cx = w / 2, cy = h / 2;
        var p = MODELS_CFG.rbfsvm.state;
        var sigmaPx = Math.round(52 / Math.sqrt(2 * p.gamma));

        var probFn = function (x, y) {
          return evalRbfSvmProb(x, y, p.gamma, p.C);
        };

        drawProbabilityField(ctx, w, h, cx, cy, th, probFn, 0.5, 5);
        drawAxesGrid(ctx, w, h, cx, cy, th);

        // Visualise effective Gaussian influence radius sigma around a representative tumour support vector
        var refPt = CLF_POINTS[25];
        ctx.save();
        ctx.strokeStyle = th.gold;
        ctx.lineWidth = 1.4;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.arc(cx + refPt.x, cy + refPt.y, sigmaPx, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();

        var mis = drawScatterPoints(ctx, cx, cy, th, probFn, 0.5);

        return {
          metrics: '<span>&gamma; = ' + p.gamma.toFixed(1) + '</span>' +
            '<span>Influence Radius &sigma; = ' + sigmaPx + ' px</span>' +
            '<span>Errors = ' + mis + ' / ' + CLF_POINTS.length + '</span>' +
            '<span>Val AUC = 0.9082 (Test 78.80%)</span>',
          caption: 'Smooth curved manifold in Hilbert space. High γ shrinks radius σ (dashed gold circle) into tight islands around individual cells.'
        };
      },
      eqs: '<div class="tex display">K(\\mathbf{x}, \\mathbf{x}^\\prime) = \\exp\\left(-\\gamma \\|\\mathbf{x} - \\mathbf{x}^\\prime\\|_2^2\\right) = \\exp\\left(-\\frac{\\|\\mathbf{x} - \\mathbf{x}^\\prime\\|_2^2}{2\\sigma^2}\\right)</div>' +
        '<div class="tex display">f(\\mathbf{x}) = \\text{sign}\\left( \\sum_{i \\in \\text{SV}} \\alpha_i y_i K(\\mathbf{x}_i, \\mathbf{x}) + b \\right), \\quad \\sigma = \\frac{1}{\\sqrt{2\\gamma}}</div>',
      theory: '<h4>RBF Kernel SVM &amp; Infinite-Dimensional RKHS</h4>' +
        '<ul>' +
        '<li><strong>Breaking the Linear Ceiling:</strong> Lifting the 59 features via a radial basis kernel boosts test accuracy by <strong>+3.34 points</strong> over Linear SVM (78.80% vs 75.46%) and achieves the highest sensitivity among classical models (73.28%).</li>' +
        '<li><strong>Bandwidth Trade-off:</strong> Small <span class="tex">\\gamma</span> acts like a global linear model; excessive <span class="tex">\\gamma</span> memorises training slide noise (15,569 support vectors at <span class="tex">C=3.0</span>).</li>' +
        '</ul>'
    },

    /* ================================================================
       5. Random Forest (Bagging & Axis-Aligned Rectangular Splits)
       ================================================================ */
    rf: {
      badge: 'Rank #8: Bagged Tree Ensemble (78.46% Test Acc)',
      badgeClass: 'badge-normal',
      state: { trees: 12, depth: 3, subspace: 0.70 },
      controls: function () {
        var s = MODELS_CFG.rf.state;
        return '<div class="ctrl-row">' +
          '<div class="ctrl-label"><span>Number of Bagged Trees (B)</span><span id="rfVal1">' + s.trees + '</span></div>' +
          '<input type="range" id="rfSlider1" min="1" max="25" step="1" value="' + s.trees + '">' +
          '</div>' +
          '<div class="ctrl-row">' +
          '<div class="ctrl-label"><span>Maximum Tree Depth (d)</span><span id="rfVal2">' + s.depth + '</span></div>' +
          '<input type="range" id="rfSlider2" min="1" max="5" step="1" value="' + s.depth + '">' +
          '</div>' +
          '<div class="ctrl-row">' +
          '<div class="ctrl-label"><span>Feature Subspace Ratio (m / p)</span><span id="rfVal3">' + s.subspace.toFixed(2) + '</span></div>' +
          '<input type="range" id="rfSlider3" min="0.20" max="1.00" step="0.05" value="' + s.subspace + '">' +
          '</div>';
      },
      init: function (update) {
        var s1 = $('#rfSlider1'), s2 = $('#rfSlider2'), s3 = $('#rfSlider3');
        if (s1) s1.addEventListener('input', function () {
          MODELS_CFG.rf.state.trees = parseInt(s1.value, 10);
          $('#rfVal1').textContent = s1.value;
          update();
        });
        if (s2) s2.addEventListener('input', function () {
          MODELS_CFG.rf.state.depth = parseInt(s2.value, 10);
          $('#rfVal2').textContent = s2.value;
          update();
        });
        if (s3) s3.addEventListener('input', function () {
          MODELS_CFG.rf.state.subspace = parseFloat(s3.value);
          $('#rfVal3').textContent = MODELS_CFG.rf.state.subspace.toFixed(2);
          update();
        });
      },
      render: function (ctx, w, h) {
        var th = getTheme();
        var cx = w / 2, cy = h / 2;
        var p = MODELS_CFG.rf.state;

        var probFn = function (x, y) {
          return evalForestProb(x, y, p.trees, p.depth, p.subspace);
        };

        // Render true axis-aligned stair-step partition field
        drawProbabilityField(ctx, w, h, cx, cy, th, probFn, 0.5, 6);
        drawAxesGrid(ctx, w, h, cx, cy, th);

        // Faintly overlay individual tree orthogonal split planes
        ctx.save();
        ctx.lineWidth = 1;
        for (var b = 0; b < Math.min(p.trees, 12); b++) {
          var tr = RF_TREES[b];
          ctx.strokeStyle = th.isDark ? 'rgba(169, 154, 232, 0.16)' : 'rgba(76, 58, 142, 0.14)';
          ctx.beginPath();
          if (tr.flip) {
            var sx = cx + tr.ax1 * p.subspace;
            ctx.moveTo(sx, 12); ctx.lineTo(sx, h - 12);
          } else {
            var sy = cy + tr.ay1 * p.subspace;
            ctx.moveTo(12, sy); ctx.lineTo(w - 12, sy);
          }
          ctx.stroke();
        }
        ctx.restore();

        var mis = drawScatterPoints(ctx, cx, cy, th, probFn, 0.5);

        // Variance calculation: rho * sigma^2 + (1-rho)/B * sigma^2
        var rho = 0.22 + 0.45 * p.subspace;
        var varRatio = rho + (1 - rho) / p.trees;
        var varReduction = ((1 - varRatio) * 100).toFixed(1);

        return {
          metrics: '<span>Trees B = ' + p.trees + '</span>' +
            '<span>Depth = ' + p.depth + '</span>' +
            '<span>Var Reduction = ' + varReduction + '%</span>' +
            '<span>Specificity = 88.16% (Best Classical)</span>',
          caption: 'Orthogonal stair-step boundary formed by averaging B axis-aligned trees. Set B=1 vs B=25 to see bagging stabilise noisy splits.'
        };
      },
      eqs: '<div class="tex display">\\hat{P}_{\\text{RF}}(y=1 \\mid \\mathbf{x}) = \\frac{1}{B}\\sum_{b=1}^B T_b(\\mathbf{x}; \\Theta_b), \\quad I_{\\text{Gini}}(t) = 1 - \\sum_{k \\in \\{0,1\\}} p_{tk}^2</div>' +
        '<div class="tex display">\\text{Var}(\\bar{T}) = \\rho \\sigma^2 + \\frac{1 - \\rho}{B}\\sigma^2 \\quad (\\text{Subspace sampling lowers tree correlation } \\rho)</div>',
      theory: '<h4>Random Forest Bagging &amp; Orthogonal Partitioning</h4>' +
        '<ul>' +
        '<li><strong>Axis-Aligned Geometry:</strong> Each node tests a single feature threshold <span class="tex">x_j \\le t</span>, carving feature space into hyper-rectangles.</li>' +
        '<li><strong>Lowest Generalisation Drop:</strong> Dropped only <strong>0.98 points</strong> from validation (79.44%) to sealed test (78.46%) with 88.16% specificity.</li>' +
        '</ul>'
    },

    /* ================================================================
       6. HistGradientBoosting (Sequential Residual Boosting)
       ================================================================ */
    histgbm: {
      badge: 'Rank #6: Single-Model Classical Champion (78.82%)',
      badgeClass: 'badge-normal',
      state: { stages: 6, lr: 0.15 },
      controls: function () {
        var s = MODELS_CFG.histgbm.state;
        return '<div class="ctrl-row">' +
          '<div class="ctrl-label"><span>Additive Boosting Stages (M)</span><span id="hgbVal1">Stage ' + s.stages + ' / 10</span></div>' +
          '<input type="range" id="hgbSlider1" min="1" max="10" step="1" value="' + s.stages + '">' +
          '</div>' +
          '<div class="ctrl-row">' +
          '<div class="ctrl-label"><span>Shrinkage Learning Rate (&eta;)</span><span id="hgbVal2">' + s.lr.toFixed(2) + '</span></div>' +
          '<input type="range" id="hgbSlider2" min="0.05" max="0.50" step="0.05" value="' + s.lr + '">' +
          '</div>';
      },
      init: function (update) {
        var s1 = $('#hgbSlider1'), s2 = $('#hgbSlider2');
        if (s1) s1.addEventListener('input', function () {
          MODELS_CFG.histgbm.state.stages = parseInt(s1.value, 10);
          $('#hgbVal1').textContent = 'Stage ' + s1.value + ' / 10';
          update();
        });
        if (s2) s2.addEventListener('input', function () {
          MODELS_CFG.histgbm.state.lr = parseFloat(s2.value);
          $('#hgbVal2').textContent = MODELS_CFG.histgbm.state.lr.toFixed(2);
          update();
        });
      },
      render: function (ctx, w, h) {
        var th = getTheme();
        var cx = w / 2, cy = h / 2;
        var p = MODELS_CFG.histgbm.state;

        var probFn = function (x, y) {
          return evalHistGbmProb(x, y, p.stages, p.lr);
        };

        drawProbabilityField(ctx, w, h, cx, cy, th, probFn, 0.5, 6);

        // Draw subtle 256-bin histogram quantization grid
        ctx.save();
        ctx.strokeStyle = th.grid;
        ctx.lineWidth = 0.7;
        for (var bx = 0; bx < w; bx += 18) {
          ctx.beginPath(); ctx.moveTo(bx, 0); ctx.lineTo(bx, h); ctx.stroke();
        }
        for (var by = 0; by < h; by += 18) {
          ctx.beginPath(); ctx.moveTo(0, by); ctx.lineTo(w, by); ctx.stroke();
        }
        ctx.restore();

        // Highlight the active stage corrective region box
        var activeStage = GBM_STAGES[p.stages - 1];
        if (activeStage) {
          var rx = cx + activeStage.x0, ry = cy + activeStage.y0;
          var rw = activeStage.x1 - activeStage.x0, rh = activeStage.y1 - activeStage.y0;
          ctx.save();
          ctx.strokeStyle = activeStage.val > 0 ? th.gold : th.emerald;
          ctx.lineWidth = 2.2;
          ctx.setLineDash([5, 3]);
          ctx.strokeRect(rx, ry, rw, rh);
          ctx.fillStyle = th.surface;
          ctx.fillRect(12, 10, 255, 22);
          ctx.fillStyle = th.text;
          ctx.font = '600 11px sans-serif';
          ctx.fillText('Stage ' + p.stages + ': ' + activeStage.label, 18, 25);
          ctx.restore();
        }

        // Compute log-loss and draw residual error bars on points
        var logLoss = 0;
        CLF_POINTS.forEach(function (pt) {
          var prob = Math.max(0.01, Math.min(0.99, probFn(pt.x, pt.y)));
          var resid = Math.abs(pt.label - prob);
          logLoss += -(pt.label * Math.log(prob) + (1 - pt.label) * Math.log(1 - prob));

          // Residual error bar above point
          if (resid > 0.25) {
            var sx = cx + pt.x, sy = cy + pt.y;
            ctx.fillStyle = th.gold;
            ctx.fillRect(sx - 1.5, sy - 4 - resid * 16, 3, resid * 16);
          }
        });
        logLoss /= CLF_POINTS.length;

        var mis = drawScatterPoints(ctx, cx, cy, th, probFn, 0.5);

        return {
          metrics: '<span>Stage M = ' + p.stages + '</span>' +
            '<span>Log-Loss = ' + logLoss.toFixed(3) + '</span>' +
            '<span>Errors = ' + mis + ' / ' + CLF_POINTS.length + '</span>' +
            '<span>256 Histogram Bins (Test 78.82%)</span>',
          caption: 'Sweep M from 1 to 10: each stage targets high-residual patches (vertical gold bars) and adds a corrective region box (dashed).'
        };
      },
      eqs: '<div class="tex display">r_{im} = -\\left[\\frac{\\partial \\mathcal{L}(y_i, F(\\mathbf{x}_i))}{\\partial F(\\mathbf{x}_i)}\\right]_{F=F_{m-1}} = y_i - \\sigma(F_{m-1}(\\mathbf{x}_i))</div>' +
        '<div class="tex display">F_M(\\mathbf{x}) = F_0(\\mathbf{x}) + \\sum_{m=1}^M \\eta \\cdot h_m(\\mathbf{x}) \\quad (\\text{256-bin histogram split scanning})</div>',
      theory: '<h4>HistGradientBoosting Functional Gradient Descent</h4>' +
        '<ul>' +
        '<li><strong>Sequential Residual Targeting:</strong> Unlike Random Forest (parallel independent trees), HistGBM fits tree <span class="tex">h_m</span> directly to the pseudo-residuals <span class="tex">r_{im} = y_i - p_i</span> of stage <span class="tex">m-1</span>.</li>' +
        '<li><strong>Top Permutation Features:</strong> Relies heavily on <code>lbp_bin_5</code> (nuclear chromatin pattern) and CIE-Lab <code>color_b_mean</code>.</li>' +
        '</ul>'
    },

    /* ================================================================
       7. Multi-Paradigm Blend (Convex Probability Fusion)
       ================================================================ */
    blend: {
      badge: 'Rank #4: Classical Benchmark Champion (79.76% Test Acc)',
      badgeClass: 'badge-tumor',
      state: { weight: 0.50, tau: 0.50 },
      controls: function () {
        var s = MODELS_CFG.blend.state;
        return '<div class="ctrl-row">' +
          '<div class="ctrl-label"><span>RBF SVM Weight (w) vs. HistGBM (1&minus;w)</span><span id="bldVal1">w = ' + s.weight.toFixed(2) + '</span></div>' +
          '<input type="range" id="bldSlider1" min="0.00" max="1.00" step="0.05" value="' + s.weight + '">' +
          '</div>' +
          '<div class="ctrl-row">' +
          '<div class="ctrl-label"><span>Decision Threshold (&tau;)</span><span id="bldVal2">' + s.tau.toFixed(2) + '</span></div>' +
          '<input type="range" id="bldSlider2" min="0.20" max="0.80" step="0.02" value="' + s.tau + '">' +
          '</div>';
      },
      init: function (update) {
        var s1 = $('#bldSlider1'), s2 = $('#bldSlider2');
        if (s1) s1.addEventListener('input', function () {
          MODELS_CFG.blend.state.weight = parseFloat(s1.value);
          $('#bldVal1').textContent = 'w = ' + MODELS_CFG.blend.state.weight.toFixed(2);
          update();
        });
        if (s2) s2.addEventListener('input', function () {
          MODELS_CFG.blend.state.tau = parseFloat(s2.value);
          $('#bldVal2').textContent = MODELS_CFG.blend.state.tau.toFixed(2);
          update();
        });
      },
      render: function (ctx, w, h) {
        var th = getTheme();
        var cx = w / 2, cy = h / 2;
        var p = MODELS_CFG.blend.state;

        var svmFn = function (x, y) { return evalRbfSvmProb(x, y, 1.3, 3.0); };
        var gbmFn = function (x, y) { return evalHistGbmProb(x, y, 8, 0.16); };
        var blendFn = function (x, y) {
          return p.weight * svmFn(x, y) + (1.0 - p.weight) * gbmFn(x, y);
        };

        drawProbabilityField(ctx, w, h, cx, cy, th, blendFn, p.tau, 6);
        drawAxesGrid(ctx, w, h, cx, cy, th);

        // Highlight disagreement zones where SVM (curved) and HistGBM (rectangular) disagree
        var disputedPts = 0;
        CLF_POINTS.forEach(function (pt) {
          var cSvm = svmFn(pt.x, pt.y) >= p.tau ? 1 : 0;
          var cGbm = gbmFn(pt.x, pt.y) >= p.tau ? 1 : 0;
          if (cSvm !== cGbm) {
            disputedPts++;
            var sx = cx + pt.x, sy = cy + pt.y;
            ctx.save();
            ctx.strokeStyle = th.purple;
            ctx.fillStyle = th.isDark ? 'rgba(192, 132, 252, 0.20)' : 'rgba(124, 58, 237, 0.14)';
            ctx.lineWidth = 1.8;
            ctx.fillRect(sx - 8, sy - 8, 16, 16);
            ctx.strokeRect(sx - 8, sy - 8, 16, 16);
            ctx.restore();
          }
        });

        var mis = drawScatterPoints(ctx, cx, cy, th, blendFn, p.tau);
        // Interpolate empirical Brier score across w
        var brier = (0.1426 + 0.0064 * Math.pow(p.weight - 0.5, 2) * 4).toFixed(4);

        return {
          metrics: '<span>w = ' + p.weight.toFixed(2) + ' SVM / ' + (1 - p.weight).toFixed(2) + ' GBM</span>' +
            '<span>Disputed Zones = ' + disputedPts + '</span>' +
            '<span>Brier Score = ' + brier + '</span>' +
            '<span>Test Acc = 79.76%</span>',
          caption: 'Violet squares highlight disagreement zones where curved SVM and rectangular HistGBM conflict. Blending at w=0.50 cancels individual blind spots.'
        };
      },
      eqs: '<div class="tex display">P_{\\text{blend}}(y = 1 \\mid \\mathbf{x}) = w \\cdot P_{\\text{RBF-SVM}}(\\mathbf{x}) + (1 - w) \\cdot P_{\\text{HistGBM}}(\\mathbf{x})</div>' +
        '<div class="tex display">\\text{Brier Score} = \\frac{1}{N}\\sum_{i=1}^N (P_{\\text{blend}}(\\mathbf{x}_i) - y_i)^2 = 0.1426 \\quad (\\text{Error Corr } \\rho = 0.67)</div>',
      theory: '<h4>Orthogonal Inductive Bias Synergy</h4>' +
        '<ul>' +
        '<li><strong>Why Blending Wins:</strong> RBF SVM measures smooth Euclidean kernel distances across all 59 features at once, whereas HistGBM cuts axis-by-axis rectangles.</li>' +
        '<li><strong>1,064 Disputed Test Patches:</strong> On the 5,000 test patches, the two families disagree on 1,064 cases; convex probability averaging resolves the majority in favour of the true label.</li>' +
        '</ul>'
    },

    /* ================================================================
       8. ResNet-18 (Deep Residual Identity Highway)
       ================================================================ */
    resnet18: {
      badge: 'Rank #1: Overall Project Champion (84.94% Test Acc, 0.9392 AUC)',
      badgeClass: 'badge-tumor',
      state: { depth: 18, skip: true },
      controls: function () {
        var s = MODELS_CFG.resnet18.state;
        return '<div class="ctrl-row">' +
          '<div class="ctrl-label"><span>Active Network Depth (Conv Layers)</span><span id="rnVal1">' + s.depth + ' Layers (' + (s.depth / 2 - 1) + ' ResBlocks)</span></div>' +
          '<input type="range" id="rnSlider1" min="4" max="18" step="2" value="' + s.depth + '">' +
          '</div>' +
          '<div class="explorebtnrow">' +
          '<button type="button" class="btn ' + (s.skip ? 'primary' : '') + '" id="btnSkipToggle">' +
          'Identity Skip Connections (+x): ' + (s.skip ? 'ON (99.2% Signal)' : 'OFF (Vanishing Gradients)') +
          '</button>' +
          '</div>';
      },
      init: function (update) {
        var s1 = $('#rnSlider1'), btn = $('#btnSkipToggle');
        if (s1) s1.addEventListener('input', function () {
          MODELS_CFG.resnet18.state.depth = parseInt(s1.value, 10);
          var d = MODELS_CFG.resnet18.state.depth;
          $('#rnVal1').textContent = d + ' Layers (' + (d / 2 - 1) + ' ResBlocks)';
          update();
        });
        if (btn) btn.addEventListener('click', function () {
          MODELS_CFG.resnet18.state.skip = !MODELS_CFG.resnet18.state.skip;
          var on = MODELS_CFG.resnet18.state.skip;
          btn.textContent = 'Identity Skip Connections (+x): ' + (on ? 'ON (99.2% Signal)' : 'OFF (Vanishing Gradients)');
          btn.className = on ? 'btn primary' : 'btn';
          update();
        });
      },
      render: function (ctx, w, h) {
        var th = getTheme();
        var p = MODELS_CFG.resnet18.state;
        var numBlocks = Math.max(1, Math.floor((p.depth - 2) / 2)); // Up to 8 residual blocks
        var startX = 32, endX = w - 24, cy = 132;
        var spanW = endX - startX;
        var stepX = spanW / (numBlocks + 0.4);

        // Title header
        ctx.fillStyle = th.text;
        ctx.font = '600 12px sans-serif';
        ctx.fillText('ResNet-18 Residual Pipeline & Backpropagation Gradient Highway', 24, 24);

        // Compute per-block backprop gradient magnitude (from right to left: output -> input)
        var retention = p.skip ? (100 - numBlocks * 0.1) : Math.max(1.2, Math.pow(0.60, numBlocks) * 100);

        for (var i = 0; i < numBlocks; i++) {
          var bx = startX + i * stepX + 18;
          var bw = Math.min(34, stepX * 0.58);
          var revIdx = numBlocks - 1 - i; // distance from loss output
          var localGrad = p.skip ? Math.pow(0.99, revIdx) : Math.pow(0.60, revIdx + 1);

          // Main feed-forward stem line
          if (i < numBlocks - 1) {
            ctx.strokeStyle = th.axis;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(bx + bw, cy);
            ctx.lineTo(bx + stepX, cy);
            ctx.stroke();
          }

          // Residual Block F(x)
          roundRect(ctx, bx, cy - 28, bw, 56, 5);
          ctx.fillStyle = p.skip
            ? (th.isDark ? 'rgba(56, 189, 248, 0.18)' : 'rgba(2, 132, 199, 0.12)')
            : (th.isDark ? 'rgba(244, 63, 94, ' + (0.1 + 0.35 * localGrad).toFixed(2) + ')' : 'rgba(225, 29, 72, ' + (0.08 + 0.35 * localGrad).toFixed(2) + ')');
          ctx.fill();
          ctx.strokeStyle = p.skip ? th.normal : th.tumor;
          ctx.lineWidth = 1.6;
          ctx.stroke();

          ctx.fillStyle = th.text;
          ctx.font = '600 9.5px monospace';
          ctx.fillText('B' + (i + 1), bx + bw / 2 - 7, cy - 4);
          ctx.fillStyle = th.muted;
          ctx.font = '8.5px monospace';
          ctx.fillText((localGrad * 100).toFixed(0) + '%', bx + bw / 2 - 9, cy + 12);

          // Identity skip connection arc (+x)
          if (p.skip) {
            var arcR = bw * 0.62;
            ctx.save();
            ctx.strokeStyle = th.emerald;
            ctx.lineWidth = 2.2;
            ctx.beginPath();
            ctx.arc(bx + bw / 2, cy - 28, arcR, Math.PI, 0, false);
            ctx.stroke();

            // Animated backprop gradient pulse travelling along the identity highway
            var pulseT = 1.0 - ((animPhase * 0.45 + i * 0.25) % 1.0);
            var ang = Math.PI * (1.0 - pulseT);
            var px = (bx + bw / 2) + Math.cos(ang) * arcR;
            var py = (cy - 28) - Math.sin(ang) * arcR;
            ctx.fillStyle = th.emerald;
            ctx.beginPath();
            ctx.arc(px, py, 3.5, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
          }
        }

        // Gradient retention meter at bottom
        var barX = 28, barY = 225, barW = w - 56, barH = 16;
        ctx.fillStyle = th.text;
        ctx.font = '600 11.5px sans-serif';
        ctx.fillText(
          'Early-Layer Backprop Gradient Retention (∂E / ∂x₁): ' + retention.toFixed(1) + '%',
          barX, barY - 8
        );

        roundRect(ctx, barX, barY, barW, barH, 8);
        ctx.fillStyle = th.surface2;
        ctx.fill();

        var fillW = Math.max(8, barW * (retention / 100));
        roundRect(ctx, barX, barY, fillW, barH, 8);
        ctx.fillStyle = p.skip ? th.emerald : th.tumor;
        ctx.fill();

        ctx.fillStyle = th.muted;
        ctx.font = '11px sans-serif';
        ctx.fillText(
          p.skip
            ? 'Identity shortcut (+I) preserves gradient magnitude all the way to 3x3 nuclear filters.'
            : 'WARNING: Without skip connections, ∂E/∂x₁ decays as ∏ Wₗ → 0 (early layers freeze!).',
          barX, barY + 34
        );

        return {
          metrics: '<span>Depth = ' + p.depth + ' Layers</span>' +
            '<span>Skip Highway = ' + (p.skip ? 'ACTIVE (+x)' : 'DISABLED') + '</span>' +
            '<span>Signal Retention = ' + retention.toFixed(1) + '%</span>' +
            '<span>Test Acc = 84.94% (ROC-AUC 0.9392)</span>',
          caption: 'Green pulses show error gradients flowing backwards from the loss to early layers. Toggle Skip Connections OFF to observe vanishing gradients.'
        };
      },
      eqs: '<div class="tex display">\\mathbf{y}_l = \\mathcal{F}(\\mathbf{x}_l, \\{W_l\\}) + \\mathbf{x}_l \\quad \\Longrightarrow \\quad \\mathbf{x}_L = \\mathbf{x}_l + \\sum_{i=l}^{L-1} \\mathcal{F}(\\mathbf{x}_i, W_i)</div>' +
        '<div class="tex display">\\frac{\\partial \\mathcal{E}}{\\partial \\mathbf{x}_l} = \\frac{\\partial \\mathcal{E}}{\\partial \\mathbf{x}_L} \\left( \\mathbf{I} + \\frac{\\partial}{\\partial \\mathbf{x}_l}\\sum_{i=l}^{L-1} \\mathcal{F}(\\mathbf{x}_i, W_i) \\right)</div>',
      theory: '<h4>ResNet-18 Breakthrough Over Handcrafted Features</h4>' +
        '<ul>' +
        '<li><strong>Spatial Hierarchy vs. Global Pooling:</strong> Rather than collapsing a patch into 59 pooled numbers, ResNet-18 learns <span class="tex">3\\times 3</span> convolutional filters for nuclear pleomorphism, mitotic figures, and stromal invasion.</li>' +
        '<li><strong>61.9% Fewer False Positives:</strong> Cuts false alarms from 339 (Blend) down to 129 on the 5,000 test patches (+5.18% accuracy leap).</li>' +
        '</ul>'
    },

    /* ================================================================
       9. EfficientNet-B0 (Compound Scaling & MBConv SE Attention)
       ================================================================ */
    effnet: {
      badge: 'Rank #2: Compound Scaled CNN (82.64% Test Acc, Best Sensitivity 78.80%)',
      badgeClass: 'badge-normal',
      state: { phi: 1.0, r: 4 },
      controls: function () {
        var s = MODELS_CFG.effnet.state;
        return '<div class="ctrl-row">' +
          '<div class="ctrl-label"><span>Compound Scaling Coefficient (&phi;)</span><span id="effVal1">&phi; = ' + s.phi.toFixed(1) + '</span></div>' +
          '<input type="range" id="effSlider1" min="0.5" max="2.5" step="0.25" value="' + s.phi + '">' +
          '</div>' +
          '<div class="ctrl-row">' +
          '<div class="ctrl-label"><span>Squeeze-and-Excitation Reduction Ratio (r)</span><span id="effVal2">r = ' + s.r + '</span></div>' +
          '<input type="range" id="effSlider2" min="2" max="16" step="2" value="' + s.r + '">' +
          '</div>';
      },
      init: function (update) {
        var s1 = $('#effSlider1'), s2 = $('#effSlider2');
        if (s1) s1.addEventListener('input', function () {
          MODELS_CFG.effnet.state.phi = parseFloat(s1.value);
          $('#effVal1').textContent = 'φ = ' + MODELS_CFG.effnet.state.phi.toFixed(2);
          update();
        });
        if (s2) s2.addEventListener('input', function () {
          MODELS_CFG.effnet.state.r = parseInt(s2.value, 10);
          $('#effVal2').textContent = 'r = ' + s2.value;
          update();
        });
      },
      render: function (ctx, w, h) {
        var th = getTheme();
        var p = MODELS_CFG.effnet.state;
        var dScale = Math.pow(1.2, p.phi);
        var wScale = Math.pow(1.1, p.phi);
        var rScale = Math.pow(1.15, p.phi);
        var flopsMult = Math.pow(2, p.phi);

        ctx.fillStyle = th.text;
        ctx.font = '600 11.5px sans-serif';
        ctx.fillText('MBConv6 Inverted Bottleneck + Squeeze-and-Excitation (SE) Gate', 22, 22);

        // Draw 5 MBConv stages
        var stages = [
          { t: '1x1 Expand', sub: '6C ch', col: th.normal },
          { t: '3x3 DWConv', sub: 'Spatial', col: th.normal },
          { t: 'SE Squeeze', sub: 'GAP (1x1)', col: th.gold },
          { t: 'SE Excite', sub: 'C/' + p.r + '→σ', col: th.gold },
          { t: '1x1 Project', sub: 'Linear C', col: th.accent }
        ];
        var cy = 74, boxW = 74, boxH = 48;
        for (var i = 0; i < stages.length; i++) {
          var bx = 22 + i * 86;
          roundRect(ctx, bx, cy - boxH / 2, boxW, boxH, 6);
          ctx.fillStyle = th.surface;
          ctx.fill();
          ctx.strokeStyle = stages[i].col;
          ctx.lineWidth = 1.6;
          ctx.stroke();

          ctx.fillStyle = th.text;
          ctx.font = '600 10px sans-serif';
          ctx.fillText(stages[i].t, bx + 8, cy - 3);
          ctx.fillStyle = th.muted;
          ctx.font = '9.5px monospace';
          ctx.fillText(stages[i].sub, bx + 10, cy + 13);

          if (i < stages.length - 1) {
            ctx.strokeStyle = th.axis;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(bx + boxW, cy);
            ctx.lineTo(bx + 86, cy);
            ctx.stroke();
          }
        }

        // Dynamic Channel Attention Bars s_c = sigma(W2 ReLU(W1 z))
        var chartY = 262, maxBarH = 92;
        ctx.fillStyle = th.text;
        ctx.font = '600 11px sans-serif';
        ctx.fillText('SE Channel Attention Gates s = σ(W₂ δ(W₁ GAP(X))) across 12 Feature Maps:', 22, 146);

        for (var c = 0; c < 12; c++) {
          var rawScore = Math.sin(c * 0.75 + p.phi * 1.4 + Math.sin(animPhase * 0.4) * 0.15) * (4.0 / Math.sqrt(p.r));
          var sWeight = 1.0 / (1.0 + Math.exp(-rawScore));
          var bh = Math.max(8, sWeight * maxBarH);
          var xPos = 24 + c * 34;

          // Background track
          roundRect(ctx, xPos, chartY - maxBarH, 22, maxBarH, 4);
          ctx.fillStyle = th.surface2;
          ctx.fill();

          // Active attention fill
          roundRect(ctx, xPos, chartY - bh, 22, bh, 4);
          ctx.fillStyle = sWeight > 0.62 ? th.emerald : (sWeight > 0.38 ? th.normal : th.faint);
          ctx.fill();

          ctx.fillStyle = th.muted;
          ctx.font = '9px monospace';
          ctx.fillText((sWeight).toFixed(2), xPos - 1, chartY + 13);
          ctx.fillText('C' + (c + 1), xPos + 3, chartY + 25);
        }

        return {
          metrics: '<span>Depth d = ' + dScale.toFixed(2) + '&times;</span>' +
            '<span>Width w = ' + wScale.toFixed(2) + '&times;</span>' +
            '<span>Res r = ' + rScale.toFixed(2) + '&times;</span>' +
            '<span>FLOPs &asymp; ' + flopsMult.toFixed(1) + '&times; (Sens 78.80%)</span>',
          caption: 'Green channels (s > 0.62) are amplified (e.g. chromatin texture maps); grey channels are suppressed via Squeeze-and-Excitation.'
        };
      },
      eqs: '<div class="tex display">d = \\alpha^\\phi, \\quad w = \\beta^\\phi, \\quad r = \\gamma^\\phi \\quad \\text{s.t.} \\quad \\alpha \\cdot \\beta^2 \\cdot \\gamma^2 \\approx 2, \\quad \\alpha, \\beta, \\gamma \\ge 1</div>' +
        '<div class="tex display">\\mathbf{s} = \\sigma\\left(W_2 \\, \\text{SiLU}(W_1 \\, \\text{GAP}(\\mathbf{X}))\\right) \\in (0,1)^C, \\quad \\tilde{\\mathbf{X}}_c = s_c \\cdot \\mathbf{X}_c</div>',
      theory: '<h4>EfficientNet-B0 &amp; Channel-Wise SE Recalibration</h4>' +
        '<ul>' +
        '<li><strong>Highest Sensitivity on the Leaderboard (78.80%):</strong> Evaluated on the central diagnostic region, catching 1,970 / 2,500 tumour patches (fewest false negatives = 530).</li>' +
        '<li><strong>Parameter Efficiency:</strong> Depthwise separable convolutions + SE attention achieve 82.64% test accuracy with only 5.3M parameters.</li>' +
        '</ul>'
    },

    /* ================================================================
       10. ResNet-50 (3-Stage Bottleneck Compression)
       ================================================================ */
    resnet50: {
      badge: 'Rank #3: Deep Bottleneck CNN (82.14% Test Acc, 0.9285 ROC-AUC)',
      badgeClass: 'badge-normal',
      state: { k: 4, C: 64 },
      controls: function () {
        var s = MODELS_CFG.resnet50.state;
        return '<div class="ctrl-row">' +
          '<div class="ctrl-label"><span>1&times;1 Bottleneck Compression Ratio (k)</span><span id="r50Val1">' + s.k + '&times; Squeeze</span></div>' +
          '<input type="range" id="r50Slider1" min="1" max="8" step="1" value="' + s.k + '">' +
          '</div>' +
          '<div class="ctrl-row">' +
          '<div class="ctrl-label"><span>Bottleneck Inner Channel Width (C)</span><span id="r50Val2">' + s.C + ' Channels (Out: ' + (s.C * 4) + ')</span></div>' +
          '<input type="range" id="r50Slider2" min="32" max="128" step="16" value="' + s.C + '">' +
          '</div>';
      },
      init: function (update) {
        var s1 = $('#r50Slider1'), s2 = $('#r50Slider2');
        if (s1) s1.addEventListener('input', function () {
          MODELS_CFG.resnet50.state.k = parseInt(s1.value, 10);
          $('#r50Val1').textContent = s1.value + '× Squeeze';
          update();
        });
        if (s2) s2.addEventListener('input', function () {
          MODELS_CFG.resnet50.state.C = parseInt(s2.value, 10);
          var c = MODELS_CFG.resnet50.state.C;
          $('#r50Val2').textContent = c + ' Channels (Out: ' + (c * 4) + ')';
          update();
        });
      },
      render: function (ctx, w, h) {
        var th = getTheme();
        var p = MODELS_CFG.resnet50.state;
        var outCh = p.C * 4;
        var midCh = Math.round(outCh / p.k);

        ctx.fillStyle = th.text;
        ctx.font = '600 11.5px sans-serif';
        ctx.fillText('ResNet-50 3-Stage Bottleneck Block vs. Standard 2-Layer Block', 22, 22);

        // Draw 3 bottleneck stages with height proportional to channel dimension
        var cy = 104;
        var stages = [
          { x: 45, w: 98, h: 64, title: '1x1 Reduce', ch: outCh + ' → ' + midCh + ' ch' },
          { x: 180, w: 98, h: Math.max(26, Math.round(64 / p.k)), title: '3x3 Spatial', ch: midCh + ' → ' + midCh + ' ch' },
          { x: 315, w: 98, h: 64, title: '1x1 Restore', ch: midCh + ' → ' + outCh + ' ch' }
        ];

        // Long identity skip arc over all 3 stages
        ctx.save();
        ctx.strokeStyle = th.purple;
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        ctx.moveTo(30, cy);
        ctx.lineTo(30, cy - 52);
        ctx.lineTo(428, cy - 52);
        ctx.lineTo(428, cy);
        ctx.stroke();
        ctx.fillStyle = th.purple;
        ctx.font = '600 10px monospace';
        ctx.fillText('Identity Shortcut (+x, ' + outCh + ' channels)', 140, cy - 58);
        ctx.restore();

        for (var i = 0; i < 3; i++) {
          var st = stages[i];
          roundRect(ctx, st.x, cy - st.h / 2, st.w, st.h, 6);
          ctx.fillStyle = i === 1 ? th.normalSoft : th.surface;
          ctx.fill();
          ctx.strokeStyle = i === 1 ? th.normal : th.purple;
          ctx.lineWidth = 1.8;
          ctx.stroke();

          ctx.fillStyle = th.text;
          ctx.font = '600 10.5px sans-serif';
          ctx.fillText(st.title, st.x + 14, cy - 2);
          ctx.fillStyle = th.muted;
          ctx.font = '9.5px monospace';
          ctx.fillText(st.ch, st.x + 10, cy + 12);

          if (i < 2) {
            ctx.strokeStyle = th.axis;
            ctx.lineWidth = 1.8;
            ctx.beginPath();
            ctx.moveTo(st.x + st.w, cy);
            ctx.lineTo(stages[i + 1].x, cy);
            ctx.stroke();
          }
        }

        // Exact relative FLOPs ratio: (2 * outCh * midCh + 9 * midCh^2) / (18 * outCh^2)
        // Since midCh = outCh / k => (2/k + 9/k^2) / 18
        var ratioPct = ((2.0 / p.k + 9.0 / (p.k * p.k)) / 18.0) * 100;
        var savingsFactor = (100 / ratioPct).toFixed(1);

        var barX = 32, barY = 205, maxW = w - 64;
        ctx.fillStyle = th.text;
        ctx.font = '600 11px sans-serif';
        ctx.fillText('Per-Block Multiply-Accumulate (FLOPs) Cost Comparison:', barX, barY - 10);

        // Bar 1: Standard 2-Layer 3x3 block (100%)
        roundRect(ctx, barX, barY, maxW, 22, 5);
        ctx.fillStyle = th.surface2;
        ctx.fill();
        ctx.fillStyle = th.muted;
        ctx.font = '600 10.5px monospace';
        ctx.fillText('Standard 2×(3x3) at ' + outCh + ' ch: 100.0% FLOPs', barX + 10, barY + 15);

        // Bar 2: 3-stage bottleneck
        var botW = Math.max(36, Math.min(maxW, (ratioPct / 100) * maxW));
        roundRect(ctx, barX, barY + 30, botW, 22, 5);
        ctx.fillStyle = th.purple;
        ctx.fill();
        ctx.fillStyle = th.text;
        ctx.font = '600 10.5px monospace';
        ctx.fillText(
          '3-Stage Bottleneck (k=' + p.k + '): ' + ratioPct.toFixed(1) + '% FLOPs (' + savingsFactor + '× faster)',
          barX + 10, barY + 45
        );

        return {
          metrics: '<span>Compression k = ' + p.k + '&times;</span>' +
            '<span>3x3 Width = ' + midCh + ' ch</span>' +
            '<span>FLOPs Cost = ' + ratioPct.toFixed(1) + '% (' + savingsFactor + '&times; savings)</span>' +
            '<span>Test Acc = 82.14%</span>',
          caption: '1x1 bottleneck convolutions squeeze channels before the expensive 3x3 spatial filter, cutting FLOPs by 4.2x at k=4.'
        };
      },
      eqs: '<div class="tex display">\\mathbf{x} \\in \\mathbb{R}^{4C} \\xrightarrow{1\\times 1} \\mathbb{R}^{4C/k} \\xrightarrow{3\\times 3} \\mathbb{R}^{4C/k} \\xrightarrow{1\\times 1} \\mathbb{R}^{4C} + \\mathbf{x}</div>' +
        '<div class="tex display">\\frac{\\text{FLOPs}_{\\text{bottleneck}}}{\\text{FLOPs}_{\\text{standard}}} = \\frac{2(4C)(4C/k) + 9(4C/k)^2}{2 \\cdot 9(4C)^2} = \\frac{2k + 9}{18k^2} \\;\\;\\underset{k=4}{=}\\;\\; \\frac{17}{72} \\approx 23.6\\%</div>',
      theory: '<h4>ResNet-50 Capacity vs. Overfitting on 40k Patches</h4>' +
        '<ul>' +
        '<li><strong>Why 18 Layers Beat 50 Layers Here:</strong> ResNet-50 (25.6M parameters) achieved 0.9285 ROC-AUC and 82.14% accuracy, trailing ResNet-18 (11.2M parameters, 84.94%).</li>' +
        '<li><strong>Takeaway:</strong> On 40,000 <span class="tex">96\\times 96</span> patches, the extra 14.4M parameters over-parameterise stain variations rather than improving generalisation.</li>' +
        '</ul>'
    },

    /* ================================================================
       11. Spatial Scope (Track A Full 96x96 vs Track B Core 32x32)
       ================================================================ */
    spatial: {
      badge: 'Preprocessing Ablation: Track A (+3.5% Acc) vs Track B',
      badgeClass: 'badge-normal',
      state: { W: 96, noise: 0.35 },
      controls: function () {
        var s = MODELS_CFG.spatial.state;
        return '<div class="ctrl-row">' +
          '<div class="ctrl-label"><span>Feature Pooling Window Size (W &times; W)</span><span id="spVal1">' + s.W + ' &times; ' + s.W + ' px</span></div>' +
          '<input type="range" id="spSlider1" min="32" max="96" step="4" value="' + s.W + '">' +
          '</div>' +
          '<div class="ctrl-row">' +
          '<div class="ctrl-label"><span>Outer Margin Tumour Invader Density</span><span id="spVal2">' + s.noise.toFixed(2) + '</span></div>' +
          '<input type="range" id="spSlider2" min="0.00" max="1.00" step="0.05" value="' + s.noise + '">' +
          '</div>' +
          '<div class="explorebtnrow">' +
          '<button type="button" class="btn primary" id="btnTrackA">Track A: Full 96&times;96 (9,216 px)</button>' +
          '<button type="button" class="btn" id="btnTrackB">Track B: Core 32&times;32 (1,024 px)</button>' +
          '</div>';
      },
      init: function (update) {
        var s1 = $('#spSlider1'), s2 = $('#spSlider2');
        var setW = function (val) {
          MODELS_CFG.spatial.state.W = val;
          if (s1) s1.value = val;
          $('#spVal1').textContent = val + ' × ' + val + ' px';
          update();
        };
        if (s1) s1.addEventListener('input', function () { setW(parseInt(s1.value, 10)); });
        if (s2) s2.addEventListener('input', function () {
          MODELS_CFG.spatial.state.noise = parseFloat(s2.value);
          $('#spVal2').textContent = MODELS_CFG.spatial.state.noise.toFixed(2);
          update();
        });
        var bA = $('#btnTrackA'), bB = $('#btnTrackB');
        if (bA) bA.addEventListener('click', function () { setW(96); });
        if (bB) bB.addEventListener('click', function () { setW(32); });
      },
      render: function (ctx, w, h) {
        var th = getTheme();
        var cx = 175, cy = h / 2;
        var p = MODELS_CFG.spatial.state;
        var scale = 2.2; // 96px * 2.2 = 211.2px on canvas
        var patchHalf = 48 * scale;
        var winHalf = (p.W / 2) * scale;
        var coreHalf = 16 * scale;

        // H&E Eosin pink tissue background
        roundRect(ctx, cx - patchHalf, cy - patchHalf, patchHalf * 2, patchHalf * 2, 6);
        ctx.fillStyle = '#f8d7e8';
        ctx.fill();
        ctx.strokeStyle = th.axis;
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Render cells & count margin invaders inside active pooling window WxW
        var marginInvadersCount = Math.round(p.noise * 42);
        var leakedCells = 0;

        SPATIAL_CELLS.forEach(function (cell, idx) {
          var sx = cx + cell.x * (scale / 2.0);
          var sy = cy + cell.y * (scale / 2.0);
          var inWin = Math.abs(sx - cx) <= winHalf && Math.abs(sy - cy) <= winHalf;

          if (cell.type === 'vacuole') {
            ctx.fillStyle = '#ffffff';
            ctx.strokeStyle = '#f3b6d4';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(sx, sy, cell.r, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
          } else if (cell.type === 'benign') {
            ctx.fillStyle = '#6b4c9a';
            ctx.beginPath();
            ctx.arc(sx, sy, cell.r, 0, Math.PI * 2);
            ctx.fill();
          } else if (cell.type === 'margin_tumor') {
            var activeIdx = idx - 95;
            if (activeIdx < marginInvadersCount) {
              if (inWin) leakedCells++;
              ctx.fillStyle = inWin ? '#be123c' : 'rgba(190, 18, 60, 0.35)';
              ctx.beginPath();
              ctx.arc(sx, sy, cell.r, 0, Math.PI * 2);
              ctx.fill();
            }
          }
        });

        // Dim area outside active pooling window WxW
        ctx.save();
        ctx.fillStyle = 'rgba(15, 23, 42, 0.38)';
        // Top, bottom, left, right masks inside patch
        ctx.fillRect(cx - patchHalf, cy - patchHalf, patchHalf * 2, patchHalf - winHalf);
        ctx.fillRect(cx - patchHalf, cy + winHalf, patchHalf * 2, patchHalf - winHalf);
        ctx.fillRect(cx - patchHalf, cy - winHalf, patchHalf - winHalf, winHalf * 2);
        ctx.fillRect(cx + winHalf, cy - winHalf, patchHalf - winHalf, winHalf * 2);
        ctx.restore();

        // Active Pooling Window (Blue solid box)
        ctx.strokeStyle = '#0284c7';
        ctx.lineWidth = 2.6;
        ctx.strokeRect(cx - winHalf, cy - winHalf, winHalf * 2, winHalf * 2);

        // Central 32x32 Ground-Truth Window (Red dashed box)
        ctx.save();
        ctx.strokeStyle = '#e11d48';
        ctx.lineWidth = 2.2;
        ctx.setLineDash([5, 4]);
        ctx.strokeRect(cx - coreHalf, cy - coreHalf, coreHalf * 2, coreHalf * 2);
        ctx.restore();

        // Right-side trade-off telemetry panel
        var rx = 310, ry = 36, rw = 134;
        var pixels = p.W * p.W;
        var statReliability = Math.min(100, ((pixels / 9216) * 100));
        var fpRisk = p.W <= 32 ? 0 : Math.min(100, ((p.W - 32) / 64) * p.noise * 100);

        ctx.fillStyle = th.text;
        ctx.font = '600 11px sans-serif';
        ctx.fillText('Spatial Trade-off', rx, ry);

        // Bar 1: Descriptor SNR / Sample Size
        ctx.fillStyle = th.muted;
        ctx.font = '10px sans-serif';
        ctx.fillText('Pooled Pixels: ' + pixels.toLocaleString(), rx, ry + 26);
        roundRect(ctx, rx, ry + 32, rw, 10, 4);
        ctx.fillStyle = th.surface2; ctx.fill();
        roundRect(ctx, rx, ry + 32, Math.max(8, rw * (statReliability / 100)), 10, 4);
        ctx.fillStyle = th.normal; ctx.fill();

        // Bar 2: Margin FP Leakage Risk
        ctx.fillStyle = th.muted;
        ctx.fillText('Margin FP Leak: ' + fpRisk.toFixed(1) + '%', rx, ry + 68);
        roundRect(ctx, rx, ry + 74, rw, 10, 4);
        ctx.fillStyle = th.surface2; ctx.fill();
        roundRect(ctx, rx, ry + 74, Math.max(4, rw * (fpRisk / 100)), 10, 4);
        ctx.fillStyle = th.tumor; ctx.fill();

        // Legend box
        ctx.fillStyle = th.text;
        ctx.font = '600 10px sans-serif';
        ctx.fillText('True Label y = 0 (Normal)', rx, ry + 114);
        ctx.fillStyle = th.muted;
        ctx.font = '9.5px sans-serif';
        ctx.fillText('Core has 0 tumour cells.', rx, ry + 130);
        ctx.fillText('Leaked margin cells: ' + leakedCells, rx, ry + 145);
        ctx.fillText('RBF SVM Track A: 82.14%', rx, ry + 168);
        ctx.fillText('RBF SVM Track B: 78.64%', rx, ry + 183);

        return {
          metrics: '<span>Window = ' + p.W + '&times;' + p.W + ' (' + pixels.toLocaleString() + ' px)</span>' +
            '<span>Leaked Margin Cells = ' + leakedCells + '</span>' +
            '<span>FP Contamination Risk = ' + fpRisk.toFixed(1) + '%</span>' +
            '<span>Track A Wins by +3.50%</span>',
          caption: 'Blue box: active pooling window (WxW). Red dashed box: central 32x32 ground-truth core. Dark crimson cells in the margin cause false positives when W > 32.'
        };
      },
      eqs: '<div class="tex display">\\text{Track A (Full Context): } \\mathcal{X}_A \\in \\mathbb{R}^{96 \\times 96 \\times 3} \\implies 9{,}216 \\text{ pixels}, \\quad \\sigma_{\\hat{\\mu}} \\propto \\frac{1}{\\sqrt{9216}}</div>' +
        '<div class="tex display">\\text{Track B (Strict Core): } \\mathcal{X}_B \\in \\mathbb{R}^{32 \\times 32 \\times 3} \\implies 1{,}024 \\text{ pixels}, \\quad \\sigma_{\\hat{\\mu}} = 3\\times \\text{noisier}</div>',
      theory: '<h4>Why Full Context (Track A) Beats Strict Cropping (Track B)</h4>' +
        '<ul>' +
        '<li><strong>3&times; Lower Estimator Standard Error:</strong> Pooling over 9,216 pixels instead of 1,024 cuts the statistical variance of GLCM, LBP, and colour moments by a factor of 9 (<span class="tex">3\\times</span> lower standard deviation).</li>' +
        '<li><strong>Empirical Verdict:</strong> Border-only tumour patches are rare (0.54% of PCam), so Track A outperforms Track B across every single model family (+3.50% on RBF SVM).</li>' +
        '</ul>'
    }
  };

  /* ------------------------------------------------------------------ */
  /* Core Render & State Dispatcher                                      */
  /* ------------------------------------------------------------------ */
  function updateExplorer() {
    var sel = $('#exploreModelSelect');
    if (!sel) return;
    var key = sel.value;
    var model = MODELS_CFG[key];
    if (!model) return;

    var badge = $('#exploreQuickBadge');
    if (badge) {
      badge.textContent = model.badge;
      badge.className = 'badge ' + (model.badgeClass || 'badge-normal');
    }

    var cv = $('#exploreCanvas');
    if (!cv) return;
    var ctx = cv.getContext('2d');
    var dpr = window.devicePixelRatio || 1;
    var w = 460, h = 320;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
    }

    var th = getTheme();
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = th.bg;
    ctx.fillRect(0, 0, w, h);

    var res = null;
    try {
      res = model.render(ctx, w, h);
    } catch (e) {
      console.error('Error rendering model ' + key + ':', e);
    }
    ctx.restore();

    if (res) {
      var metricsEl = $('#exploreMetrics');
      if (metricsEl) metricsEl.innerHTML = res.metrics || '';
      var capEl = $('#exploreCaption');
      if (capEl) capEl.textContent = res.caption || '';
    }
  }

  function setModel(key) {
    var model = MODELS_CFG[key];
    if (!model) return;

    var cv = $('#exploreCanvas');
    if (cv) {
      cv.style.cursor = key === 'pca' ? 'grab' : 'default';
    }

    var ctrlsEl = $('#exploreCtrls');
    if (ctrlsEl) {
      ctrlsEl.innerHTML = model.controls();
      if (model.init) {
        try {
          model.init(updateExplorer);
        } catch (e) {
          console.error('Error initializing controls for ' + key + ':', e);
        }
      }
    }

    var eqsEl = $('#exploreEqs');
    if (eqsEl) {
      eqsEl.innerHTML = '<h4>Mathematical Formulation</h4>' + model.eqs;
      renderLatex(eqsEl);
    }

    var theoryEl = $('#exploreTheory');
    if (theoryEl) {
      theoryEl.innerHTML = model.theory;
      renderLatex(theoryEl);
    }

    updateExplorer();
    syncAnimationLoop();
  }

  /* ------------------------------------------------------------------ */
  /* Interactive Canvas Pointer Dragging for PCA Eigenvector Rotation    */
  /* ------------------------------------------------------------------ */
  function initCanvasPointerEvents() {
    var cv = $('#exploreCanvas');
    if (!cv) return;

    var updatePcaAngleFromEvent = function (e) {
      var rect = cv.getBoundingClientRect();
      var scaleX = 460 / rect.width;
      var scaleY = 320 / rect.height;
      var clientX = e.touches && e.touches[0] ? e.touches[0].clientX : e.clientX;
      var clientY = e.touches && e.touches[0] ? e.touches[0].clientY : e.clientY;
      var x = (clientX - rect.left) * scaleX;
      var y = (clientY - rect.top) * scaleY;

      // Only respond when clicking in the left scatter plot region (x < 330)
      if (x > 335 && !MODELS_CFG.pca.state.dragging) return;

      var cx = 165, cy = 160;
      var dx = x - cx, dy = cy - y;
      var deg = Math.atan2(dy, dx) * 180 / Math.PI;
      if (deg < 0) deg += 180;
      deg = Math.max(0, Math.min(180, Math.round(deg)));

      MODELS_CFG.pca.state.angle = deg;
      var s1 = $('#expSlider1'), v1 = $('#expVal1');
      if (s1) s1.value = deg;
      if (v1) v1.textContent = deg + '°';
      updateExplorer();
    };

    cv.addEventListener('mousedown', function (e) {
      var sel = $('#exploreModelSelect');
      if (!sel || sel.value !== 'pca') return;
      MODELS_CFG.pca.state.dragging = true;
      cv.style.cursor = 'grabbing';
      updatePcaAngleFromEvent(e);
    });

    window.addEventListener('mousemove', function (e) {
      if (!MODELS_CFG.pca.state.dragging) return;
      updatePcaAngleFromEvent(e);
    });

    window.addEventListener('mouseup', function () {
      if (MODELS_CFG.pca.state.dragging) {
        MODELS_CFG.pca.state.dragging = false;
        cv.style.cursor = 'grab';
      }
    });

    cv.addEventListener('touchstart', function (e) {
      var sel = $('#exploreModelSelect');
      if (!sel || sel.value !== 'pca') return;
      MODELS_CFG.pca.state.dragging = true;
      updatePcaAngleFromEvent(e);
    }, { passive: true });

    cv.addEventListener('touchmove', function (e) {
      if (!MODELS_CFG.pca.state.dragging) return;
      updatePcaAngleFromEvent(e);
    }, { passive: true });

    cv.addEventListener('touchend', function () {
      MODELS_CFG.pca.state.dragging = false;
    });
  }

  /* ------------------------------------------------------------------ */
  /* Bootstrap & Event Listeners                                         */
  /* ------------------------------------------------------------------ */
  function init() {
    var sel = $('#exploreModelSelect');
    if (!sel) return;

    sel.addEventListener('change', function () {
      setModel(sel.value);
    });

    initCanvasPointerEvents();

    var themeBtn = $('#themeBtn');
    if (themeBtn) {
      themeBtn.addEventListener('click', function () {
        setTimeout(updateExplorer, 30);
      });
    }

    window.addEventListener('load', function () {
      var eqsEl = $('#exploreEqs');
      if (eqsEl) renderLatex(eqsEl);
      var theoryEl = $('#exploreTheory');
      if (theoryEl) renderLatex(theoryEl);
      updateExplorer();
    });

    setModel(sel.value || 'pca');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
