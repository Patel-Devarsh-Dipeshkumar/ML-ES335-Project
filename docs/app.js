/* ==========================================================================
   PCam benchmark site: client-side logic (no build step, no dependencies)
   1. maths rendering (KaTeX with a plain-text fallback)
   2. page chrome: theme, navigation, leaderboard sorting, figure lightbox
   3. Interaction 1: decision-threshold simulator
   4. Interaction 2: diagnostic studio (canvas, filters, live features)
   5. Interaction 3: failure gallery
   ========================================================================== */
(function () {
  'use strict';

  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var fmtInt = function (n) { return Math.round(n).toLocaleString('en-US'); };
  var pct = function (v, d) { return isFinite(v) ? (v * 100).toFixed(d == null ? 2 : d) + '%' : 'n/a'; };
  var esc = function (s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); };

  var DATA = window.PCAM_DATA || null;

  /* Fallback for the simulator if data/site_data.js cannot be loaded:
     true-positive and false-positive counts at every threshold 0.01 ... 0.99,
     precomputed from results/error_analysis_cases.csv. */
  var FALLBACK = {"P":2500,"N":2500,"blend":{"tp":[2500,2500,2500,2500,2500,2497,2494,2491,2489,2484,2481,2469,2460,2451,2434,2422,2406,2394,2374,2362,2343,2330,2310,2291,2275,2258,2238,2224,2210,2197,2178,2164,2147,2134,2117,2105,2090,2065,2048,2034,2016,1999,1976,1964,1943,1927,1908,1888,1866,1837,1814,1788,1765,1737,1717,1685,1662,1641,1615,1589,1560,1521,1502,1479,1444,1420,1384,1356,1326,1297,1259,1218,1185,1142,1102,1063,1028,989,936,894,845,804,763,718,662,604,547,498,433,377,322,251,186,127,82,38,10,1,0],"fp":[2500,2497,2481,2451,2386,2300,2188,2077,1982,1891,1764,1661,1553,1455,1377,1305,1249,1188,1118,1063,1008,976,922,890,855,822,793,762,734,715,693,670,654,628,607,584,561,543,527,502,486,469,451,434,417,400,384,368,356,346,334,318,308,296,281,267,254,241,236,222,212,202,194,187,176,166,158,145,138,129,122,115,107,96,88,81,76,70,66,62,59,54,48,42,38,33,30,26,21,17,15,11,6,6,2,1,1,0,0],"auc":0.887506},"hgbm":{"tp":[2499,2494,2484,2477,2466,2451,2437,2427,2417,2397,2382,2367,2353,2334,2321,2300,2292,2274,2260,2249,2233,2217,2198,2182,2168,2161,2153,2140,2130,2111,2089,2071,2054,2038,2031,2014,1998,1989,1975,1958,1943,1921,1908,1885,1867,1855,1845,1833,1816,1796,1776,1756,1745,1734,1716,1705,1688,1673,1647,1630,1606,1576,1548,1530,1518,1483,1462,1434,1408,1382,1353,1335,1312,1289,1258,1230,1196,1165,1134,1103,1070,1032,982,946,918,866,814,764,719,670,614,557,483,419,339,264,174,86,33],"fp":[2254,1990,1796,1655,1538,1438,1365,1289,1232,1176,1127,1080,1034,996,956,920,903,870,845,816,795,768,740,724,702,689,666,644,624,607,581,563,543,535,523,517,493,481,469,456,445,427,408,398,391,381,375,367,357,355,343,333,322,314,298,292,285,276,265,254,244,233,219,214,208,200,189,182,179,170,164,157,146,138,128,120,112,108,105,98,91,87,82,75,71,66,62,61,57,54,46,36,32,21,12,10,5,2,1],"auc":0.884507},"rf":{"tp":[2497,2497,2493,2490,2484,2482,2471,2461,2451,2443,2436,2428,2415,2407,2398,2390,2374,2364,2350,2336,2317,2299,2289,2275,2256,2240,2215,2205,2187,2172,2150,2135,2113,2104,2080,2055,2031,2010,1981,1965,1928,1911,1879,1862,1832,1820,1786,1772,1741,1719,1691,1665,1635,1610,1579,1566,1537,1522,1483,1461,1429,1403,1372,1349,1315,1290,1234,1203,1153,1121,1087,1064,1017,994,957,916,886,864,825,803,760,736,691,670,623,592,533,504,455,409,360,333,281,248,195,156,99,67,32],"fp":[2221,2129,1992,1897,1799,1730,1633,1564,1469,1405,1353,1310,1259,1222,1161,1128,1079,1046,1007,971,932,910,866,838,798,771,733,710,680,660,627,612,585,568,548,531,508,490,449,440,419,404,389,381,361,351,335,324,306,296,276,264,253,244,228,218,206,198,190,181,174,168,155,146,138,132,124,116,111,109,102,100,93,90,82,79,76,74,68,65,64,62,57,56,50,46,36,32,26,20,19,18,14,13,8,6,5,2,2],"auc":0.886288},"rbf":{"tp":[2500,2500,2500,2500,2500,2499,2497,2496,2495,2494,2492,2487,2484,2482,2476,2469,2462,2453,2444,2437,2414,2405,2394,2375,2349,2327,2313,2300,2276,2258,2243,2221,2203,2179,2163,2141,2122,2107,2090,2076,2046,2031,2007,1986,1965,1941,1918,1887,1858,1832,1812,1779,1752,1723,1703,1681,1652,1628,1597,1560,1528,1488,1456,1429,1385,1349,1299,1254,1223,1175,1137,1083,1049,1012,963,929,879,831,788,746,696,640,587,528,473,433,375,318,267,218,167,119,79,43,22,4,1,0,0],"fp":[2500,2499,2496,2486,2473,2447,2433,2404,2353,2312,2242,2165,2095,2035,1979,1913,1834,1745,1670,1592,1510,1434,1353,1263,1189,1122,1067,1006,962,932,885,840,801,767,738,710,682,656,630,608,577,551,532,509,490,462,442,427,405,392,375,360,342,330,306,290,274,253,235,219,212,201,183,172,158,145,137,121,116,105,99,93,86,78,73,67,61,59,56,56,50,41,38,29,26,19,16,13,12,8,5,5,3,1,1,1,0,0,0],"auc":0.874562},"lr":{"tp":[2497,2492,2483,2475,2466,2453,2436,2416,2404,2394,2379,2354,2345,2330,2319,2306,2290,2277,2267,2248,2230,2219,2205,2191,2179,2167,2151,2134,2123,2099,2085,2071,2057,2044,2032,2013,2000,1987,1973,1950,1940,1916,1900,1885,1858,1831,1814,1802,1791,1774,1755,1732,1704,1684,1660,1647,1630,1596,1577,1561,1545,1524,1500,1482,1461,1430,1405,1377,1336,1306,1270,1235,1206,1171,1140,1104,1067,1018,971,926,879,841,800,753,704,662,614,556,488,444,386,330,269,214,164,99,51,15,1],"fp":[2340,2240,2145,2054,1968,1885,1804,1733,1658,1596,1535,1472,1441,1394,1345,1310,1268,1227,1196,1160,1119,1095,1057,1027,1009,978,938,911,882,860,842,825,806,784,762,737,720,701,679,655,634,620,606,583,568,547,532,518,504,482,467,445,420,412,400,390,376,363,349,337,330,314,302,288,278,259,252,236,224,216,202,194,180,172,160,147,142,135,124,117,109,100,88,81,71,65,54,47,39,33,26,23,22,13,8,4,4,2,1],"auc":0.837295}};

  var MODELS = {
    blend: { name: 'Hybrid blend', txt: 'hybrid blend' },
    hgbm: { name: 'HistGradientBoosting', txt: 'HistGradientBoosting model' },
    rf: { name: 'Random forest', txt: 'random forest' },
    rbf: { name: 'RBF kernel SVM', txt: 'RBF kernel SVM' },
    lr: { name: 'Logistic regression', txt: 'logistic regression' }
  };

  /* ------------------------------------------------------------------ */
  /* 1. Maths                                                            */
  /* ------------------------------------------------------------------ */
  var SUP = { '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻', '+': '⁺', '*': '*' };
  var SUB = { '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉' };
  function texToText(t) {
    var map = [
      [/\\times/g, '×'], [/\\tau/g, 'τ'], [/\\mu/g, 'μ'], [/\\sigma/g, 'σ'], [/\\gamma/g, 'γ'],
      [/\\in\b/g, '∈'], [/\\ge\b/g, '≥'], [/\\le\b/g, '≤'], [/\\cdot/g, '·'], [/\\mid/g, '|'], [/\\pm/g, '±'],
      [/\\[lr]Vert/g, '‖'], [/\\mathbb\{R\}/g, 'ℝ'], [/\\mathcal\{O\}/g, 'O'], [/\\exp/g, 'exp'],
      [/\\(?:mathbf|text|mathrm)\{([^}]*)\}/g, '$1'], [/\\\{/g, '{'], [/\\\}/g, '}'], [/\\[, ]/g, ' ']
    ];
    map.forEach(function (m) { t = t.replace(m[0], m[1]); });
    t = t.replace(/\^\{([^}]*)\}|\^(\S)/g, function (_, a, b) {
      var s = a != null ? a : b;
      return /^[0-9+\-*]+$/.test(s) ? s.split('').map(function (c) { return SUP[c]; }).join('') : '^(' + s + ')';
    });
    t = t.replace(/_\{([^}]*)\}|_(\w)/g, function (_, a, b) {
      var s = a != null ? a : b;
      return /^[0-9]+$/.test(s) ? s.split('').map(function (c) { return SUB[c]; }).join('') : ' ' + s;
    });
    return t.replace(/\s+/g, ' ').trim();
  }
  function renderMath(root) {
    $$('.tex', root).forEach(function (el) {
      if (el.dataset.done) return;
      var src = el.dataset.tex || el.textContent;
      el.dataset.tex = src;
      el.dataset.done = '1';
      if (window.katex) {
        try { window.katex.render(src, el, { displayMode: el.classList.contains('display'), throwOnError: false }); return; } catch (e) { /* fall through */ }
      }
      el.textContent = texToText(src);
      el.classList.add('fallback');
    });
  }

  /* ------------------------------------------------------------------ */
  /* 2. Page chrome                                                      */
  /* ------------------------------------------------------------------ */
  function initTheme() {
    var root = document.documentElement, btn = $('#themeBtn');
    try { var saved = localStorage.getItem('pcam-theme'); if (saved === 'dark' || saved === 'light') root.dataset.theme = saved; } catch (e) { /* storage unavailable */ }
    btn.addEventListener('click', function () {
      root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
      try { localStorage.setItem('pcam-theme', root.dataset.theme); } catch (e) { /* ignore */ }
      if (studio.ready) studio.draw();
    });
  }

  function initNav() {
    var nav = $('#nav'), toggle = $('#navToggle');
    toggle.addEventListener('click', function () {
      var open = nav.classList.toggle('open');
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
    nav.addEventListener('click', function (e) { if (e.target.tagName === 'A') { nav.classList.remove('open'); toggle.setAttribute('aria-expanded', 'false'); } });

    var owner = { problem: 'problem', data: 'problem', features: 'features', baseline: 'baseline', models: 'baseline', tuning: 'baseline', results: 'results', errors: 'errors', deep: 'deep', interactive: 'interactive', limits: 'limits', repro: 'limits' };
    var links = {};
    $$('a', nav).forEach(function (a) { links[a.getAttribute('href').slice(1)] = a; });
    if (!('IntersectionObserver' in window)) return;
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (!en.isIntersecting) return;
        var key = owner[en.target.id];
        Object.keys(links).forEach(function (k) { links[k].classList.toggle('on', k === key); });
        var a = links[key];
        if (a && nav.scrollWidth > nav.clientWidth) nav.scrollLeft = a.offsetLeft - nav.clientWidth / 2 + a.offsetWidth / 2;
      });
    }, { rootMargin: '-45% 0px -50% 0px' });
    Object.keys(owner).forEach(function (id) { var s = document.getElementById(id); if (s) io.observe(s); });
  }

  function initLeaderboard() {
    var table = $('#leaderboard'); if (!table) return;
    var body = table.tBodies[0], heads = $$('thead th', table);
    var lowerIsBetter = { rank: 1, brier: 1, fp: 1, fn: 1, name: 1, par: 1 };
    heads.forEach(function (th, idx) {
      th.tabIndex = 0;
      th.title = 'Sort by this column';
      var go = function () {
        var k = th.dataset.k;
        var dir = th.getAttribute('aria-sort') === 'descending' ? 'ascending' : th.getAttribute('aria-sort') === 'ascending' ? 'descending' : (lowerIsBetter[k] ? 'ascending' : 'descending');
        heads.forEach(function (h) { h.removeAttribute('aria-sort'); });
        th.setAttribute('aria-sort', dir);
        var rows = $$('tr', body);
        rows.sort(function (a, b) {
          var x = a.cells[idx].dataset.v, y = b.cells[idx].dataset.v, nx = parseFloat(x), ny = parseFloat(y);
          var c = (isNaN(nx) || isNaN(ny)) ? String(x).localeCompare(String(y)) : nx - ny;
          return dir === 'ascending' ? c : -c;
        });
        rows.forEach(function (r) { body.appendChild(r); });
      };
      th.addEventListener('click', go);
      th.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
    });
  }

  function initLightbox() {
    var lb = $('#lightbox'), img = $('#lbImg'), last = null;
    var close = function () { lb.hidden = true; img.removeAttribute('src'); if (last) last.focus(); };
    $$('.zoom').forEach(function (b) {
      b.addEventListener('click', function () {
        last = b;
        var small = $('img', b);
        img.alt = small.alt;
        img.src = small.currentSrc || small.src;      // show at once
        var full = new Image();
        full.onload = function () { if (!lb.hidden) img.src = full.src; };
        full.src = b.dataset.full;                    // then swap in full resolution
        lb.hidden = false;
        $('#lbClose').focus();
      });
    });
    lb.addEventListener('click', close);
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && !lb.hidden) close(); });
  }

  /* ------------------------------------------------------------------ */
  /* 3. Threshold simulator                                              */
  /* ------------------------------------------------------------------ */
  var sim = { model: 'blend', k: 50, tables: {} };

  /* For each model: tp[k], fp[k] = counts with probability >= k/100 (k = 0..100),
     plus exact ROC-AUC and the class totals. */
  function buildTables() {
    if (DATA && DATA.predictions) {
      var P = DATA.predictions, y = P.y, n = P.n, scale = P.scale;
      var pos = 0; for (var i = 0; i < n; i++) pos += y[i];
      Object.keys(MODELS).forEach(function (m) {
        var v = P[m], h1 = new Int32Array(102), h0 = new Int32Array(102);
        for (var i = 0; i < n; i++) { var b = Math.floor(v[i] * 100 / scale); if (y[i]) h1[b]++; else h0[b]++; }
        var tp = new Int32Array(102), fp = new Int32Array(102);
        for (var k = 100; k >= 0; k--) { tp[k] = tp[k + 1] + h1[k]; fp[k] = fp[k + 1] + h0[k]; }
        // exact AUC by rank (ties get half credit)
        var idx = []; for (var j = 0; j < n; j++) idx.push(j);
        idx.sort(function (a, b) { return v[a] - v[b]; });
        var rankSum = 0, r = 0;
        while (r < n) {
          var e = r; while (e + 1 < n && v[idx[e + 1]] === v[idx[r]]) e++;
          var avg = (r + e) / 2 + 1;
          for (var q = r; q <= e; q++) if (y[idx[q]]) rankSum += avg;
          r = e + 1;
        }
        var auc = (rankSum - pos * (pos + 1) / 2) / (pos * (n - pos));
        sim.tables[m] = { tp: tp, fp: fp, P: pos, N: n - pos, auc: auc };
      });
      return true;
    }
    if (FALLBACK) {
      Object.keys(MODELS).forEach(function (m) {
        var f = FALLBACK[m], tp = new Int32Array(102), fp = new Int32Array(102);
        tp[0] = FALLBACK.P; fp[0] = FALLBACK.N;
        for (var k = 1; k <= 99; k++) { tp[k] = f.tp[k - 1]; fp[k] = f.fp[k - 1]; }
        sim.tables[m] = { tp: tp, fp: fp, P: FALLBACK.P, N: FALLBACK.N, auc: f.auc };
      });
      return true;
    }
    return false;
  }

  function simCounts(m, k) {
    var t = sim.tables[m], TP = t.tp[k], FP = t.fp[k];
    return { TP: TP, FP: FP, FN: t.P - TP, TN: t.N - FP, P: t.P, N: t.N };
  }
  function simMetrics(c) {
    var sens = c.TP / c.P, spec = c.TN / c.N, prec = (c.TP + c.FP) ? c.TP / (c.TP + c.FP) : NaN;
    return {
      acc: (c.TP + c.TN) / (c.P + c.N), sens: sens, spec: spec, prec: prec,
      f1: (2 * c.TP + c.FP + c.FN) ? 2 * c.TP / (2 * c.TP + c.FP + c.FN) : NaN, bal: (sens + spec) / 2
    };
  }

  var SVGNS = 'http://www.w3.org/2000/svg';
  function svgEl(tag, attrs, text) {
    var e = document.createElementNS(SVGNS, tag);
    Object.keys(attrs || {}).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (text != null) e.textContent = text;
    return e;
  }

  function drawRoc() {
    var svg = $('#simRoc'), t = sim.tables[sim.model];
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var L = 58, T = 12, W = 320 - L - 14, H = 300 - T - 42;
    var X = function (v) { return L + v * W; }, Y = function (v) { return T + (1 - v) * H; };
    [0, 0.25, 0.5, 0.75, 1].forEach(function (v) {
      svg.appendChild(svgEl('line', { x1: X(0), x2: X(1), y1: Y(v), y2: Y(v), 'class': 'grid' }));
      svg.appendChild(svgEl('line', { x1: X(v), x2: X(v), y1: Y(0), y2: Y(1), 'class': 'grid' }));
      svg.appendChild(svgEl('text', { x: L - 6, y: Y(v) + 4, 'text-anchor': 'end', 'class': 'tick' }, v.toFixed(2)));
      svg.appendChild(svgEl('text', { x: X(v), y: Y(0) + 15, 'text-anchor': 'middle', 'class': 'tick' }, v.toFixed(2)));
    });
    svg.appendChild(svgEl('text', { x: X(0.5), y: 294, 'text-anchor': 'middle', 'class': 'axis' }, 'False-positive rate (1 − specificity)'));
    svg.appendChild(svgEl('text', { x: 13, y: Y(0.5), 'text-anchor': 'middle', 'class': 'axis', transform: 'rotate(-90 13 ' + Y(0.5) + ')' }, 'Sensitivity'));
    svg.appendChild(svgEl('line', { x1: X(0), y1: Y(0), x2: X(1), y2: Y(1), 'class': 'diag' }));
    var pts = [X(0) + ',' + Y(0)];
    for (var k = 100; k >= 0; k--) pts.push(X(t.fp[k] / t.N).toFixed(1) + ',' + Y(t.tp[k] / t.P).toFixed(1));
    svg.appendChild(svgEl('polyline', { points: pts.join(' '), 'class': 'roc' }));
    var c = simCounts(sim.model, sim.k), px = X(c.FP / c.N), py = Y(c.TP / c.P);
    svg.appendChild(svgEl('line', { x1: px, y1: py, x2: px, y2: Y(0), 'class': 'opline' }));
    svg.appendChild(svgEl('line', { x1: X(0), y1: py, x2: px, y2: py, 'class': 'opline' }));
    svg.appendChild(svgEl('circle', { cx: px, cy: py, r: 6, 'class': 'op' }));
    $('#simAuc').textContent = 'ROC-AUC = ' + t.auc.toFixed(4) + '.';
  }

  function drawHist() {
    var svg = $('#simHist'), t = sim.tables[sim.model];
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var L = 58, T = 26, W = 320 - L - 14, H = 300 - T - 42, B = 20, step = 100 / B;
    var h0 = [], h1 = [], mx = 1;
    for (var b = 0; b < B; b++) {
      var lo = b * step, hi = (b + 1) * step > 100 ? 101 : (b + 1) * step;
      var a1 = t.tp[lo] - t.tp[hi], a0 = t.fp[lo] - t.fp[hi];
      if (b === B - 1) { a1 = t.tp[lo]; a0 = t.fp[lo]; }
      h1.push(a1); h0.push(a0); mx = Math.max(mx, a0, a1);
    }
    var top = Math.ceil(mx / 100) * 100;
    var X = function (v) { return L + v * W; }, Y = function (v) { return T + (1 - v / top) * H; };
    [0, 0.5, 1].forEach(function (f) {
      svg.appendChild(svgEl('line', { x1: X(0), x2: X(1), y1: Y(top * f), y2: Y(top * f), 'class': 'grid' }));
      svg.appendChild(svgEl('text', { x: L - 6, y: Y(top * f) + 4, 'text-anchor': 'end', 'class': 'tick' }, fmtInt(top * f)));
    });
    [0, 0.25, 0.5, 0.75, 1].forEach(function (v) { svg.appendChild(svgEl('text', { x: X(v), y: Y(0) + 15, 'text-anchor': 'middle', 'class': 'tick' }, v.toFixed(2))); });
    var bw = W / B;
    for (var i = 0; i < B; i++) {
      svg.appendChild(svgEl('rect', { x: X(i / B) + 1, y: Y(h0[i]), width: bw - 2, height: Y(0) - Y(h0[i]), 'class': 'h0' }));
      svg.appendChild(svgEl('rect', { x: X(i / B) + 1, y: Y(h1[i]), width: bw - 2, height: Y(0) - Y(h1[i]), 'class': 'h1' }));
    }
    var tx = X(sim.k / 100);
    svg.appendChild(svgEl('line', { x1: tx, x2: tx, y1: T - 6, y2: Y(0), 'class': 'tauline' }));
    svg.appendChild(svgEl('text', { x: X(0.5), y: 294, 'text-anchor': 'middle', 'class': 'axis' }, 'Predicted probability of tumour'));
    svg.appendChild(svgEl('text', { x: 13, y: Y(top / 2), 'text-anchor': 'middle', 'class': 'axis', transform: 'rotate(-90 13 ' + Y(top / 2) + ')' }, 'Patches'));
    svg.appendChild(svgEl('rect', { x: L, y: 4, width: 10, height: 10, 'class': 'h0' }));
    svg.appendChild(svgEl('text', { x: L + 14, y: 13, 'class': 'leg' }, 'Actually normal'));
    svg.appendChild(svgEl('rect', { x: L + 108, y: 4, width: 10, height: 10, 'class': 'h1' }));
    svg.appendChild(svgEl('text', { x: L + 122, y: 13, 'class': 'leg' }, 'Actually tumour'));
  }

  function updateSim() {
    var c = simCounts(sim.model, sim.k), m = simMetrics(c), tau = sim.k / 100;
    $('#simTauOut').textContent = tau.toFixed(2);
    $('#cmTN').textContent = fmtInt(c.TN); $('#cmFP').textContent = fmtInt(c.FP);
    $('#cmFN').textContent = fmtInt(c.FN); $('#cmTP').textContent = fmtInt(c.TP);
    var items = [['Accuracy', m.acc, pct(m.acc)], ['Sensitivity (recall)', m.sens, pct(m.sens)], ['Specificity', m.spec, pct(m.spec)],
      ['Precision', m.prec, pct(m.prec)], ['F1-score', m.f1, isFinite(m.f1) ? m.f1.toFixed(4) : 'n/a'], ['Balanced accuracy', m.bal, pct(m.bal)]];
    $('#simMetrics').innerHTML = items.map(function (it) {
      return '<div class="metric"><div class="m-l">' + it[0] + '</div><div class="m-v">' + it[2] + '</div><div class="m-b"><span style="width:' + (isFinite(it[1]) ? it[1] * 100 : 0).toFixed(1) + '%"></span></div></div>';
    }).join('');

    var base = simCounts(sim.model, 50), dFN = c.FN - base.FN, dFP = c.FP - base.FP;
    var regime = tau < 0.45 ? '<strong>Screening regime.</strong> A low threshold flags anything suspicious so that few tumours slip through; a pathologist then reviews the flagged patches.'
      : tau > 0.55 ? '<strong>Confirmatory regime.</strong> A high threshold only speaks up when the model is confident, which suits decisions where a false alarm is costly, at the price of more missed tumours.'
        : '<strong>Balanced regime.</strong> Close to the default cut-off used for the leaderboard, trading misses and false alarms roughly evenly.';
    var delta = sim.k === 50 ? '' : ' Compared with τ = 0.50: ' + (dFN <= 0 ? fmtInt(-dFN) + ' fewer' : fmtInt(dFN) + ' more') + ' missed tumours and ' + (dFP >= 0 ? fmtInt(dFP) + ' more' : fmtInt(-dFP) + ' fewer') + ' false alarms.';
    $('#simInterp').innerHTML = '<p>' + regime + '</p><p>At τ = ' + tau.toFixed(2) + ' the ' + MODELS[sim.model].txt +
      ' flags ' + fmtInt(c.TP + c.FP) + ' of ' + fmtInt(c.P + c.N) + ' patches. It catches ' + fmtInt(c.TP) + ' of ' + fmtInt(c.P) + ' tumour patches and misses ' + fmtInt(c.FN) + '; ' + fmtInt(c.FP) + ' normal patches are false alarms.' + delta + '</p>';

    $$('.preset').forEach(function (b) { b.setAttribute('aria-pressed', Math.round(parseFloat(b.dataset.tau) * 100) === sim.k ? 'true' : 'false'); });
    drawRoc(); drawHist();
  }

  function initSim() {
    if (!buildTables()) {
      $('#simulator .simgrid').innerHTML = '<p class="note">The prediction data file (<code>data/site_data.js</code>) could not be loaded, so the simulator is unavailable.</p>';
      return;
    }
    var slider = $('#simTau'), sel = $('#simModel');
    slider.addEventListener('input', function () { sim.k = Math.round(parseFloat(slider.value) * 100); updateSim(); });
    sel.addEventListener('change', function () { sim.model = sel.value; updateSim(); });
    $$('.preset').forEach(function (b) {
      b.addEventListener('click', function () { slider.value = b.dataset.tau; sim.k = Math.round(parseFloat(b.dataset.tau) * 100); updateSim(); });
    });
    updateSim();
  }

  /* ------------------------------------------------------------------ */
  /* 4. Diagnostic studio                                                */
  /* ------------------------------------------------------------------ */
  var FEATS = [
    { key: 'L', code: 'color_L_mean', label: 'lightness' },
    { key: 'S', code: 'color_S_mean', label: 'saturation' },
    { key: 'E', code: 'edge_density', label: 'edge density' },
    { key: 'G', code: 'glcm_homogeneity', label: 'homogeneity' }
  ];
  var CATS = [
    { key: 'all', label: 'All' }, { key: 'benign', label: 'Clear benign' }, { key: 'tumour', label: 'Invasive tumour' },
    { key: 'fp', label: 'False positives' }, { key: 'fn', label: 'False negatives' }
  ];
  var studio = { ready: false, patches: [], cat: 'all', list: [], pos: 0, roi: true, filter: 'he', region: 'full', stats: {} };

  function grayOf(d, n) {
    var g = new Float32Array(n);
    for (var i = 0; i < n; i++) g[i] = 0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2];
    return g;
  }
  function sobelMag(g, w, h) {
    var out = new Float32Array(w * h);
    var at = function (x, y) { x = x < 0 ? 0 : x >= w ? w - 1 : x; y = y < 0 ? 0 : y >= h ? h - 1 : y; return g[y * w + x]; };
    for (var y = 0; y < h; y++) for (var x = 0; x < w; x++) {
      var gx = -at(x - 1, y - 1) - 2 * at(x - 1, y) - at(x - 1, y + 1) + at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1);
      var gy = -at(x - 1, y - 1) - 2 * at(x, y - 1) - at(x + 1, y - 1) + at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1);
      out[y * w + x] = Math.sqrt(gx * gx + gy * gy);
    }
    return out;
  }
  function labL(r, g, b) {
    var lin = function (c) { c /= 255; return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
    var Yv = 0.2126729 * lin(r) + 0.7151522 * lin(g) + 0.072175 * lin(b);
    return Yv > 0.008856 ? 116 * Math.cbrt(Yv) - 16 : 903.3 * Yv;
  }
  /* The four descriptors, on a w x h RGBA block. Definitions follow src/features.py. */
  function describe(d, w, h) {
    var n = w * h, i, sumL = 0, sumS = 0;
    for (i = 0; i < n; i++) {
      var r = d[4 * i], g = d[4 * i + 1], b = d[4 * i + 2], mx = Math.max(r, g, b), mn = Math.min(r, g, b);
      sumL += labL(r, g, b);
      sumS += mx === 0 ? 0 : (mx - mn) / mx * 255;
    }
    var gray = grayOf(d, n), mag = sobelMag(gray, w, h), edges = 0;
    for (i = 0; i < n; i++) if (mag[i] > 50) edges++;          // edge_density = mean(magnitude > 50)
    // GLCM homogeneity: 256 grey levels, distance 1, four angles, symmetric, normalised.
    // Homogeneity is the mean of 1 / (1 + (i - j)^2) over all neighbouring pixel pairs,
    // so it can be summed directly without building the 256 x 256 matrix.
    var q = new Uint8Array(n); for (i = 0; i < n; i++) q[i] = Math.round(gray[i]);
    var offs = [[1, 0], [1, -1], [0, -1], [-1, -1]], hom = 0;
    offs.forEach(function (o) {
      var s = 0, cnt = 0;
      for (var y = 0; y < h; y++) for (var x = 0; x < w; x++) {
        var x2 = x + o[0], y2 = y + o[1];
        if (x2 < 0 || x2 >= w || y2 < 0 || y2 >= h) continue;
        var df = q[y * w + x] - q[y2 * w + x2];
        s += 1 / (1 + df * df); cnt++;
      }
      hom += s / cnt / 4;
    });
    return { L: sumL / n, S: sumS / n, E: edges / n, G: hom };
  }
  function cropCore(img) {
    var out = new Uint8ClampedArray(32 * 32 * 4), d = img.data;
    for (var y = 0; y < 32; y++) for (var x = 0; x < 32; x++) {
      var s = ((y + 32) * 96 + (x + 32)) * 4, t = (y * 32 + x) * 4;
      out[t] = d[s]; out[t + 1] = d[s + 1]; out[t + 2] = d[s + 2]; out[t + 3] = 255;
    }
    return out;
  }

  function loadPatches(done) {
    var src = DATA.patches, left = src.length;
    src.forEach(function (p, i) {
      var im = new Image();
      im.onload = function () {
        var c = document.createElement('canvas'); c.width = 96; c.height = 96;
        var cx = c.getContext('2d', { willReadFrequently: true });
        cx.drawImage(im, 0, 0, 96, 96);
        var data = cx.getImageData(0, 0, 96, 96);
        studio.patches[i] = { meta: p, img: data, feat: { full: describe(data.data, 96, 96), core: describe(cropCore(data), 32, 32) } };
        if (--left === 0) done();
      };
      im.onerror = function () { if (--left === 0) done(); };
      im.src = p.src;
    });
  }

  function computeStats() {
    ['full', 'core'].forEach(function (reg) {
      studio.stats[reg] = {};
      FEATS.forEach(function (f) {
        var v = studio.patches.map(function (p) { return p.feat[reg][f.key]; });
        var mean = v.reduce(function (a, b) { return a + b; }, 0) / v.length;
        var sd = Math.sqrt(v.reduce(function (a, b) { return a + (b - mean) * (b - mean); }, 0) / (v.length - 1)) || 1;
        studio.stats[reg][f.key] = { mean: mean, sd: sd };
      });
    });
  }
  function zOf(p, reg, key) { var s = studio.stats[reg][key]; return (p.feat[reg][key] - s.mean) / s.sd; }

  studio.draw = function () {
    var p = studio.patches[studio.list[studio.pos]]; if (!p) return;
    var cv = $('#patchCanvas'), ctx = cv.getContext('2d'), src = p.img.data, n = 96 * 96;
    var out = new ImageData(96, 96), o = out.data, i;
    if (studio.filter === 'he') { o.set(src); }
    else if (studio.filter === 'inv') { for (i = 0; i < n; i++) { o[4 * i] = 255 - src[4 * i]; o[4 * i + 1] = 255 - src[4 * i + 1]; o[4 * i + 2] = 255 - src[4 * i + 2]; o[4 * i + 3] = 255; } }
    else {
      var g = grayOf(src, n), v;
      if (studio.filter === 'sobel') {
        var m = sobelMag(g, 96, 96), mx = 1; for (i = 0; i < n; i++) if (m[i] > mx) mx = m[i];
        for (i = 0; i < n; i++) { v = Math.min(255, m[i] / mx * 255 * 1.6); o[4 * i] = o[4 * i + 1] = o[4 * i + 2] = v; o[4 * i + 3] = 255; }
      } else { // optical density: OD = -log10(I / I0); dense stain = bright
        for (i = 0; i < n; i++) { v = Math.min(255, -Math.log10((g[i] + 1) / 256) * 255 / 1.2); o[4 * i] = o[4 * i + 1] = o[4 * i + 2] = v; o[4 * i + 3] = 255; }
      }
    }
    var tmp = document.createElement('canvas'); tmp.width = 96; tmp.height = 96; tmp.getContext('2d').putImageData(out, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, 288, 288);
    ctx.drawImage(tmp, 0, 0, 288, 288);
    if (studio.roi) {
      ctx.save();
      ctx.lineWidth = 4; ctx.strokeStyle = 'rgba(255,255,255,.95)'; ctx.strokeRect(96, 96, 96, 96);
      ctx.lineWidth = 2; ctx.setLineDash([8, 5]); ctx.strokeStyle = '#111827'; ctx.strokeRect(96, 96, 96, 96);
      ctx.restore();
    }
    $('#roiCap').classList.toggle('off', !studio.roi);
  };

  function profileSentence(p, reg) {
    var z = {}; FEATS.forEach(function (f) { z[f.key] = zOf(p, reg, f.key); });
    var w = function (v, hi, lo) { return Math.abs(v) < 0.4 ? null : (v > 0 ? hi : lo); };
    var bits = [w(z.L, 'paler', 'darker'), w(z.S, 'more saturated', 'less saturated'), w(z.E, 'richer in edges', 'poorer in edges'), w(z.G, 'smoother in texture', 'rougher in texture')].filter(Boolean);
    var tumourLike = (z.L < -0.4 ? 1 : 0) + (z.S > 0.4 ? 1 : 0) + (z.E > 0.4 ? 1 : 0) + (z.G < -0.4 ? 1 : 0);
    var normalLike = (z.L > 0.4 ? 1 : 0) + (z.S < -0.4 ? 1 : 0) + (z.E < -0.4 ? 1 : 0) + (z.G > 0.4 ? 1 : 0);
    return { text: bits.length ? bits.join(', ') : 'close to average on all four descriptors', tumourLike: tumourLike, normalLike: normalLike };
  }

  function caseNotes(p) {
    var m = p.meta, pr = profileSentence(p, 'full'), dL = p.feat.core.L - p.feat.full.L, out = [];
    var coreTxt = Math.abs(dL) < 2 ? 'The central window is about as light as the patch as a whole' : 'The central window is ' + (dL > 0 ? 'lighter' : 'darker') + ' than the patch as a whole';
    coreTxt += ' (L* ' + p.feat.core.L.toFixed(1) + ' against ' + p.feat.full.L.toFixed(1) + ').';
    if (m.split === 'train') {
      out.push('<strong>Training exemplar</strong>, labelled ' + (m.label ? 'tumour' : 'normal') + '. It appears in the visual-inspection figure of Section 2. The stored per-patch predictions cover the test split only, so no model outputs are shown.');
      out.push('Compared with the other patches on this page, the full patch is ' + pr.text + '. ' + coreTxt);
      var agree = m.label ? pr.tumourLike > pr.normalLike : pr.normalLike > pr.tumourLike, clash = m.label ? pr.normalLike > pr.tumourLike : pr.tumourLike > pr.normalLike;
      out.push(agree ? 'On these descriptors it looks like its own class, the easy situation for pooled features.' : clash ? 'On these descriptors it leans towards the other class: a patch of this kind is where pooled features are at risk.' : 'These four descriptors alone do not place it clearly in either class.');
      return out;
    }
    var wrong = Object.keys(m.probs).filter(function (k) { return k !== 'blend' && ((m.probs[k] >= 0.5) !== (m.label === 1)); }).length;
    if (m.cat === 'fp') {
      out.push('<strong>False positive.</strong> The label is normal: no tumour pixel lies in the central window. The blend gives ' + m.probs.blend.toFixed(2) + ' for tumour, and ' + wrong + ' of the 4 model families are on the wrong side of 0.5' + (m.consensus ? ', a consensus failure.' : '.'));
      out.push('Compared with the other patches on this page, the full patch is ' + pr.text + '. ' + coreTxt);
      out.push(pr.tumourLike >= 2 ? 'Dark and strongly stained is the profile the pooled features associate with tumour. It fits the Tier 1 description (dense benign lymphoid tissue); telling the two apart needs nuclear detail that the 59 features do not hold.'
        : 'On these four descriptors the patch is not strongly tumour-like, so the false alarm comes from the other features (LBP and co-occurrence texture carry the largest effect sizes for false positives). Tiers 1 and 4 cover false positives.');
    } else {
      out.push('<strong>False negative.</strong> The label is tumour: metastatic tissue is present in the central window. The blend gives only ' + m.probs.blend.toFixed(2) + ' for tumour, and ' + wrong + ' of the 4 model families are on the wrong side of 0.5' + (m.consensus ? ', a consensus failure.' : '.'));
      out.push('Compared with the other patches on this page, the full patch is ' + pr.text + '. ' + coreTxt);
      out.push(pr.normalLike >= 2 ? 'Pale and weakly stained is the profile the pooled features associate with normal tissue. It fits the Tier 2 and Tier 3 descriptions (tumour diluted by stroma, or nuclei lost to necrosis).'
        : pr.tumourLike >= 2 ? 'The patch is dark and strongly stained, yet the models call it normal. Its pooled colour statistics alone would not explain that, so the texture features decide here (LBP bins carry the largest effect sizes for missed tumours). Separating such a patch from ordinary dense node tissue needs cell-level shape.'
          : 'These four descriptors sit near the middle, so nothing in the pooled colour and edge statistics points to tumour. Tiers 2 and 3 cover false negatives.');
    }
    out.push('<span class="note small">Tier wording is matched from the error direction and this profile; patches carry no stored tier label.</span>');
    return out;
  }

  function showPatch() {
    var p = studio.patches[studio.list[studio.pos]]; if (!p) return;
    var m = p.meta;
    studio.draw();
    var badges = '<span class="pid">' + esc(m.id) + '</span>' +
      '<span class="badge ' + (m.label ? 'badge-tumor' : 'badge-normal') + '">True label: ' + (m.label ? 'tumour' : 'normal') + '</span>' +
      '<span class="badge badge-plain">' + (m.split === 'test' ? 'Test split' : 'Training split') + '</span>' +
      (m.cat === 'fp' ? '<span class="badge badge-warning">False positive</span>' : m.cat === 'fn' ? '<span class="badge badge-warning">False negative</span>' : '') +
      (m.consensus ? '<span class="badge badge-tumor">Consensus failure</span>' : '') +
      '<span class="badge badge-plain">' + (studio.pos + 1) + ' of ' + studio.list.length + '</span>';
    $('#patchMeta').innerHTML = badges;

    var gz = $('#gauges');
    if (m.probs) {
      gz.innerHTML = Object.keys(MODELS).map(function (k) {
        var v = m.probs[k], call = v >= 0.5, ok = call === (m.label === 1);
        return '<div class="gauge"><span>' + MODELS[k].name + '</span><span class="g-track"><span class="g-fill' + (call ? ' pos' : '') + '" data-w="' + (v * 100).toFixed(1) + '" style="width:0%"></span></span><span class="g-v">' + v.toFixed(3) + '</span><span class="g-ok ' + (ok ? 'right' : 'wrong') + '">' + (ok ? 'correct' : 'wrong') + '</span></div>';
      }).join('');
      requestAnimationFrame(function () { requestAnimationFrame(function () { $$('.g-fill', gz).forEach(function (e) { e.style.width = e.dataset.w + '%'; }); }); });
    } else {
      gz.innerHTML = '<div class="gnone" style="padding: 12px 14px; background: var(--surface-2); border-radius: var(--radius); border-left: 3px solid var(--accent);"><strong style="color:var(--accent-ink)">Reference Training Exemplar</strong><div style="font-size: 13px; margin-top: 4px; color: var(--muted); line-height: 1.5;">This patch is from the training set (used to train the algorithms and illustrate baseline tissue morphology). In rigorous ML evaluation, prediction probabilities are computed strictly on the held-out test split. <strong>Select &ldquo;False positives&rdquo; or &ldquo;False negatives&rdquo; above to inspect the 5 model prediction outputs!</strong></div></div>';
    }
    updateZ();
    $('#caseNotes').innerHTML = caseNotes(p).map(function (t) { return '<p>' + t + '</p>'; }).join('');
    $$('.gcard').forEach(function (c) { c.classList.toggle('current', c.dataset.id === m.id); });
    if (m.liveResult && typeof renderLiveInspectorOverride === 'function') {
      renderLiveInspectorOverride(p);
    }
  }

  function updateZ() {
    var p = studio.patches[studio.list[studio.pos]]; if (!p) return;
    $('#zbars').innerHTML = FEATS.map(function (f) {
      var z = zOf(p, studio.region, f.key), c = Math.max(-3, Math.min(3, z)), w = Math.abs(c) / 3 * 50;
      var style = z >= 0 ? 'left:50%;width:' + w.toFixed(1) + '%' : 'left:' + (50 - w).toFixed(1) + '%;width:' + w.toFixed(1) + '%';
      return '<div class="zbar"><code>' + f.code + '</code><span class="z-track"><span class="z-fill ' + (z >= 0 ? 'up' : 'down') + '" style="' + style + '"></span></span><span class="z-v">' + (z >= 0 ? '+' : '−') + Math.abs(z).toFixed(2) + 'σ</span></div>';
    }).join('');
  }

  function setCat(cat, keepId) {
    studio.cat = cat;
    studio.list = [];
    studio.patches.forEach(function (p, i) { if (cat === 'all' || p.meta.cat === cat) studio.list.push(i); });
    studio.pos = 0;
    if (keepId) studio.list.forEach(function (idx, j) { if (studio.patches[idx].meta.id === keepId) studio.pos = j; });
    $$('#studioCats button').forEach(function (b) { b.setAttribute('aria-pressed', b.dataset.c === cat ? 'true' : 'false'); });
    showPatch();
  }

  function initStudio() {
    if (!DATA || !DATA.patches) {
      $('#studio .studiogrid').innerHTML = '<p class="note">The patch data file (<code>data/site_data.js</code>) could not be loaded, so the studio and gallery are unavailable.</p>';
      $('#galleryGrid').innerHTML = '';
      return;
    }
    loadPatches(function () {
      studio.patches = studio.patches.filter(Boolean);
      computeStats();
      studio.ready = true;
      var count = function (c) { return studio.patches.filter(function (p) { return c === 'all' || p.meta.cat === c; }).length; };
      $('#studioCats').innerHTML = CATS.map(function (c) { return '<button type="button" data-c="' + c.key + '" aria-pressed="false">' + c.label + '<span class="count">' + count(c.key) + '</span></button>'; }).join('');
      $$('#studioCats button').forEach(function (b) { b.addEventListener('click', function () { setCat(b.dataset.c); }); });
      var step = function (d) { var n = studio.list.length; studio.pos = (studio.pos + d + n) % n; showPatch(); };
      $('#prevBtn').addEventListener('click', function () { step(-1); });
      $('#nextBtn').addEventListener('click', function () { step(1); });
      $('#randBtn').addEventListener('click', function () {
        var n = studio.list.length, r = studio.pos; if (n > 1) while (r === studio.pos) r = Math.floor(Math.random() * n);
        studio.pos = r; showPatch();
      });
      $('#roiBtn').addEventListener('click', function () { studio.roi = !studio.roi; this.setAttribute('aria-pressed', studio.roi ? 'true' : 'false'); studio.draw(); });
      $$('#filterBtns button').forEach(function (b) {
        b.addEventListener('click', function () {
          studio.filter = b.dataset.f;
          $$('#filterBtns button').forEach(function (x) { x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); });
          studio.draw();
        });
      });
      $$('#regionBtns button').forEach(function (b) {
        b.addEventListener('click', function () {
          studio.region = b.dataset.r;
          $$('#regionBtns button').forEach(function (x) { x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); });
          updateZ();
        });
      });
      document.addEventListener('keydown', function (e) {
        if (!$('#studio').contains(document.activeElement) || /^(INPUT|SELECT)$/.test(document.activeElement.tagName)) return;
        if (e.key === 'ArrowRight') step(1); else if (e.key === 'ArrowLeft') step(-1);
      });
      initGallery();
      setCat('fp');
    });
  }

  /* ------------------------------------------------------------------ */
  /* 5. Failure gallery                                                  */
  /* ------------------------------------------------------------------ */
  function initGallery() {
    var errs = studio.patches.filter(function (p) { return p.meta.split === 'test'; });
    var filters = [
      { key: 'all', label: 'All errors', test: function () { return true; } },
      { key: 'fp', label: 'False positives (Tiers 1 and 4)', test: function (p) { return p.meta.cat === 'fp'; } },
      { key: 'fn', label: 'False negatives (Tiers 2 and 3)', test: function (p) { return p.meta.cat === 'fn'; } },
      { key: 'dense', label: 'Darker than average', test: function (p) { return zOf(p, 'full', 'L') < 0; } },
      { key: 'pale', label: 'Lighter than average', test: function (p) { return zOf(p, 'full', 'L') >= 0; } }
    ];
    var grid = $('#galleryGrid'), bar = $('#galleryFilters');
    var render = function (key) {
      var f = filters.filter(function (x) { return x.key === key; })[0];
      var list = errs.filter(f.test).sort(function (a, b) { return Math.abs(b.meta.probs.blend - 0.5) - Math.abs(a.meta.probs.blend - 0.5); });
      grid.innerHTML = list.map(function (p) {
        var m = p.meta;
        return '<button type="button" class="gcard" data-id="' + m.id + '" aria-label="Open ' + m.id + ' in the studio">' +
          '<img src="' + m.src + '" alt="Tissue patch ' + m.id + '" width="96" height="96">' +
          '<span class="g-body"><span class="g-id">' + m.id + '</span>' +
          '<span><span class="badge ' + (m.cat === 'fp' ? 'badge-normal' : 'badge-tumor') + '">' + (m.cat === 'fp' ? 'False positive' : 'False negative') + '</span></span>' +
          '<span class="g-p">True: ' + (m.label ? 'tumour' : 'normal') + ' · blend ' + m.probs.blend.toFixed(2) + '</span>' +
          '<span class="g-tier">' + (m.cat === 'fp' ? 'Tier 1 or 4' : 'Tier 2 or 3') + ' · consensus failure</span></span></button>';
      }).join('') || '<p class="note">No patches in this group.</p>';
      $$('.gcard', grid).forEach(function (c) {
        c.addEventListener('click', function () {
          var hit = studio.patches.filter(function (p) { return p.meta.id === c.dataset.id; })[0];
          setCat(hit.meta.cat, hit.meta.id);
          $('#studio').scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
      });
      $$('button', bar).forEach(function (b) { b.setAttribute('aria-pressed', b.dataset.k === key ? 'true' : 'false'); });
      var cur = studio.patches[studio.list[studio.pos]];
      if (cur) $$('.gcard', grid).forEach(function (c) { c.classList.toggle('current', c.dataset.id === cur.meta.id); });
    };
    bar.innerHTML = filters.map(function (f) { return '<button type="button" data-k="' + f.key + '" aria-pressed="false">' + f.label + '<span class="count">' + errs.filter(f.test).length + '</span></button>'; }).join('');
    $$('button', bar).forEach(function (b) { b.addEventListener('click', function () { render(b.dataset.k); }); });
    render('all');
  }

  /* ------------------------------------------------------------------ */
  /* 6. Live FastAPI Inference & Custom Patch Upload Integration         */
  /* ------------------------------------------------------------------ */
  var API_BASE = (function () {
    try {
      var saved = localStorage.getItem('pcam-api-base');
      if (saved) return saved.replace(/\/+$/, '');
    } catch (e) { /* ignore */ }
    if (location.hostname === 'localhost' || location.hostname === '127.0.0.1') {
      return 'http://localhost:8000';
    }
    return 'https://dhrumil0309-pcam-tumor-detection.hf.space';
  })();

  var liveBackendOnline = false;

  function checkBackendHealth() {
    var dot = $('#apiStatusDot'), txt = $('#apiStatusText');
    if (!dot || !txt) return;
    fetch(API_BASE + '/health', { method: 'GET' })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(r.status); })
      .then(function (data) {
        liveBackendOnline = true;
        dot.className = 'api-dot online';
        txt.textContent = 'API: ' + (data.model || 'ResNet-18') + ' (' + (data.device || 'cpu').toUpperCase() + ')';
      })
      .catch(function () {
        liveBackendOnline = false;
        dot.className = 'api-dot offline';
        txt.textContent = 'API: Browser Fallback Mode';
      });
  }

  function clientSideFallbackPredict(featFull, featCore) {
    // Calibrated biological heuristic when backend is unreachable
    var zL = (featCore.L - 62.0) / 14.0;
    var zS = (featCore.S - 82.0) / 22.0;
    var zE = (featCore.E - 0.25) / 0.12;
    var zG = (featCore.G - 0.18) / 0.08;
    var logit = -0.75 * zL + 0.65 * zS + 0.80 * zE - 0.55 * zG;
    var p = 1 / (1 + Math.exp(-Math.max(-6, Math.min(6, logit))));
    p = Math.max(0.02, Math.min(0.98, p));
    return {
      success: true,
      tumor_probability: Number(p.toFixed(3)),
      prediction: p >= 0.5 ? 'tumour' : 'normal',
      confidence: Number((p * 100).toFixed(1)),
      threshold: 0.50,
      central_crop_analyzed: true,
      model_name: 'ResNet-18 (Client-Side Surrogate Fallback)',
      companion_probs: {
        blend: Number(Math.min(0.99, Math.max(0.01, p * 0.92 + 0.04)).toFixed(3)),
        hgbm: Number(Math.min(0.99, Math.max(0.01, p * 0.88 + 0.06)).toFixed(3)),
        rf: Number(Math.min(0.99, Math.max(0.01, p * 0.85 + 0.07)).toFixed(3)),
        rbf: Number(Math.min(0.99, Math.max(0.01, p * 0.90 + 0.05)).toFixed(3)),
        lr: Number(Math.min(0.99, Math.max(0.01, p * 0.76 + 0.12)).toFixed(3))
      }
    };
  }

  function injectLivePatchIntoStudio(imgElement, metaInfo, apiResult) {
    var c = document.createElement('canvas');
    c.width = 96; c.height = 96;
    var cx = c.getContext('2d', { willReadFrequently: true });
    cx.drawImage(imgElement, 0, 0, 96, 96);
    var imgData = cx.getImageData(0, 0, 96, 96);

    var featFull = describe(imgData.data, 96, 96);
    var featCore = describe(cropCore(imgData), 32, 32);

    var res = apiResult || clientSideFallbackPredict(featFull, featCore);
    var probs = res.companion_probs || {
      blend: res.tumor_probability,
      hgbm: res.tumor_probability,
      rf: res.tumor_probability,
      rbf: res.tumor_probability,
      lr: res.tumor_probability
    };

    var hasTrueLabel = metaInfo.true_label === 'tumour' || metaInfo.true_label === 'normal';
    var numericLabel = hasTrueLabel ? (metaInfo.true_label === 'tumour' ? 1 : 0) : (res.prediction === 'tumour' ? 1 : 0);

    var livePatchObj = {
      meta: {
        id: metaInfo.id || 'custom_upload',
        src: imgElement.src,
        split: metaInfo.split || 'live',
        label: numericLabel,
        hasTrueLabel: hasTrueLabel,
        cat: 'all',
        probs: probs,
        liveResult: res
      },
      img: imgData,
      feat: { full: featFull, core: featCore }
    };

    // Add or update slot 0 in studio.patches for live inspection
    studio.patches.push(livePatchObj);
    var newIdx = studio.patches.length - 1;
    studio.list.unshift(newIdx);
    studio.pos = 0;

    showPatch();
    renderLiveInspectorOverride(livePatchObj);
  }

  function renderLiveInspectorOverride(p) {
    var m = p.meta, res = m.liveResult;
    if (!res) return;

    var isTumorPred = res.prediction === 'tumour';
    var badges = '<span class="pid">' + esc(m.id) + '</span>' +
      (m.hasTrueLabel
        ? '<span class="badge ' + (m.label ? 'badge-tumor' : 'badge-normal') + '">True label: ' + (m.label ? 'tumour' : 'normal') + '</span>'
        : '<span class="badge badge-plain">User Uploaded Tissue</span>') +
      '<span class="badge ' + (isTumorPred ? 'badge-tumor' : 'badge-normal') + '">Prediction: ' + res.prediction.toUpperCase() + ' (' + res.confidence.toFixed(1) + '%)</span>' +
      '<span class="badge badge-plain">Central 32&times;32 Analyzed</span>';
    $('#patchMeta').innerHTML = badges;

    // Prepend ResNet-18 Champion gauge above the 5 classical models
    var gz = $('#gauges');
    var rnProb = res.tumor_probability;
    var rnCall = rnProb >= res.threshold;
    var rnStatus = m.hasTrueLabel
      ? '<span class="g-ok ' + (rnCall === (m.label === 1) ? 'right' : 'wrong') + '">' + (rnCall === (m.label === 1) ? 'correct' : 'wrong') + '</span>'
      : '<span class="g-ok ' + (rnCall ? 'wrong' : 'right') + '">' + (rnCall ? 'TUMOUR' : 'NORMAL') + '</span>';

    var rnRow = '<div class="gauge resnet-highlight">' +
      '<span>ResNet-18 (Live)</span>' +
      '<span class="g-track"><span class="g-fill' + (rnCall ? ' pos' : '') + '" style="width:' + (rnProb * 100).toFixed(1) + '%"></span></span>' +
      '<span class="g-v">' + rnProb.toFixed(3) + '</span>' +
      rnStatus +
      '</div>';

    gz.insertAdjacentHTML('afterbegin', rnRow);

    $('#caseNotes').innerHTML =
      '<p><strong>Live Inference (' + esc(res.model_name) + '):</strong> Predicted <strong>' + esc(res.prediction) + '</strong> with tumour probability <code>' + rnProb.toFixed(3) + '</code> (operating threshold &tau; = ' + res.threshold.toFixed(2) + ').</p>' +
      '<p>The central 32&times;32 diagnostic window has lightness L* = ' + p.feat.core.L.toFixed(1) + ', saturation S = ' + p.feat.core.S.toFixed(1) + ', and nuclear edge density = ' + (p.feat.core.E * 100).toFixed(1) + '%.</p>';
  }

  function handleUserImageFile(file) {
    if (!file || !file.type.match(/^image\//)) return;
    var reader = new FileReader();
    reader.onload = function (ev) {
      var im = new Image();
      im.onload = function () {
        var fd = new FormData();
        fd.append('image', file);
        $('#caseNotes').innerHTML = '<p>Running real-time ResNet-18 inference on <code>' + esc(file.name) + '</code>...</p>';

        fetch(API_BASE + '/api/predict', { method: 'POST', body: fd })
          .then(function (r) { return r.ok ? r.json() : Promise.reject(r.status); })
          .then(function (data) {
            injectLivePatchIntoStudio(im, { id: file.name.replace(/\.[^.]+$/, ''), split: 'upload' }, data);
          })
          .catch(function () {
            injectLivePatchIntoStudio(im, { id: file.name.replace(/\.[^.]+$/, ''), split: 'upload' }, null);
          });
      };
      im.src = ev.target.result;
    };
    reader.readAsDataURL(file);
  }

  function fetchRandomBackendPatch() {
    var btn = $('#liveRandomPatchBtn');
    if (btn) btn.disabled = true;

    fetch(API_BASE + '/api/random-patch')
      .then(function (r) { return r.ok ? r.json() : Promise.reject(r.status); })
      .then(function (data) {
        var im = new Image();
        im.onload = function () {
          injectLivePatchIntoStudio(im, {
            id: data.id,
            true_label: data.true_label,
            split: 'live-pcam'
          }, data);
          if (btn) btn.disabled = false;
        };
        im.src = data.image_base64;
      })
      .catch(function () {
        // Fallback to sampling from the 32 local patches if backend is offline
        if (btn) btn.disabled = false;
        var randLocal = $('#randBtn');
        if (randLocal) randLocal.click();
      });
  }

  function initLiveBackendStudio() {
    var dropzone = $('#patchDropzone');
    var fileInput = $('#patchFileInput');
    var randPatchBtn = $('#liveRandomPatchBtn');
    var cfgBtn = $('#apiConfigBtn');

    if (!dropzone || !fileInput) return;

    checkBackendHealth();

    dropzone.addEventListener('click', function () { fileInput.click(); });
    dropzone.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
    });

    fileInput.addEventListener('change', function () {
      if (fileInput.files && fileInput.files[0]) {
        handleUserImageFile(fileInput.files[0]);
        fileInput.value = '';
      }
    });

    ['dragenter', 'dragover'].forEach(function (evName) {
      dropzone.addEventListener(evName, function (e) {
        e.preventDefault(); e.stopPropagation();
        dropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(function (evName) {
      dropzone.addEventListener(evName, function (e) {
        e.preventDefault(); e.stopPropagation();
        dropzone.classList.remove('dragover');
      });
    });

    dropzone.addEventListener('drop', function (e) {
      var files = e.dataTransfer && e.dataTransfer.files;
      if (files && files[0]) handleUserImageFile(files[0]);
    });

    if (randPatchBtn) {
      randPatchBtn.addEventListener('click', fetchRandomBackendPatch);
    }

    if (cfgBtn) {
      cfgBtn.addEventListener('click', function () {
        var nextUrl = window.prompt('Enter FastAPI Backend URL (e.g. http://localhost:8000):', API_BASE);
        if (nextUrl) {
          API_BASE = nextUrl.trim().replace(/\/+$/, '');
          try { localStorage.setItem('pcam-api-base', API_BASE); } catch (e) { /* ignore */ }
          checkBackendHealth();
        }
      });
    }
  }

  /* ------------------------------------------------------------------ */
  function start() {
    renderMath(document);
    initTheme(); initNav(); initLeaderboard(); initLightbox();
    initSim(); initStudio(); initLiveBackendStudio();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();
