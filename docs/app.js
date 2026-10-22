/**
 * PatchCamelyon (PCam) Diagnostic Showcase - Application Engine
 * Course: ES 335 - Machine Learning | IIT Gandhinagar
 * Authors: Devarsh Patel (24110245) & Dhrumil Bhutaiya (24110075)
 */

// Application State
const state = {
  curatedPatches: [],
  testPredictions: [],
  currentPatchIdx: 0,
  activeFilter: 'all',
  canvasFilter: 'none',
  showROI: true,
  simThreshold: 0.50,
  simModel: 'b' // 'b'=blend, 'h'=histgbm, 'f'=rf, 'r'=rbf, 'l'=logreg
};

// DOM Elements
const elements = {
  tabs: document.querySelectorAll('.tab-btn'),
  panes: document.querySelectorAll('.tab-pane'),
  patchCanvas: document.getElementById('patchCanvas'),
  currentPatchLabel: document.getElementById('currentPatchLabel'),
  patchIdTitle: document.getElementById('patchIdTitle'),
  patchCategoryBadge: document.getElementById('patchCategoryBadge'),
  consensusBadge: document.getElementById('consensusBadge'),
  riskAlertBox: document.getElementById('riskAlertBox'),
  clinicalNoteText: document.getElementById('clinicalNoteText'),
  featureBarContainer: document.getElementById('featureBarContainer'),
  btnTheme: document.getElementById('btnTheme'),
  themeIcon: document.getElementById('themeIcon'),
  
  // Model meters
  meterBlend: document.getElementById('meterBlend'),
  meterHgbm: document.getElementById('meterHgbm'),
  meterRf: document.getElementById('meterRf'),
  meterRbf: document.getElementById('meterRbf'),
  meterLr: document.getElementById('meterLr'),
  
  valBlend: document.getElementById('valBlend'),
  valHgbm: document.getElementById('valHgbm'),
  valRf: document.getElementById('valRf'),
  valRbf: document.getElementById('valRbf'),
  valLr: document.getElementById('valLr'),
  
  // Simulator
  sliderThreshold: document.getElementById('sliderThreshold'),
  thresholdVal: document.getElementById('thresholdVal'),
  selectModel: document.getElementById('selectModel'),
  cmTP: document.getElementById('cmTP'),
  cmFP: document.getElementById('cmFP'),
  cmFN: document.getElementById('cmFN'),
  cmTN: document.getElementById('cmTN'),
  metricAcc: document.getElementById('metricAcc'),
  metricSens: document.getElementById('metricSens'),
  metricSpec: document.getElementById('metricSpec'),
  metricPrec: document.getElementById('metricPrec'),
  metricF1: document.getElementById('metricF1'),
  metricBalAcc: document.getElementById('metricBalAcc'),
  simCallout: document.getElementById('simCallout'),
  
  // Gallery
  galleryGrid: document.getElementById('galleryGrid'),
  galleryFilterPills: document.querySelectorAll('.gallery-pill')
};

// -------------------------------------------------------------
// 1. Math Rendering (KaTeX Integration)
// -------------------------------------------------------------
function triggerMathRender() {
  if (typeof renderMathInElement === 'function') {
    renderMathInElement(document.body, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$',  right: '$',  display: false }
      ],
      throwOnError: false
    });
  }
}

// -------------------------------------------------------------
// 2. Theme Management (Light / Dark Mode)
// -------------------------------------------------------------
function initTheme() {
  const savedTheme = localStorage.getItem('pcam-theme') || 'light';
  applyTheme(savedTheme);

  if (elements.btnTheme) {
    elements.btnTheme.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
      const next = current === 'dark' ? 'light' : 'dark';
      applyTheme(next);
      localStorage.setItem('pcam-theme', next);
    });
  }
}

function applyTheme(theme) {
  if (theme === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
    if (elements.btnTheme) {
      elements.btnTheme.innerHTML = '<span id="themeIcon">&#9788;</span> Light';
    }
  } else {
    document.documentElement.removeAttribute('data-theme');
    if (elements.btnTheme) {
      elements.btnTheme.innerHTML = '<span id="themeIcon">&#9790;</span> Dark';
    }
  }
}

// -------------------------------------------------------------
// 3. Initialization & Data Loading
// -------------------------------------------------------------
async function initApp() {
  initTheme();
  setupTabs();
  setupCanvasControls();
  setupSimulatorControls();
  
  try {
    const curRes = await fetch('data/curated_patches.json');
    state.curatedPatches = await curRes.json();
    
    const predRes = await fetch('data/test_predictions.json');
    state.testPredictions = await predRes.json();
    
    renderCurrentPatch();
    renderErrorGallery();
    recalculateSimulator();
    
    // Trigger KaTeX parsing on all math markup
    setTimeout(triggerMathRender, 200);
  } catch (err) {
    console.error('Failed to load showcase data:', err);
  }
}

// -------------------------------------------------------------
// 4. Tab Navigation
// -------------------------------------------------------------
function setupTabs() {
  elements.tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.tab;
      elements.tabs.forEach(t => t.classList.remove('active'));
      elements.panes.forEach(p => p.classList.remove('active'));
      
      tab.classList.add('active');
      const targetPane = document.getElementById(target);
      if (targetPane) targetPane.classList.add('active');
      
      // Re-trigger KaTeX in case newly visible pane has unrendered math
      setTimeout(triggerMathRender, 50);
    });
  });
}

// -------------------------------------------------------------
// 5. Diagnostic Studio Canvas & Patch Viewer
// -------------------------------------------------------------
function setupCanvasControls() {
  const roiBtn = document.getElementById('btnToggleROI');
  if (roiBtn) {
    roiBtn.addEventListener('click', () => {
      state.showROI = !state.showROI;
      roiBtn.classList.toggle('active', state.showROI);
      renderPatchCanvas();
    });
  }

  const filterBtns = document.querySelectorAll('.btn-filter');
  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.canvasFilter = btn.dataset.filter;
      renderPatchCanvas();
    });
  });

  document.getElementById('btnPrevPatch')?.addEventListener('click', () => navigatePatch(-1));
  document.getElementById('btnNextPatch')?.addEventListener('click', () => navigatePatch(1));
  document.getElementById('btnRandomPatch')?.addEventListener('click', selectRandomPatch);

  const inspectorPills = document.querySelectorAll('.inspector-pill');
  inspectorPills.forEach(pill => {
    pill.addEventListener('click', () => {
      inspectorPills.forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      state.activeFilter = pill.dataset.category;
      
      const idx = state.curatedPatches.findIndex(matchesCategory);
      if (idx !== -1) {
        state.currentPatchIdx = idx;
        renderCurrentPatch();
      }
    });
  });
}

function matchesCategory(patch) {
  if (state.activeFilter === 'all') return true;
  if (state.activeFilter === 'benign') return patch.label === 0 && patch.error_type === 'Correct';
  if (state.activeFilter === 'tumor') return patch.label === 1 && patch.error_type === 'Correct';
  if (state.activeFilter === 'fp') return patch.label === 0 && patch.error_type === 'False Positive';
  if (state.activeFilter === 'fn') return patch.label === 1 && patch.error_type === 'False Negative';
  if (state.activeFilter === 'disputed') return patch.consensus_status.includes('Disputed');
  return true;
}

function navigatePatch(dir) {
  const total = state.curatedPatches.length;
  if (!total) return;
  
  let newIdx = state.currentPatchIdx;
  let count = 0;
  
  do {
    newIdx = (newIdx + dir + total) % total;
    count++;
    if (matchesCategory(state.curatedPatches[newIdx])) {
      state.currentPatchIdx = newIdx;
      renderCurrentPatch();
      return;
    }
  } while (count < total);
}

function selectRandomPatch() {
  const matchingIndices = [];
  state.curatedPatches.forEach((p, idx) => {
    if (matchesCategory(p)) matchingIndices.push(idx);
  });
  
  if (matchingIndices.length > 0) {
    const r = Math.floor(Math.random() * matchingIndices.length);
    state.currentPatchIdx = matchingIndices[r];
    renderCurrentPatch();
  }
}

function renderCurrentPatch() {
  const patch = state.curatedPatches[state.currentPatchIdx];
  if (!patch) return;

  elements.patchIdTitle.textContent = `Patch ID: ${patch.filename}`;
  elements.currentPatchLabel.textContent = patch.ground_truth.toUpperCase();
  elements.currentPatchLabel.className = patch.label === 1 ? 'badge badge-tumor' : 'badge badge-normal';
  
  elements.patchCategoryBadge.textContent = patch.category;
  elements.consensusBadge.textContent = patch.consensus_status;
  elements.consensusBadge.className = patch.consensus_status.includes('Success') 
    ? 'badge badge-consensus' 
    : (patch.consensus_status.includes('Failure') ? 'badge badge-tumor' : 'badge badge-warning');

  elements.clinicalNoteText.textContent = patch.clinical_note;

  if (patch.failure_tier) {
    elements.riskAlertBox.className = 'alert-box warning';
    elements.riskAlertBox.style.display = 'block';
    elements.riskAlertBox.innerHTML = `<strong>Pathological Failure Alert:</strong> ${patch.failure_tier}. ${patch.clinical_note}`;
  } else {
    elements.riskAlertBox.className = 'alert-box success';
    elements.riskAlertBox.style.display = 'block';
    elements.riskAlertBox.innerHTML = `<strong>Diagnostic Status:</strong> Unanimous consensus agreement. Clear-cut morphology with high diagnostic confidence.`;
  }

  const p = patch.probabilities;
  updateMeter(elements.meterBlend, elements.valBlend, p.hybrid_blend);
  updateMeter(elements.meterHgbm, elements.valHgbm, p.histgbm);
  updateMeter(elements.meterRf, elements.valRf, p.random_forest);
  updateMeter(elements.meterRbf, elements.valRbf, p.rbf_svm);
  updateMeter(elements.meterLr, elements.valLr, p.logreg);

  renderFeatureBars(patch.key_features);
  renderPatchCanvas();
}

function updateMeter(barEl, valEl, prob) {
  const pct = Math.round(prob * 100);
  barEl.style.width = `${pct}%`;
  valEl.textContent = `${(prob).toFixed(3)} (${pct}%)`;
}

function renderFeatureBars(feats) {
  if (!elements.featureBarContainer) return;
  elements.featureBarContainer.innerHTML = '';
  
  for (const [name, zval] of Object.entries(feats)) {
    const row = document.createElement('div');
    row.className = 'feature-bar-row';
    
    const isHigh = zval > 0;
    const absZ = Math.min(Math.abs(zval) / 3.0 * 50, 50);
    
    row.innerHTML = `
      <div class="feature-bar-header">
        <span class="feature-name">${name}</span>
        <span class="feature-zscore" style="color: ${isHigh ? 'var(--accent)' : 'var(--red)'};">
          ${zval > 0 ? '+' : ''}${zval.toFixed(2)} &sigma;
        </span>
      </div>
      <div class="feature-bar-track">
        <div class="feature-bar-center"></div>
        <div class="feature-bar-fill ${isHigh ? 'positive' : 'negative'}"
             style="${isHigh ? 'left: 50%' : 'right: 50%'}; width: ${absZ}%;"></div>
      </div>
    `;
    elements.featureBarContainer.appendChild(row);
  }
}

function renderPatchCanvas() {
  const patch = state.curatedPatches[state.currentPatchIdx];
  if (!patch) return;

  const canvas = elements.patchCanvas;
  const ctx = canvas.getContext('2d');
  const img = new Image();
  img.src = patch.image_url;
  
  img.onload = () => {
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(img, 0, 0, 288, 288);
    
    if (state.canvasFilter !== 'none') {
      applyCanvasFilter(ctx, 288, 288, state.canvasFilter);
    }
    
    if (state.showROI) {
      ctx.strokeStyle = '#dc2626';
      ctx.lineWidth = 2.5;
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(96, 96, 96, 96);
      ctx.setLineDash([]);
      
      ctx.fillStyle = '#dc2626';
      ctx.font = 'bold 11px sans-serif';
      ctx.fillText('Central 32×32 ROI', 98, 90);
    }
  };
}

function applyCanvasFilter(ctx, w, h, filterType) {
  const imgData = ctx.getImageData(0, 0, w, h);
  const d = imgData.data;
  
  if (filterType === 'grayscale') {
    for (let i = 0; i < d.length; i += 4) {
      const avg = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
      d[i] = avg; d[i + 1] = avg; d[i + 2] = avg;
    }
  } else if (filterType === 'invert') {
    for (let i = 0; i < d.length; i += 4) {
      d[i] = 255 - d[i];
      d[i + 1] = 255 - d[i + 1];
      d[i + 2] = 255 - d[i + 2];
    }
  } else if (filterType === 'sobel') {
    const gray = new Float32Array(w * h);
    for (let i = 0, j = 0; i < d.length; i += 4, j++) {
      gray[j] = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    }
    
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const idx = y * w + x;
        const gx = -gray[idx - w - 1] + gray[idx - w + 1]
                   - 2 * gray[idx - 1] + 2 * gray[idx + 1]
                   - gray[idx + w - 1] + gray[idx + w + 1];
        const gy = -gray[idx - w - 1] - 2 * gray[idx - w] - gray[idx - w + 1]
                   + gray[idx + w - 1] + 2 * gray[idx + w] + gray[idx + w + 1];
        const mag = Math.min(255, Math.hypot(gx, gy));
        
        const pi = idx * 4;
        d[pi] = mag; d[pi + 1] = mag; d[pi + 2] = mag;
      }
    }
  }
  
  ctx.putImageData(imgData, 0, 0);
}

// -------------------------------------------------------------
// 6. Real-Time Decision Threshold Simulator (5,000 Patches)
// -------------------------------------------------------------
function setupSimulatorControls() {
  if (!elements.sliderThreshold) return;

  elements.sliderThreshold.addEventListener('input', (e) => {
    state.simThreshold = parseFloat(e.target.value);
    elements.thresholdVal.textContent = state.simThreshold.toFixed(2);
    recalculateSimulator();
  });

  elements.selectModel?.addEventListener('change', (e) => {
    state.simModel = e.target.value;
    recalculateSimulator();
  });

  document.querySelectorAll('.btn-preset').forEach(btn => {
    btn.addEventListener('click', () => {
      const val = parseFloat(btn.dataset.preset);
      state.simThreshold = val;
      elements.sliderThreshold.value = val;
      elements.thresholdVal.textContent = val.toFixed(2);
      recalculateSimulator();
    });
  });
}

function recalculateSimulator() {
  const preds = state.testPredictions;
  if (!preds || !preds.length) return;

  const tau = state.simThreshold;
  const modKey = state.simModel;

  let tp = 0, fp = 0, fn = 0, tn = 0;

  for (let i = 0; i < preds.length; i++) {
    const p = preds[i];
    const prob = p[modKey];
    const yTrue = p.y;
    const yPred = prob >= tau ? 1 : 0;

    if (yTrue === 1 && yPred === 1) tp++;
    else if (yTrue === 0 && yPred === 1) fp++;
    else if (yTrue === 1 && yPred === 0) fn++;
    else tn++;
  }

  const total = tp + fp + fn + tn;
  const sens = tp / (tp + fn) || 0;
  const spec = tn / (tn + fp) || 0;
  const prec = tp / (tp + fp) || 0;
  const f1 = (prec + sens > 0) ? (2 * prec * sens) / (prec + sens) : 0;
  const acc = (tp + tn) / total;
  const balAcc = (sens + spec) / 2;

  elements.cmTP.textContent = tp.toLocaleString();
  elements.cmFP.textContent = fp.toLocaleString();
  elements.cmFN.textContent = fn.toLocaleString();
  elements.cmTN.textContent = tn.toLocaleString();

  elements.metricAcc.textContent = `${(acc * 100).toFixed(2)}%`;
  elements.metricSens.textContent = `${(sens * 100).toFixed(2)}%`;
  elements.metricSpec.textContent = `${(spec * 100).toFixed(2)}%`;
  elements.metricPrec.textContent = `${(prec * 100).toFixed(2)}%`;
  elements.metricF1.textContent = f1.toFixed(4);
  elements.metricBalAcc.textContent = `${(balAcc * 100).toFixed(2)}%`;

  let advice = '';
  if (tau <= 0.35) {
    advice = `<strong>Clinical Screening Regime (&tau; = ${tau.toFixed(2)}):</strong> Maximizes sensitivity (${(sens*100).toFixed(1)}%). Catches ${tp} out of 2,500 tumours, missing only ${fn} malignant foci. Suitable for preliminary triage where false negatives must be minimized and pathologists review all flagged samples.`;
  } else if (tau >= 0.65) {
    advice = `<strong>High-Precision Confirmatory Regime (&tau; = ${tau.toFixed(2)}):</strong> Maximizes precision (${(prec*100).toFixed(1)}%). Yields only ${fp} false alarms across 5,000 patches. Suitable for automated secondary alert systems to prevent unnecessary surgical biopsies.`;
  } else {
    advice = `<strong>Balanced Operating Point (&tau; = ${tau.toFixed(2)}):</strong> Reaches peak overall accuracy (${(acc*100).toFixed(2)}%) and balanced F1-score (${f1.toFixed(3)}), trading off false alarms against missed micrometastases.`;
  }
  elements.simCallout.innerHTML = advice;
}

// -------------------------------------------------------------
// 7. Forensic Error Gallery
// -------------------------------------------------------------
function renderErrorGallery() {
  const grid = elements.galleryGrid;
  if (!grid) return;
  grid.innerHTML = '';

  const errorPatches = state.curatedPatches.filter(p => p.error_type !== 'Correct' || p.consensus_status.includes('Disputed'));

  errorPatches.forEach(patch => {
    const card = document.createElement('div');
    card.className = 'gallery-card';
    card.dataset.tier = patch.failure_tier || 'Disputed';

    const probBlend = (patch.probabilities.hybrid_blend * 100).toFixed(1);
    const badgeClass = patch.label === 1 ? 'badge-tumor' : 'badge-normal';

    card.innerHTML = `
      <div class="gallery-img-wrapper">
        <img src="${patch.image_url}" alt="${patch.filename}">
        <span class="badge ${badgeClass}" style="position:absolute; top:8px; left:8px; z-index:2;">
          ${patch.ground_truth}
        </span>
        <span class="badge badge-warning" style="position:absolute; top:8px; right:8px; z-index:2;">
          P(Tumor): ${probBlend}%
        </span>
      </div>
      <div class="gallery-body">
        <div>
          <div class="gallery-title">${patch.filename}</div>
          <div class="gallery-tier">${patch.failure_tier || 'Ensemble Disagreement'}</div>
          <div class="gallery-desc">${patch.clinical_note}</div>
        </div>
        <div class="gallery-inspect-btn">Inspect in Studio &rarr;</div>
      </div>
    `;

    card.addEventListener('click', () => {
      const idx = state.curatedPatches.findIndex(p => p.id === patch.id);
      if (idx !== -1) {
        state.currentPatchIdx = idx;
        renderCurrentPatch();
        document.querySelector('.tab-btn[data-tab="tab-inspector"]')?.click();
      }
    });

    grid.appendChild(card);
  });

  elements.galleryFilterPills.forEach(pill => {
    pill.addEventListener('click', () => {
      elements.galleryFilterPills.forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      const tierFilter = pill.dataset.tier;
      
      const cards = grid.querySelectorAll('.gallery-card');
      cards.forEach(c => {
        if (tierFilter === 'all' || c.dataset.tier.toLowerCase().includes(tierFilter.toLowerCase())) {
          c.style.display = 'flex';
        } else {
          c.style.display = 'none';
        }
      });
    });
  });
}

// -------------------------------------------------------------
// Bootstrapping
// -------------------------------------------------------------
document.addEventListener('DOMContentLoaded', initApp);
