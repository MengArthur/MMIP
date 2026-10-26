// RouteVision front end: talks to the Flask API and draws the maps on canvas.
// The analysis view is a four-step story; every step shows three maps side by side:
//   1 熱點預測  eGR | chosen AI method | real hotspots
//   2 使用率圖  eGR map | U-Net map | real utilisation map
//   3 可信度    Ensemble prediction | ensemble disagreement | real hotspots
//   4 判斷依據  input (RUDY) | Grad-CAM | real hotspots
const $ = (id) => document.getElementById(id);
const GRID = 8;
const CELL = 512 / GRID;                       // canvases are 512x512 internally
const EGR = 'eGR (EDA estimate)', RUDY = 'RUDY (EDA estimate)', ENS = 'Ensemble';
const NAME = { [RUDY]: 'RUDY', [EGR]: 'eGR', 'OpenCV + RandomForest': 'Random Forest', 'U-Net (multimodal)': 'U-Net 多模態' };
const STEPS = ['hot', 'map', 'trust', 'why'];
const DESIGN_COLORS = ['#2a78d6', '#eb6834', '#1baf7a'];

const S = {
  info: null, demo: null, result: null, step: 'hot', method: null, camMethod: null, cams: {},
  hover: null, file: null, metric: 'ROC-AUC', gray: [], rudyImg: null, mapImg: {}, busyT: null, busyTick: null,
};
const disp = (m) => NAME[m] || m;
const f3 = (v) => (v == null ? '–' : v.toFixed(3));
const pct = (v, d = 0) => (v == null ? '–' : `${(v * 100).toFixed(d)}%`);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function toast(msg) {
  const el = $('toast');
  el.textContent = msg;
  el.classList.remove('hidden');
  clearTimeout(toast.t);
  toast.t = setTimeout(() => el.classList.add('hidden'), 6000);
}

async function api(path, opts = {}) {
  let res, data;
  try {
    res = await fetch(path, opts);
    data = await res.json();
  } catch (e) {
    throw new Error(res ? `回應讀取失敗（HTTP ${res.status}），請再試一次` : '無法連線到後端，請確認 server.py 正在執行');
  }
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

function loadImage(src) {
  return new Promise((ok, err) => { const im = new Image(); im.onload = () => ok(im); im.onerror = err; im.src = src; });
}

// ================================================================== views
function showView(v) {
  if (!['analyze', 'results', 'method'].includes(v)) v = 'analyze';
  document.querySelectorAll('.view').forEach((el) => el.classList.toggle('hidden', el.id !== `view-${v}`));
  document.querySelectorAll('.view-tab').forEach((b) => {
    b.classList.toggle('active', b.dataset.view === v);
    b.setAttribute('aria-selected', b.dataset.view === v);
  });
  if (location.hash.slice(1) !== v) history.replaceState(null, '', v === 'analyze' ? location.pathname : `#${v}`);
  hideTip();
}
document.querySelectorAll('.view-tab').forEach((b) => (b.onclick = () => showView(b.dataset.view)));
document.addEventListener('click', (e) => {
  const a = e.target.closest('[data-goto]');
  if (a) { e.preventDefault(); showView(a.dataset.goto); window.scrollTo(0, 0); }
});
window.addEventListener('hashchange', () => showView(location.hash.slice(1)));

// ================================================================== setup
async function init() {
  try {
    S.info = await api('/api/info');
  } catch (e) {
    toast('無法連線到後端：' + e.message);
    return;
  }
  S.method = bestLearned() || 'Multimodal';
  // explicit deep link only (e.g. ?demo=0&step=map, for bookmarks and screenshots); a plain visit opens nothing.
  // Read before showView(), which rewrites the address.
  const q = new URLSearchParams(location.search);
  buildGallery();
  buildResults();
  buildMethodPage();
  showView(location.hash.slice(1));
  pollStatus();
  if (q.has('demo')) {
    if (STEPS.includes(q.get('step'))) S.step = q.get('step');
    selectDemo(Number(q.get('demo')));
  }
}

function methodMeta(name) { return S.info.methods.find((m) => m.name === name); }
function learnedMethods() { return S.info.methods.filter((m) => m.family !== 'EDA 粗估').map((m) => m.name); }

// learned method with the highest mean cross-design ROC-AUC (taken from the results, not hard-coded)
function bestLearned() {
  const avail = new Set(learnedMethods());
  const rows = lodoBy('ROC-AUC').filter((r) => avail.has(r.method) && r.vals.every((v) => v != null));
  return rows.length ? rows.reduce((a, r) => (r.mean > a.mean ? r : a)).method : null;
}

// ================================================================== gallery / upload
function groupStyle(g) {
  if (g.includes('表現好')) return { short: '沒看過的晶片 · AI 較準', tip: 'Vortex-small 完全沒參與訓練；這幾張是 AI 明顯贏過 EDA 粗估的例子', cls: 'good', color: '#17663a' };
  if (g.includes('eGR')) return { short: '沒看過的晶片 · eGR 較準', tip: '也呈現 EDA 粗估比較準的版圖，避免只挑好看的例子', cls: '', color: '#c27803' };
  if (g.includes('驗證')) return { short: '訓練設計的驗證版圖', tip: '同設計但這張版圖沒參與訓練；比沒看過的晶片容易，只供參考', cls: 'val', color: '#0f766e' };
  return { short: g, tip: g, cls: 'up', color: '#7a7e85' };
}

function buildGallery() {
  const groups = [];
  S.info.demos.forEach((d, k) => {
    let g = groups.find((x) => x.name === d.group);
    if (!g) groups.push((g = { name: d.group, items: [] }));
    g.items.push(k);
  });
  $('gallery').innerHTML = groups.map((g) => {
    const st = groupStyle(g.name);
    return `<div class="g-group"><h4 title="${esc(st.tip)}"><i style="background:${st.color}"></i>${esc(st.short)}</h4>
      <div class="g-list">${g.items.map((k) => {
        const d = S.info.demos[k];
        return `<button class="g-item" data-k="${k}" title="${esc(d.design + '_' + d.sample)}">
          <img src="${d.thumb}" alt="">
          <span><span class="g-name">${esc(d.design)}<i class="g-ready ${d.ready ? '' : 'pending'}"></i></span>
          <span class="g-meta">${d.params.freq} MHz · ${d.params.fpu}% · ${d.n_hot} 熱點</span></span></button>`;
      }).join('')}</div></div>`;
  }).join('');
  $('gallery').querySelectorAll('.g-item').forEach((b) => (b.onclick = () => selectDemo(Number(b.dataset.k))));
}

async function pollStatus() {
  try {
    const st = await api('/api/status');
    const ready = new Set(st.ready);
    document.querySelectorAll('.g-item').forEach((b) => {
      const k = Number(b.dataset.k), dot = b.querySelector('.g-ready');
      S.info.demos[k].ready = ready.has(k);
      dot.classList.toggle('pending', !ready.has(k));
      dot.title = ready.has(k) ? '已預先算好，點了立即顯示' : '背景預先計算中';
    });
    if (st.ready.length < st.total) setTimeout(pollStatus, 2500);
  } catch (e) { setTimeout(pollStatus, 5000); }
}

$('uploadOpen').onclick = () => $('uploadDlg').showModal();
$('uploadCancel').onclick = () => $('uploadDlg').close();
function setFile(f) {
  if (!f) return;
  if (!f.name.toLowerCase().endsWith('.npz')) { toast('請選擇 .npz 檔'); return; }
  S.file = f;
  $('fileName').textContent = f.name;
  $('uploadBtn').disabled = false;
}
$('fileInput').onchange = (e) => setFile(e.target.files[0]);
const drop = $('drop');
['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', (e) => setFile(e.dataTransfer.files[0]));
$('uploadBtn').onclick = () => {
  if (!S.file) return;
  $('uploadDlg').close();
  const fd = new FormData();
  fd.append('file', S.file);
  S.demo = null;
  document.querySelectorAll('.g-item').forEach((b) => b.classList.remove('active'));
  analyse({ method: 'POST', body: fd }, '上傳的版圖推論中，所有方法一次算完（CPU 約 20 秒）');
};

function selectDemo(k) {
  if (!S.info.demos[k]) return;
  S.demo = k;
  document.querySelectorAll('.g-item').forEach((b) => b.classList.toggle('active', Number(b.dataset.k) === k));
  analyse({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ demo: k }) },
          S.info.demos[k].ready ? '載入中…' : '這張版圖還沒預先算好，所有方法推論中（CPU 約 20 秒）');
}

let analyseSeq = 0;
async function analyse(opts, busyMsg) {
  const seq = ++analyseSeq;
  setBusy(busyMsg);
  try {
    const r = await api('/api/predict', opts);
    if (seq !== analyseSeq) return;
    const mapKeys = Object.keys(r.maps || {});
    const [gray, rudy, maps] = await Promise.all([
      Promise.all(r.channels.map((c) => loadImage(c.gray))),
      loadImage(r.channels[1].src),
      Promise.all(mapKeys.map((k) => loadImage(r.maps[k]))),
    ]);
    if (seq !== analyseSeq) return;
    S.gray = gray; S.rudyImg = rudy;
    S.mapImg = Object.fromEntries(mapKeys.map((k, i) => [k, maps[i]]));
    S.result = r;
    S.hover = null;
    if (!r.scores[S.method]) S.method = bestLearned() || learnedMethods().find((m) => r.scores[m]);
    if (!stepAvailable(S.step)) S.step = 'hot';
    renderHeader();
    renderStep();
  } catch (e) {
    if (seq === analyseSeq) toast(e.message);
  } finally {
    if (seq === analyseSeq) setBusy(null);
  }
}

function setBusy(msg) {
  clearTimeout(S.busyT); clearInterval(S.busyTick);
  if (!msg) { $('busy').classList.add('hidden'); return; }
  S.busyT = setTimeout(() => {                                // only if it takes a moment (cached: ~50 ms)
    $('busyText').textContent = msg;
    $('busy').classList.remove('hidden');
    const t0 = Date.now();
    $('busyTimer').textContent = '';
    S.busyTick = setInterval(() => ($('busyTimer').textContent = `已經過 ${Math.round((Date.now() - t0) / 1000)} 秒`), 1000);
  }, 250);
}

// ================================================================== analysis: header + steps
function renderHeader() {
  const r = S.result;
  $('welcome').classList.add('hidden');
  $('analysis').classList.remove('hidden');
  const st = groupStyle(r.group);
  $('layoutGroup').textContent = st.short;
  $('layoutGroup').title = st.tip;
  $('layoutGroup').className = 'tag ' + st.cls;
  const [design, sample] = r.name.includes(' | ') ? r.name.split(' | ') : [r.name, ''];
  const type = (S.info.lodo?.designs || []).find((d) => d.name === design)?.type;
  $('layoutTitle').textContent = design + (type ? `（${type}）` : '');
  const P = r.params;
  $('layoutParams').textContent = `${P.freq} MHz · 使用率 ${P.fpu}%`;
  $('layoutParams').title = `設計參數（CircuitNet）\n時脈 freq：${P.freq} MHz\n使用率 fpu：${P.fpu}%\nmacro 擺法 mp：${P.mp}\n電源網路 p：${P.p}\n`
    + `filler fi：${P.fi === 'ar' ? '繞線後' : '繞線前'}\n長寬比 fpa：${P.fpa}${sample ? `\n\n${sample}` : ''}`;
}

function unetName() { return S.result?.maps ? learnedMethods().find((m) => m.startsWith('U-Net') && S.result.maps[m]) : null; }
function camMethods() { return S.info.methods.filter((m) => m.gradcam).map((m) => m.name); }
function uncOK() { return !!(S.result?.uncertainty && S.info.uncertainty && S.result.scores[ENS]); }
function baseline() { return S.result.scores[EGR] ? EGR : RUDY; }

function stepAvailable(step) {
  if (!S.result) return step === 'hot';
  return { hot: true, map: !!unetName(), trust: uncOK(), why: camMethods().length > 0 }[step];
}

document.querySelectorAll('.step').forEach((b) => (b.onclick = () => goStep(b.dataset.step)));
function goStep(step) {
  if (!stepAvailable(step)) return;
  S.step = step;
  renderStep();
}

function layoutStats(m) {
  const r = S.result, sc = r.scores[m];
  if (!sc) return null;
  const thr = r.thresholds[m], truth = r.truth;
  let tp = 0, fp = 0, fn = 0, np = 0, nt = 0;
  for (let i = 0; i < GRID; i++) for (let j = 0; j < GRID; j++) {
    const p = sc[i][j] >= thr;
    np += p;
    if (truth) { const t = truth[i][j] === 1; nt += t; tp += p && t; fp += p && !t; fn += !p && t; }
  }
  return { tp, fp, fn, np, nt, thr, f1: truth ? 2 * tp / Math.max(2 * tp + fp + fn, 1) : null };
}

const STEP_TEXT = {
  hot: 'EDA 工具的粗估、AI 的預測、routing 後的真實熱點並排。',
  map: 'U-Net 直接預測整張 routing 後的繞線使用率（逐像素），和 eGR 的使用率圖比較。越亮越擠。',
  trust: 'Ensemble 的三個成員（RF、Multimodal、ViT）意見分歧的地方，就是模型沒把握、建議人工確認的區塊。',
  why: 'Grad-CAM：模型判斷「這裡是熱點」時依賴哪些區域。顏色越暖越重要。',
};

function renderStep() {
  const r = S.result;
  if (!r) return;
  document.querySelectorAll('.step').forEach((b) => {
    b.classList.toggle('active', b.dataset.step === S.step);
    b.setAttribute('aria-selected', b.dataset.step === S.step);
    b.disabled = !stepAvailable(b.dataset.step);
  });
  $('stepDesc').textContent = STEP_TEXT[S.step];
  renderCtrl();
  ({ hot: stepHot, map: stepMap, trust: stepTrust, why: stepWhy })[S.step]();
  renderBars();
}

function renderCtrl() {
  const c = $('stepCtrl');
  if (S.step === 'hot') {
    const f1 = Object.fromEntries((S.result.comparison || []).map((x) => [x.method, x.f1]));
    const opts = learnedMethods().filter((m) => S.result.scores[m]);
    c.innerHTML = `<span class="ctrl-label">AI 方法</span><select class="select" id="methodSel">${opts.map((m) =>
      `<option value="${esc(m)}" ${m === S.method ? 'selected' : ''}>${esc(disp(m))}${f1[m] != null ? `　F1 ${f1[m].toFixed(3)}` : ''}</option>`).join('')}</select>`;
    $('methodSel').onchange = (e) => { S.method = e.target.value; renderStep(); };
  } else if (S.step === 'why') {
    const best = bestLearned();
    const ms = camMethods().sort((a, b) => (b === best) - (a === best));   // the best method first
    if (!ms.includes(S.camMethod)) S.camMethod = ms[0];
    c.innerHTML = `<div class="seg">${ms.map((m) => `<button class="seg-btn ${m === S.camMethod ? 'active' : ''}" data-cam="${esc(m)}">${esc(disp(m))}</button>`).join('')}</div>`;
    c.querySelectorAll('[data-cam]').forEach((b) => (b.onclick = () => { S.camMethod = b.dataset.cam; renderStep(); }));
  } else {
    c.innerHTML = '';
  }
}

function setPanels(titles, nums) {
  titles.forEach((t, k) => { $(`pt${k}`).textContent = t; });
  nums.forEach((n, k) => { $(`pn${k}`).innerHTML = n || ''; });
}
const num = (label, v) => (v == null ? '' : `<small>${label}</small>${typeof v === 'number' ? v.toFixed(3) : v}`);
function deltaChip(d, label) {
  if (d == null || Math.abs(d) < 0.0005) return '';
  return `<span class="chip ${d > 0 ? 'up' : 'down'}">${d > 0 ? '▲' : '▼'} ${label} ${d > 0 ? '+' : ''}${d.toFixed(3)}</span>`;
}

// ------------------------------------------------------------------ step 1: hotspots
function stepHot() {
  const r = S.result, b = baseline(), m = S.method;
  const bs = layoutStats(b), as = layoutStats(m);
  setPanels([`EDA 粗估 · ${disp(b)}`, `AI · ${disp(m)}`, '真實熱點（routing 後）'],
            [num('F1', bs.f1), num('F1', as.f1), r.truth ? `${as.nt} <small>/ 64 格</small>` : '']);
  $('verdict').innerHTML = r.truth
    ? `<b class="ai">${esc(disp(m))}</b> 抓到 <b>${as.tp}/${as.nt}</b> 個熱點、誤報 <b>${as.fp}</b> 格；${esc(disp(b))} 抓到 <b>${bs.tp}/${bs.nt}</b>、誤報 <b>${bs.fp}</b> 格`
      + deltaChip(as.f1 - bs.f1, 'F1')
    : `<b class="ai">${esc(disp(m))}</b> 預測 <b>${as.np}</b> 個熱點區塊（上傳的檔案沒有真實結果可比對）`;
  $('legend').innerHTML = r.truth
    ? '<span><i class="sw hit"></i>抓到</span><span><i class="sw fa"></i>誤報</span><span><i class="sw miss"></i>漏掉</span><span><i class="sw truth"></i>真實熱點</span>'
    : '<span><i class="sw hit"></i>預測為熱點</span>';
  $('pl1').classList.add('hidden');
  drawAll();
}

// ------------------------------------------------------------------ step 2: utilisation maps
function stepMap() {
  const r = S.result, u = unetName(), b = baseline(), M = r.map_metrics || {};
  const pr = (k) => (M[k] ? M[k].Pearson : null);
  setPanels([`EDA 粗估 · ${disp(b)} 使用率`, `AI · ${disp(u)} 預測`, '真實使用率（routing 後）'],
            [num('Pearson', pr(b)), num('Pearson', pr(u)), '']);
  $('verdict').innerHTML = r.truth && pr(u) != null
    ? `<b class="ai">${esc(disp(u))}</b> 的預測圖和真實結果的相關係數 <b>${f3(pr(u))}</b>，${esc(disp(b))} 是 <b>${f3(pr(b))}</b>`
      + deltaChip(pr(b) != null ? pr(u) - pr(b) : null, 'Pearson')
    : `<b class="ai">${esc(disp(u))}</b> 預測的整張使用率圖`;
  $('legend').innerHTML = '<span>使用率 0<i class="sw ramp magma"></i>1</span>'
    + (r.truth ? '<span><i class="sw dash"></i>真實熱點區塊</span>' : '')
    + (b === RUDY ? '<span class="muted">RUDY 為相對值</span>' : '');
  $('pl1').classList.add('hidden');
  drawAll();
}

// ------------------------------------------------------------------ step 3: trust (ensemble disagreement)
function isUnsure(i, j) { return S.result.uncertainty[i][j] >= S.info.uncertainty.cutoff; }
function uncStats() {
  const r = S.result, sc = r.scores[ENS], thr = r.thresholds[ENS];
  let n = 0, wrong = 0, allWrong = 0;
  for (let i = 0; i < GRID; i++) for (let j = 0; j < GRID; j++) {
    const u = isUnsure(i, j);
    n += u;
    if (r.truth) { const bad = (sc[i][j] >= thr) !== (r.truth[i][j] === 1); allWrong += bad; wrong += bad && u; }
  }
  return { n, wrong, allWrong };
}

function stepTrust() {
  const r = S.result, es = layoutStats(ENS), u = uncStats();
  setPanels(['AI · Ensemble 預測', '沒把握的區塊（三個模型意見分歧）', '真實熱點（routing 後）'],
            [num('F1', es.f1), `${u.n} <small>格</small>`, r.truth ? `${es.nt} <small>/ 64 格</small>` : '']);
  $('verdict').innerHTML = r.truth
    ? `Ensemble 判錯 <b>${u.allWrong}</b> 格，其中 <b class="ai">${u.wrong}</b> 格落在它標出的 <b>${u.n}</b> 個「沒把握」區塊裡`
    : `Ensemble 標出 <b>${u.n}</b> 個沒把握的區塊，建議人工確認`;
  $('legend').innerHTML = (r.truth ? '<span><i class="sw hit"></i>抓到</span><span><i class="sw fa"></i>誤報</span><span><i class="sw miss"></i>漏掉</span>' : '')
    + '<span>分歧 低<i class="sw ramp violet"></i>高</span><span><i class="sw unc"></i>沒把握（前 20%）</span>';
  $('pl1').classList.add('hidden');
  drawAll();
}

// ------------------------------------------------------------------ step 4: why (Grad-CAM)
function camKey() { return `${S.result.layout_id}|${S.camMethod}`; }
function stepWhy() {
  const r = S.result;
  setPanels(['輸入 · RUDY（繞線需求估計）', `Grad-CAM · ${disp(S.camMethod)}`, '真實熱點（routing 後）'],
            ['', '', r.truth ? `${layoutStats(S.camMethod).nt} <small>/ 64 格</small>` : '']);
  $('verdict').innerHTML = `<b class="ai">${esc(disp(S.camMethod))}</b> 判斷熱點時最依賴暖色區域；對照右邊的真實熱點，看看關注的位置是否合理`;
  $('legend').innerHTML = '<span>關注 低<i class="sw ramp jet"></i>高</span>' + (r.truth ? '<span><i class="sw dash"></i>真實熱點區塊</span>' : '');
  drawAll();
  if (!S.cams[camKey()]) loadCam();
}

async function loadCam() {
  const r = S.result, key = camKey();
  $('pl1').classList.remove('hidden');
  try {
    const d = await api('/api/gradcam', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                          body: JSON.stringify({ layout_id: r.layout_id, method: S.camMethod }) });
    S.cams[key] = await loadImage(d.heat);
  } catch (e) {
    toast(e.message);
  }
  if (S.result === r && camKey() === key) { $('pl1').classList.add('hidden'); if (S.step === 'why') drawAll(); }
}

// ================================================================== drawing
function drawGrid(ctx) {
  ctx.strokeStyle = 'rgba(255,255,255,.08)';
  ctx.lineWidth = 1;
  for (let k = 1; k < GRID; k++) {
    ctx.beginPath(); ctx.moveTo(k * CELL + .5, 0); ctx.lineTo(k * CELL + .5, 512); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, k * CELL + .5); ctx.lineTo(512, k * CELL + .5); ctx.stroke();
  }
}

function drawImage(ctx, im, dim = 0) {
  ctx.clearRect(0, 0, 512, 512);
  if (im) { ctx.imageSmoothingEnabled = true; ctx.drawImage(im, 0, 0, 512, 512); }
  if (dim) { ctx.fillStyle = `rgba(14,16,19,${dim})`; ctx.fillRect(0, 0, 512, 512); }
}

function strokeCell(ctx, i, j, color, w, inset = 0, dash = null) {
  ctx.save();
  ctx.strokeStyle = color; ctx.lineWidth = w;
  if (dash) ctx.setLineDash(dash);
  const o = inset + w / 2;
  ctx.strokeRect(j * CELL + o, i * CELL + o, CELL - 2 * o, CELL - 2 * o);
  ctx.restore();
}

function hatchCell(ctx, i, j) {
  const x = j * CELL, y = i * CELL;
  ctx.save();
  ctx.fillStyle = 'rgba(240,160,12,.22)';
  ctx.fillRect(x, y, CELL, CELL);
  ctx.beginPath(); ctx.rect(x, y, CELL, CELL); ctx.clip();
  ctx.strokeStyle = 'rgba(240,160,12,.95)'; ctx.lineWidth = 3;
  for (let d = -CELL; d < CELL; d += 12) { ctx.beginPath(); ctx.moveTo(x + d, y + CELL); ctx.lineTo(x + d + CELL, y); ctx.stroke(); }
  ctx.restore();
  strokeCell(ctx, i, j, '#f0a00c', 2.5, 1);
}

function drawMessage(ctx, msg) {
  ctx.fillStyle = 'rgba(14,16,19,.6)'; ctx.fillRect(0, 0, 512, 512);
  ctx.fillStyle = '#e5e7eb'; ctx.font = '600 26px "Segoe UI", "Microsoft JhengHei", sans-serif'; ctx.textAlign = 'center';
  ctx.fillText(msg, 256, 262);
}

function drawOutcome(ctx, m) {
  const r = S.result, sc = r.scores[m];
  drawImage(ctx, S.gray[1], 0.4);
  drawGrid(ctx);
  if (!sc) { drawMessage(ctx, '沒有資料'); return; }
  const thr = r.thresholds[m], truth = r.truth;
  for (let i = 0; i < GRID; i++) for (let j = 0; j < GRID; j++) {
    const p = sc[i][j] >= thr, t = truth ? truth[i][j] === 1 : null;
    if (p && t !== false) { ctx.fillStyle = 'rgba(229,72,77,.62)'; ctx.fillRect(j * CELL, i * CELL, CELL, CELL); strokeCell(ctx, i, j, '#ff6b70', 2.5, 1); }
    else if (p) hatchCell(ctx, i, j);
    else if (t) strokeCell(ctx, i, j, '#22c3e6', 5, 4);
  }
}

function drawTruthHot(ctx) {
  const r = S.result;
  drawImage(ctx, S.gray[1], 0.4);
  drawGrid(ctx);
  if (!r.truth) { drawMessage(ctx, '沒有真實結果'); return; }
  for (let i = 0; i < GRID; i++) for (let j = 0; j < GRID; j++) {
    if (r.truth[i][j]) { ctx.fillStyle = 'rgba(180,35,42,.72)'; ctx.fillRect(j * CELL, i * CELL, CELL, CELL); strokeCell(ctx, i, j, '#ff6b70', 2.5, 1); }
  }
}

function truthOutlines(ctx) {
  const t = S.result.truth;
  if (t) for (let i = 0; i < GRID; i++) for (let j = 0; j < GRID; j++) if (t[i][j]) strokeCell(ctx, i, j, 'rgba(255,255,255,.85)', 2, 2, [8, 6]);
}

function drawMapPanel(ctx, key) {
  const im = S.mapImg[key];
  drawImage(ctx, im);
  if (!im) { drawMessage(ctx, '沒有這張圖'); return; }
  truthOutlines(ctx);
}

function drawUncertainty(ctx) {
  const r = S.result, U = r.uncertainty;
  drawImage(ctx, S.gray[1], 0.55);
  drawGrid(ctx);
  const top = Math.max(S.info.uncertainty.cutoff * 1.6, ...U.flat());
  for (let i = 0; i < GRID; i++) for (let j = 0; j < GRID; j++) {
    const t = Math.min(U[i][j] / top, 1);
    ctx.fillStyle = `rgba(139,92,246,${0.04 + 0.8 * Math.pow(t, 1.3)})`;
    ctx.fillRect(j * CELL, i * CELL, CELL, CELL);
    if (isUnsure(i, j)) {
      strokeCell(ctx, i, j, '#c4b5fd', 3, 3, [7, 5]);
      ctx.fillStyle = '#f5f3ff';
      ctx.font = '700 22px "Segoe UI", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('?', j * CELL + CELL / 2, i * CELL + CELL / 2 + 8);
    }
  }
}

function drawCamPanel(ctx) {
  drawImage(ctx, S.gray[0], 0.45);                         // cell density as the base
  const heat = S.cams[camKey()];
  if (heat) ctx.drawImage(heat, 0, 0, 512, 512);
  truthOutlines(ctx);
}

function drawHover(ctx) {
  if (!S.hover) return;
  const [i, j] = S.hover;
  strokeCell(ctx, i, j, 'rgba(0,0,0,.6)', 6, 0);
  strokeCell(ctx, i, j, '#ffffff', 3, 1.5);
}

function drawAll() {
  const r = S.result;
  if (!r) return;
  const c = [0, 1, 2].map((k) => $(`cv${k}`).getContext('2d'));
  const step = S.step;
  if (step === 'hot') { drawOutcome(c[0], baseline()); drawOutcome(c[1], S.method); drawTruthHot(c[2]); }
  else if (step === 'map') { drawMapPanel(c[0], baseline()); drawMapPanel(c[1], unetName()); drawMapPanel(c[2], 'truth'); }
  else if (step === 'trust') { drawOutcome(c[0], ENS); drawUncertainty(c[1]); drawTruthHot(c[2]); }
  else { drawImage(c[0], S.rudyImg); drawGrid(c[0]); drawCamPanel(c[1]); drawTruthHot(c[2]); }
  c.forEach(drawHover);
}

// ------------------------------------------------------------------ hover (synchronised across the three maps)
function cellAt(e) {
  const box = e.currentTarget.getBoundingClientRect();
  const j = Math.min(GRID - 1, Math.max(0, Math.floor((e.clientX - box.left) / box.width * GRID)));
  const i = Math.min(GRID - 1, Math.max(0, Math.floor((e.clientY - box.top) / box.height * GRID)));
  return [i, j];
}

function scoreLine(m, i, j, label) {
  const r = S.result, sc = r.scores[m];
  if (!sc) return '';
  const p = sc[i][j] >= r.thresholds[m];
  return `<div class="t-row"><span>${esc(label || disp(m))}　${sc[i][j].toFixed(3)}</span><b class="${p ? 't-hot' : ''}">${p ? '預測熱點' : '非熱點'}</b></div>`;
}

function tipHtml(i, j) {
  const r = S.result;
  let h = `<div class="t-head">區塊（第 ${i + 1} 列，第 ${j + 1} 行）</div>`;
  if (r.util) h += `<div class="t-row"><span>真實使用率　${r.util[i][j].toFixed(2)}</span><b class="${r.truth[i][j] ? 't-hot' : ''}">${r.truth[i][j] ? '熱點' : '非熱點'}</b></div>`;
  if (S.step === 'hot') h += scoreLine(baseline(), i, j) + scoreLine(S.method, i, j);
  else if (S.step === 'map') h += scoreLine(baseline(), i, j, `${disp(baseline())} 區塊平均`) + scoreLine(unetName(), i, j, `${disp(unetName())} 區塊平均`);
  else if (S.step === 'trust') {
    h += scoreLine(ENS, i, j);
    h += `<div class="t-row"><span>分歧　${r.uncertainty[i][j].toFixed(3)}</span><b style="color:#c4b5fd">${isUnsure(i, j) ? '沒把握' : ''}</b></div>`
      + `<div class="t-head">${S.info.uncertainty.members.map((m) => `${disp(m)} ${r.scores[m][i][j].toFixed(2)}`).join(' · ')}</div>`;
  } else h += scoreLine(S.camMethod, i, j);
  return h;
}

function showTip(e, html) {
  const tip = $('tooltip');
  tip.innerHTML = html;
  tip.classList.remove('hidden');
  const w = tip.offsetWidth, h = tip.offsetHeight;
  let x = e.clientX + 16, y = e.clientY + 16;
  if (x + w > innerWidth - 8) x = e.clientX - w - 16;
  if (y + h > innerHeight - 8) y = e.clientY - h - 16;
  tip.style.left = `${x}px`; tip.style.top = `${y}px`;
}
function hideTip() { $('tooltip').classList.add('hidden'); }

[0, 1, 2].forEach((k) => {
  const cv = $(`cv${k}`);
  cv.addEventListener('mousemove', (e) => {
    if (!S.result) return;
    const [i, j] = cellAt(e);
    if (!S.hover || S.hover[0] !== i || S.hover[1] !== j) { S.hover = [i, j]; drawAll(); }
    showTip(e, tipHtml(i, j));
  });
  cv.addEventListener('mouseleave', () => { S.hover = null; hideTip(); drawAll(); });
});

// ------------------------------------------------------------------ per-layout F1 of every method (collapsed)
function renderBars() {
  const rows = S.result.comparison;
  if (!rows) { $('f1Bars').innerHTML = '<p class="note">上傳的檔案沒有真實結果，無法計算 F1。</p>'; return; }
  const sorted = [...rows].sort((a, b) => b.f1 - a.f1);
  $('f1Bars').innerHTML = sorted.map((x) => `<button class="bar-row ${S.step === 'hot' && x.method === S.method ? 'active' : ''} ${methodMeta(x.method)?.family === 'EDA 粗估' ? 'eda' : ''}"
      data-m="${esc(x.method)}" title="抓到 ${x.tp}/${x.n_true}、誤報 ${x.fp}、漏掉 ${x.fn}">
      <span class="b-name">${esc(disp(x.method))}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${(x.f1 * 100).toFixed(1)}%"></span></span>
      <span class="b-val">${x.f1.toFixed(3)}</span></button>`).join('');
  $('f1Bars').querySelectorAll('[data-m]').forEach((b) => (b.onclick = () => {
    if (methodMeta(b.dataset.m)?.family === 'EDA 粗估') return;
    S.method = b.dataset.m; S.step = 'hot'; renderStep();
  }));
}

// ------------------------------------------------------------------ keyboard
document.addEventListener('keydown', (e) => {
  if ($('view-analyze').classList.contains('hidden') || e.altKey || e.ctrlKey || e.metaKey || $('uploadDlg').open) return;
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return;
  if ((e.key === 'ArrowLeft' || e.key === 'ArrowRight') && S.result) {
    const avail = STEPS.filter(stepAvailable);
    const k = avail.indexOf(S.step) + (e.key === 'ArrowRight' ? 1 : -1);
    if (k >= 0 && k < avail.length) goStep(avail[k]);
    e.preventDefault();
  } else if ((e.key === 'ArrowUp' || e.key === 'ArrowDown') && S.info.demos.length) {
    const n = S.info.demos.length;
    const k = S.demo == null ? 0 : (S.demo + (e.key === 'ArrowDown' ? 1 : -1) + n) % n;
    selectDemo(k);
    document.querySelector(`.g-item[data-k="${k}"]`)?.scrollIntoView({ block: 'nearest' });
    e.preventDefault();
  }
});

// ================================================================== results page
function meanStd(xs) {
  const m = xs.reduce((a, b) => a + b, 0) / xs.length;
  const sd = xs.length > 1 ? Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1)) : 0;
  return [m, sd];
}

function lodoBy(metric) {
  const L = S.info.lodo;
  if (!L) return [];
  const methods = [...new Set(L.rows.map((r) => r.method))];
  return methods.map((m) => {
    const vals = L.designs.map((d) => L.rows.find((r) => r.method === m && r.design === d.name)?.[metric]);
    const [mean, sd] = meanStd(vals.filter((v) => v != null));
    return { method: m, vals, mean, sd };
  });
}

function pixelMean(method, key = 'Pearson') {
  const rows = (S.info.pixel?.rows || []).filter((r) => r.method === method);
  return rows.length ? rows.reduce((a, r) => a + r[key], 0) / rows.length : null;
}

function tileHtml(t) {
  return `<div class="tile"><div class="t-label">${esc(t.label)}</div><div class="t-value">${t.value}</div><div class="t-cmp">${esc(t.cmp)}</div></div>`;
}
function miniHtml(label, value, vs) {
  return `<div class="mini"><div class="m-label">${esc(label)}</div><div class="m-value">${value}${vs ? `<span class="vs">${esc(vs)}</span>` : ''}</div></div>`;
}

function buildResults() {
  const O = S.info.overall, L = S.info.lodo;
  const row = (m) => O?.rows.find((r) => r.method === m);
  const best = bestLearned() || 'Multimodal';
  const tiles = [];
  if (row(best) && row(EGR)) {
    tiles.push({ label: `沒看過的晶片 Vortex-small · ROC-AUC（${disp(best)}）`, value: f3(row(best)['ROC-AUC']),
                 cmp: `eGR ${f3(row(EGR)['ROC-AUC'])} · RUDY ${f3(row(RUDY)?.['ROC-AUC'])}` });
  }
  if (L) {
    const roc = lodoBy('ROC-AUC'), b = roc.find((x) => x.method === best), e = roc.find((x) => x.method === EGR);
    if (b && e) tiles.push({ label: `三個設計輪流測試 · 平均 ROC-AUC（${disp(best)}）`, value: `${f3(b.mean)} <small>± ${f3(b.sd)}</small>`,
                             cmp: `eGR ${f3(e.mean)} ± ${f3(e.sd)}：更準也更穩定` });
  }
  const chosen = S.info.pixel?.chosen?.variant;
  if (chosen && pixelMean(chosen) != null) {
    tiles.push({ label: `預測整張使用率圖 · 像素相關係數（${disp(chosen)}）`, value: f3(pixelMean(chosen)),
                 cmp: `eGR ${f3(pixelMean(EGR))}（1 = 完全一致）` });
  } else if (row(best)?.wins_vs_egr != null) {
    tiles.push({ label: `逐張比較：F1 贏 eGR 的測試版圖（${disp(best)}）`, value: `${row(best).wins_vs_egr} <small>/ ${O.n_layouts}</small>`, cmp: '' });
  }
  $('tiles').innerHTML = tiles.map(tileHtml).join('');

  if (L) {
    $('lodoSub').textContent = '在另外兩個設計上訓練。每個點是一個設計當測試的結果，黑線是三者平均。';
    $('lodoLegend').innerHTML = L.designs.map((d, k) => `<span><i style="background:${DESIGN_COLORS[k]}"></i>${esc(d.name)}（${esc(d.type)}，熱點 ${pct(d.hotspot_rate, 1)}）</span>`).join('')
      + '<span><i class="mean-key"></i>平均</span>';
    document.querySelectorAll('#lodoMetric [data-metric]').forEach((b) => (b.onclick = () => {
      S.metric = b.dataset.metric;
      document.querySelectorAll('#lodoMetric [data-metric]').forEach((x) => x.classList.toggle('active', x === b));
      drawDotplot();
    }));
    drawDotplot();
  } else {
    $('lodoChart').closest('.card').classList.add('hidden');
  }

  buildPixelCard();
  buildUncertaintyCard();

  if (O) {
    $('testTitle').textContent = `完整數據：測試設計 Vortex-small（${O.n_layouts} 張版圖、${O.n_patches.toLocaleString()} 個區塊）`;
    const cols = [['ROC-AUC', 'ROC-AUC'], ['PR-AUC', 'PR-AUC'], ['F1(hotspot)', '熱點 F1'], ['Macro-F1', 'Macro-F1'],
                  ['Precision', 'Precision'], ['Recall', 'Recall'], ['mean_layout_f1', '平均每張 F1']];
    const top = Object.fromEntries(cols.map(([k]) => [k, Math.max(...O.rows.map((r) => r[k] ?? -1))]));
    $('testTable').innerHTML = `<thead><tr><th>方法</th>${cols.map(([, h]) => `<th>${h}</th>`).join('')}<th>贏 eGR 的版圖</th></tr></thead><tbody>`
      + O.rows.map((r) => `<tr class="${r.method === EGR || r.method === RUDY ? 'ref' : ''}"><td>${esc(disp(r.method))}</td>`
        + cols.map(([k]) => `<td class="${r[k] === top[k] ? 'best' : ''}">${f3(r[k])}</td>`).join('')
        + `<td>${r.wins_vs_egr == null ? '（基準）' : `${r.wins_vs_egr} / ${O.n_layouts}`}</td></tr>`).join('') + '</tbody>';
  }
}

function buildPixelCard() {
  const X = S.info.pixel;
  if (!X) { $('pixelCard').classList.add('hidden'); return; }
  const designs = [...new Set(X.rows.map((r) => r.test_design))];
  const methods = [...new Set(X.rows.map((r) => r.method))];
  const get = (m, d, k) => X.rows.find((r) => r.method === m && r.test_design === d)?.[k];
  const u = X.chosen?.variant || methods.find((m) => m.startsWith('U-Net'));
  $('pixelStats').innerHTML = miniHtml('相關係數 Pearson（越高越好）', f3(pixelMean(u)), `eGR ${f3(pixelMean(EGR))}`)
    + miniHtml('結構相似度 SSIM（越高越好）', f3(pixelMean(u, 'SSIM')), `eGR ${f3(pixelMean(EGR, 'SSIM'))}`)
    + miniHtml('誤差 NRMSE（越低越好）', f3(pixelMean(u, 'NRMSE')), `eGR ${f3(pixelMean(EGR, 'NRMSE'))}`);
  const NAMES = { [EGR]: 'eGR（原始值）', 'eGR (linear calibrated)': 'eGR（線性校正）', 'RUDY (linear calibrated)': 'RUDY（線性校正）' };
  const best = (k, low = false) => (low ? Math.min : Math.max)(...methods.map((m) => pixelMean(m, k)));
  const bP = best('Pearson'), bS = best('SSIM'), bN = best('NRMSE', true);
  $('pixelTable').innerHTML = `<thead><tr><th>方法</th>${designs.map((d) => `<th>${esc(d)}</th>`).join('')}<th>Pearson 平均</th><th>SSIM 平均</th><th>NRMSE 平均</th></tr></thead><tbody>`
    + methods.map((m) => `<tr class="${m.startsWith('U-Net') ? '' : 'ref'}"><td>${esc(NAMES[m] || disp(m))}${X.chosen?.variant === m ? ' <span class="muted">（網頁使用）</span>' : ''}</td>`
      + designs.map((d) => `<td>${f3(get(m, d, 'Pearson'))}</td>`).join('')
      + `<td class="${pixelMean(m) === bP ? 'best' : ''}">${f3(pixelMean(m))}</td><td class="${pixelMean(m, 'SSIM') === bS ? 'best' : ''}">${f3(pixelMean(m, 'SSIM'))}</td>`
      + `<td class="${pixelMean(m, 'NRMSE') === bN ? 'best' : ''}">${f3(pixelMean(m, 'NRMSE'))}</td></tr>`).join('') + '</tbody>';
  const notes = ['各設計欄位為該設計當測試時的像素 Pearson（每張版圖分別計算再平均）。'];
  if (methods.includes('U-Net') && methods.includes('U-Net (multimodal)')) {
    const d = pixelMean('U-Net (multimodal)') - pixelMean('U-Net');
    const roc = Object.fromEntries(lodoBy('ROC-AUC').map((r) => [r.method, r.mean]));
    notes.push(`<b>消融：加入設計參數（FiLM）</b>：像素 Pearson 幾乎不變（${d >= 0 ? '+' : ''}${d.toFixed(3)}），`
      + (roc['U-Net'] != null && roc['U-Net (multimodal)'] != null
        ? `但區塊熱點 ROC-AUC 從 ${f3(roc['U-Net'])} 升到 ${f3(roc['U-Net (multimodal)'])}，三個設計都提升。可能是設計參數主要幫助判斷整體使用率高低，而不是空間樣式。`
        : ''));
  }
  if (X.chosen) notes.push(`網頁使用的版本依事先定好的規則選擇（三設計平均像素 Pearson 較高者）：${esc(disp(X.chosen.variant))}。`);
  $('pixelNotes').innerHTML = notes.map((n) => `<li>${n}</li>`).join('');
}

function buildUncertaintyCard() {
  const E = S.info.uncertainty?.eval;
  if (!E) { $('uncCard').classList.add('hidden'); return; }
  const designs = [...new Set(E.map((r) => r.test_design))];
  const U = (d, m) => E.find((r) => r.test_design === d && r.measure === m);
  const mean = (m, k) => designs.reduce((a, d) => a + U(d, m)[k], 0) / designs.length;
  const worst = designs.reduce((a, d) => (U(d, 'margin').AUROC_error < U(a, 'margin').AUROC_error ? d : a), designs[0]);
  $('uncStats').innerHTML = miniHtml('只看最沒把握的 20% 區塊，就涵蓋了', `${pct(mean('disagreement', 'errors_in_top20%'))} 的錯誤`, `對照組 ${pct(mean('margin', 'errors_in_top20%'))}`)
    + miniHtml('用不確定度找出判錯區塊（AUROC）', f3(mean('disagreement', 'AUROC_error')), `對照組 ${f3(mean('margin', 'AUROC_error'))}`)
    + miniHtml(`分布差最多的 ${worst}`, f3(U(worst, 'disagreement').AUROC_error), `對照組 ${f3(U(worst, 'margin').AUROC_error)}`);
  const better = (a, b) => (a > b + 1e-9 ? 'best' : '');
  $('uncTable').innerHTML = '<thead><tr><th>測試設計</th><th>Ensemble 錯誤率</th><th>AUROC：模型間分歧</th><th>AUROC：離閾值遠近</th>'
    + '<th>前 20% 涵蓋錯誤：分歧</th><th>前 20% 涵蓋錯誤：遠近</th></tr></thead><tbody>'
    + designs.map((d) => {
      const a = U(d, 'disagreement'), b = U(d, 'margin');
      return `<tr><td>${esc(d)}</td><td>${pct(a.error_rate, 1)}</td>
        <td class="${better(a.AUROC_error, b.AUROC_error)}">${f3(a.AUROC_error)}</td><td class="${better(b.AUROC_error, a.AUROC_error)}">${f3(b.AUROC_error)}</td>
        <td class="${better(a['errors_in_top20%'], b['errors_in_top20%'])}">${pct(a['errors_in_top20%'])}</td>
        <td class="${better(b['errors_in_top20%'], a['errors_in_top20%'])}">${pct(b['errors_in_top20%'])}</td></tr>`;
    }).join('') + '</tbody>';
}

function drawDotplot() {
  const L = S.info.lodo, rows = lodoBy(S.metric);
  const W = 1000, left = 170, right = 140, rowH = 34, top = 10, axisH = 34;
  const H = top + rows.length * rowH + axisH;
  const all = rows.flatMap((r) => r.vals.filter((v) => v != null));
  const lo = Math.max(0, Math.floor(Math.min(...all) * 20) / 20), hi = 1;
  const x = (v) => left + (v - lo) / (hi - lo) * (W - left - right);
  const ticks = [];
  for (let t = lo; t <= hi + 1e-9; t += 0.05) ticks.push(+t.toFixed(2));
  const step = ticks.length > 12 ? 2 : 1;
  const best = bestLearned();
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="跨設計 ${S.metric}">`;
  ticks.forEach((t, k) => {
    svg += `<line x1="${x(t)}" x2="${x(t)}" y1="${top}" y2="${H - axisH + 4}" stroke="#ebe9e3" stroke-width="1"/>`;
    if (k % step === 0) svg += `<text x="${x(t)}" y="${H - 12}" text-anchor="middle" font-size="12" fill="#7a7e85">${t.toFixed(2)}</text>`;
  });
  const edaCount = rows.filter((r) => r.method === EGR || r.method === RUDY).length;
  if (edaCount) svg += `<rect x="0" y="${top}" width="${W}" height="${edaCount * rowH}" fill="#f6f5f1" rx="6"/>`;
  rows.forEach((r, k) => {
    const cy = top + k * rowH + rowH / 2;
    const isEda = r.method === EGR || r.method === RUDY;
    const vs = r.vals.filter((v) => v != null);
    svg += `<text x="12" y="${cy + 4.5}" font-size="13.5" fill="${isEda ? '#7a7e85' : '#1c1e21'}" font-weight="${r.method === best ? 700 : 400}">${esc(disp(r.method))}${isEda ? ' <tspan font-size="11" fill="#a3a7ad">EDA</tspan>' : ''}</text>`;
    svg += `<line x1="${x(Math.min(...vs))}" x2="${x(Math.max(...vs))}" y1="${cy}" y2="${cy}" stroke="#d2cec3" stroke-width="2" stroke-linecap="round"/>`;
    svg += `<line x1="${x(r.mean)}" x2="${x(r.mean)}" y1="${cy - 10}" y2="${cy + 10}" stroke="#1c1e21" stroke-width="2" stroke-linecap="round"/>`;
    r.vals.forEach((v, d) => { if (v != null) svg += `<circle class="dot" cx="${x(v)}" cy="${cy}" r="6.5" fill="${DESIGN_COLORS[d]}"/>`; });
    r.vals.forEach((v, d) => {
      if (v == null) return;
      svg += `<circle class="hit" cx="${x(v)}" cy="${cy}" r="13" data-tip="${esc(`<div class='t-head'>${disp(r.method)}</div><div class='t-row'><span>${L.designs[d].name}（${L.designs[d].type}）當測試</span></div><div class='t-row'><span>${S.metric}</span><b>${v.toFixed(3)}</b></div><div class='t-row'><span>三設計平均</span><b>${r.mean.toFixed(3)} ± ${r.sd.toFixed(3)}</b></div>`)}"/>`;
    });
    svg += `<text x="${W - 8}" y="${cy + 4.5}" text-anchor="end" font-size="13" fill="#1c1e21" font-weight="${r.method === best ? 700 : 400}" style="font-variant-numeric:tabular-nums">${r.mean.toFixed(3)} <tspan fill="#7a7e85" font-weight="400">± ${r.sd.toFixed(3)}</tspan></text>`;
  });
  svg += `<text x="${W - 8}" y="${H - 12}" text-anchor="end" font-size="12" fill="#7a7e85">平均 ± 標準差</text></svg>`;
  $('lodoChart').innerHTML = svg;
  $('lodoChart').querySelectorAll('.hit').forEach((c) => {
    c.addEventListener('mousemove', (e) => showTip(e, c.dataset.tip));
    c.addEventListener('mouseleave', hideTip);
  });
  $('lodoTable').innerHTML = `<table class="data"><thead><tr><th>方法</th>${L.designs.map((d) => `<th>${esc(d.name)}</th>`).join('')}<th>平均 ± 標準差</th></tr></thead><tbody>`
    + rows.map((r) => `<tr><td>${esc(disp(r.method))}</td>${r.vals.map((v) => `<td>${f3(v)}</td>`).join('')}<td>${f3(r.mean)} ± ${f3(r.sd)}</td></tr>`).join('')
    + '</tbody></table>';
}

// ================================================================== method page
function buildMethodPage() {
  const roc = Object.fromEntries(lodoBy('ROC-AUC').map((r) => [r.method, r]));
  $('methodGrid').innerHTML = S.info.methods.map((m) => {
    const t = m.test?.['ROC-AUC'], c = roc[m.name];
    return `<div class="m-card"><div class="m-top"><span class="m-fam">${esc(m.family)}</span><span class="m-week">${esc(m.course)}</span></div>
      <h3>${esc(disp(m.name))}</h3><p>${esc(m.description)}</p>
      <div class="m-score">Vortex-small <b>${f3(t)}</b>${c ? `　三設計平均 <b>${f3(c.mean)}</b>` : ''}</div></div>`;
  }).join('');
  const L = S.info.lodo;
  if (L) {
    $('dataTable').innerHTML = '<thead><tr><th>設計</th><th>類型</th><th>版圖數</th><th>熱點比例</th><th>網頁模型中的角色</th></tr></thead><tbody>'
      + L.designs.map((d) => `<tr><td>${esc(d.name)}</td><td>${esc(d.type)}</td><td>96</td><td>${pct(d.hotspot_rate, 1)}</td>
        <td>${d.name === 'Vortex-small' ? '測試（完全不參與訓練）' : '訓練 85% ／ 驗證 15%'}</td></tr>`).join('') + '</tbody>';
  }
}

init();
