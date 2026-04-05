/* ============================================================
   NoteSeek — main.js
   Frontend logic for all IRT search engine features
   ============================================================ */

// ------------------------------------------------------------------ //
//  UTILITY HELPERS                                                     //
// ------------------------------------------------------------------ //

const $ = id => document.getElementById(id);
const show = id => { const el = $(id); if (el) el.style.display = ''; };
const hide = id => { const el = $(id); if (el) el.style.display = 'none'; };
const showSpinner = () => $('spinner').style.display = 'flex';
const hideSpinner = () => $('spinner').style.display = 'none';

async function post(url, body) {
  showSpinner();
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    return await res.json();
  } catch (e) {
    console.error('Request failed:', e);
    return null;
  } finally {
    hideSpinner();
  }
}

async function get(url) {
  showSpinner();
  try {
    const res = await fetch(url);
    return await res.json();
  } catch (e) {
    console.error('Request failed:', e);
    return null;
  } finally {
    hideSpinner();
  }
}

function unitClass(unit) {
  const map = {
    'Unit 1': 'u1', 'Unit 2': 'u2', 'Unit 3': 'u3',
    'Unit 4': 'u4', 'Unit 5': 'u5'
  };
  return map[unit] || 'u1';
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;')
            .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ------------------------------------------------------------------ //
//  SIDEBAR & TAB NAVIGATION                                            //
// ------------------------------------------------------------------ //

function toggleSidebar() {
  document.querySelector('.sidebar').classList.toggle('collapsed');
}

function showTab(name) {
  // Hide all panes
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));

  // Show selected pane
  const pane = $('tab-' + name);
  if (pane) pane.classList.add('active');
  const btn = $('nav-' + name);
  if (btn) btn.classList.add('active');

  // Update topbar title
  const titles = {
    search: 'VSM / BM25 Search', boolean: 'Boolean Model',
    lm: 'Language Model', classify: 'Classify Text',
    cluster: 'Cluster Notes', evaluate: 'Evaluate (P/R/F1)',
    upload: 'Upload Notes',
    recommend: 'Recommendation System', notes: 'All Notes'
  };
  $('topbar-title').textContent = titles[name] || 'NoteSeek';

  // Lazy-load all notes tab
  if (name === 'notes') loadAllNotes();
}

// ------------------------------------------------------------------ //
//  LOAD INDEX STATS ON PAGE LOAD                                       //
// ------------------------------------------------------------------ //

async function loadStats() {
  const data = await get('/index_stats');
  if (!data) return;
  $('stat-docs').textContent = data.total_documents;
  $('stat-vocab').textContent = data.vocabulary_size?.toLocaleString();
  $('stat-postings').textContent = data.total_postings?.toLocaleString();
}

// ------------------------------------------------------------------ //
//  MAIN SEARCH (VSM / BM25)                                            //
// ------------------------------------------------------------------ //

function fillSearch(text) {
  $('search-input').value = text;
  doSearch();
}

async function doSearch() {
  const query = $('search-input').value.trim();
  if (!query) return;

  const model = document.querySelector('input[name="model"]:checked')?.value || 'vsm';
  const topK = parseInt($('top-k').value) || 10;

  const data = await post('/search', { query, model, top_k: topK });
  if (!data) return;

  renderQueryTerms(data.query_terms || []);
  renderSearchResults(data.results || [], data.total, data.model_used);
  renderSearchRecommendations(data.recommendations || []);
}

function renderQueryTerms(terms) {
  if (!terms.length) { hide('query-terms-bar'); return; }

  $('query-terms-list').innerHTML = terms.map(t =>
    `<span class="qt-term">${escapeHtml(t.term)} <span style="opacity:0.6">${t.weight}</span></span>`
  ).join('');
  $('query-terms-bar').style.display = 'flex';
}

function renderSearchResults(results, total, model) {
  const area = $('search-results-area');
  if (!results.length) {
    area.innerHTML = `<div class="no-results">No results found. Try different keywords.</div>`;
    return;
  }

  const header = `
    <div class="results-header">
      <div class="results-count">Found <strong>${total}</strong> result${total !== 1 ? 's' : ''}</div>
      <div style="font-size:12px;color:var(--text-3);font-family:var(--mono)">${model}</div>
    </div>`;

  const cards = results.map((r, i) => {
    const scorePercent = Math.min(r.score * 100, 100).toFixed(1);
    const barWidth = Math.min(r.score * 100, 100).toFixed(1);
    return `
      <div class="result-card" onclick="openDoc(${r.id})">
        <div class="result-top">
          <div class="result-rank ${i < 3 ? 'top' : ''}">#${i + 1}</div>
          <div class="result-title">${escapeHtml(r.topic)}</div>
          <div class="result-score">${scorePercent}%</div>
        </div>
        <div class="result-meta">
          <span class="unit-badge ${unitClass(r.unit)}">${escapeHtml(r.unit)}</span>
          <span class="model-badge">${escapeHtml(r.model || '')}</span>
        </div>
        <div class="result-snippet">${escapeHtml(r.snippet || r.content?.slice(0, 200) + '...')}</div>
        <div class="score-bar-wrap">
          <div class="score-bar-track">
            <div class="score-bar-fill" style="width:${barWidth}%"></div>
          </div>
        </div>
      </div>`;
  }).join('');

  area.innerHTML = header + cards;
}

function renderSearchRecommendations(recs) {
  if (!recs.length) { hide('search-recommendations'); return; }
  const list = $('rec-list');
  list.className = 'rec-list';
  list.innerHTML = recs.map(r => `
    <div class="rec-card" onclick="openDoc(${r.id})">
      <div class="rec-topic">${escapeHtml(r.topic)}</div>
      <div class="rec-unit">${escapeHtml(r.unit)}</div>
      <div class="rec-sim">sim: ${r.similarity?.toFixed(3) || r.score?.toFixed(3) || '—'}</div>
    </div>`
  ).join('');
  $('search-recommendations').style.display = 'block';
}

// ------------------------------------------------------------------ //
//  BOOLEAN SEARCH                                                      //
// ------------------------------------------------------------------ //

function fillBool(text) {
  $('bool-input').value = text;
  doBooleanSearch();
}

async function doBooleanSearch() {
  const query = $('bool-input').value.trim();
  if (!query) return;

  const data = await post('/boolean_search', { query });
  if (!data) return;

  // Explanation
  if (data.explanation?.length) {
    $('bool-explanation').innerHTML = data.explanation.map(e =>
      `<div>${escapeHtml(e)}</div>`
    ).join('');
    $('bool-explanation').style.display = 'block';
  }

  const area = $('bool-results-area');
  if (!data.results?.length) {
    area.innerHTML = `<div class="no-results">No documents matched the Boolean query.</div>`;
    return;
  }

  const header = `<div class="results-header"><div class="results-count">Found <strong>${data.total}</strong> document(s)</div></div>`;
  const cards = data.results.map((r, i) => `
    <div class="result-card" onclick="openDoc(${r.id})">
      <div class="result-top">
        <div class="result-rank">#${i + 1}</div>
        <div class="result-title">${escapeHtml(r.topic)}</div>
        <span class="unit-badge ${unitClass(r.unit)}">${escapeHtml(r.unit)}</span>
      </div>
      <div class="result-snippet">${escapeHtml(r.content)}</div>
    </div>`).join('');
  area.innerHTML = header + cards;
}

// ------------------------------------------------------------------ //
//  LANGUAGE MODEL SEARCH                                               //
// ------------------------------------------------------------------ //

async function doLMSearch() {
  const query = $('lm-input').value.trim();
  const smoothing = $('lm-smoothing').value;
  if (!query) return;

  const data = await post('/language_model_search', { query, smoothing });
  if (!data) return;

  const area = $('lm-results-area');
  if (!data.results?.length) {
    area.innerHTML = `<div class="no-results">No results found.</div>`;
    return;
  }

  const header = `
    <div class="results-header">
      <div class="results-count">Found <strong>${data.total}</strong> result(s)</div>
      <div style="font-size:12px;color:var(--text-3);font-family:var(--mono)">Smoothing: ${data.smoothing}</div>
    </div>`;
  const cards = data.results.map((r, i) => `
    <div class="result-card" onclick="openDoc(${r.id})">
      <div class="result-top">
        <div class="result-rank ${i < 3 ? 'top' : ''}">#${i + 1}</div>
        <div class="result-title">${escapeHtml(r.topic)}</div>
        <div class="result-score">${(r.score * 100).toFixed(1)}%</div>
      </div>
      <div class="result-meta">
        <span class="unit-badge ${unitClass(r.unit)}">${escapeHtml(r.unit)}</span>
        <span class="model-badge">${escapeHtml(r.model || '')}</span>
      </div>
      <div class="result-snippet">${escapeHtml(r.content?.slice(0, 200) + '...')}</div>
    </div>`).join('');
  area.innerHTML = header + cards;
}

// ------------------------------------------------------------------ //
//  CLASSIFICATION                                                      //
// ------------------------------------------------------------------ //

async function doClassify() {
  const text = $('classify-input').value.trim();
  const method = $('classify-method').value;
  if (!text) return;

  const data = await post('/classify', { text, method });
  if (!data) return;

  const cl = data.classification;
  let html = '';

  if (cl.final_prediction) {
    html += `
      <div class="final-pred">
        <div class="final-label">Final Prediction (Majority Vote)</div>
        <div class="final-value">${escapeHtml(cl.final_prediction)}</div>
      </div>`;
  }

  html += `<div class="classify-grid">`;

  if (cl.naive_bayes) {
    const conf = cl.naive_bayes.confidence ? (cl.naive_bayes.confidence * 100).toFixed(1) : '—';
    html += `
      <div class="classify-card">
        <div class="model-name">Naive Bayes</div>
        <div class="predicted-label">${escapeHtml(cl.naive_bayes.label)}</div>
        <div style="font-size:12px;color:var(--text-3)">Confidence: ${conf}%</div>
        <div class="conf-bar"><div class="conf-fill" style="width:${conf}%"></div></div>
      </div>`;
  }

  if (cl.svm) {
    html += `
      <div class="classify-card">
        <div class="model-name">SVM</div>
        <div class="predicted-label">${escapeHtml(cl.svm.label)}</div>
        <div style="font-size:12px;color:var(--text-3)">Linear SVC</div>
      </div>`;
  }

  if (cl.knn) {
    const conf2 = (cl.knn.confidence * 100).toFixed(0);
    html += `
      <div class="classify-card">
        <div class="model-name">KNN (k=${cl.knn.k})</div>
        <div class="predicted-label">${escapeHtml(cl.knn.label)}</div>
        <div style="font-size:12px;color:var(--text-3)">Confidence: ${conf2}%</div>
        <div class="conf-bar"><div class="conf-fill" style="width:${conf2}%"></div></div>
      </div>`;
  }

  html += `</div>`;

  // Top terms per class
  if (data.top_terms_per_unit) {
    html += `<div style="margin-top:20px">
      <div class="rec-header">Top Informative Terms per Unit (Naive Bayes)</div>
      <div class="cluster-grid">`;
    for (const [unit, terms] of Object.entries(data.top_terms_per_unit)) {
      html += `<div class="cluster-card">
        <div class="cluster-id">${escapeHtml(unit)}</div>
        <div class="cluster-terms">${terms.map(t => `<span class="cluster-term">${escapeHtml(t)}</span>`).join('')}</div>
      </div>`;
    }
    html += `</div></div>`;
  }

  $('classify-results').innerHTML = html;
}

// ------------------------------------------------------------------ //
//  CLUSTERING                                                          //
// ------------------------------------------------------------------ //

async function doCluster() {
  const method = document.querySelector('input[name="cluster-method"]:checked')?.value || 'kmeans';
  const k = parseInt($('cluster-k').value) || 5;

  const data = await post('/cluster', { method, k });
  if (!data || data.error) {
    $('cluster-results').innerHTML = `<div class="no-results">${data?.error || 'Error during clustering'}</div>`;
    return;
  }

  let html = `
    <div class="results-header">
      <div class="results-count">Method: <strong>${escapeHtml(data.method)}</strong> · K = <strong>${data.k || data.n_clusters}</strong></div>
      ${data.silhouette_score != null ? `<div style="font-size:12px;color:var(--green);font-family:var(--mono)">Silhouette: ${data.silhouette_score}</div>` : ''}
    </div>
    <div class="cluster-grid">`;

  for (const cluster of (data.clusters || [])) {
    const docs = cluster.documents || [];
    html += `
      <div class="cluster-card">
        <div class="cluster-id">Cluster ${cluster.cluster_id} · ${cluster.size} notes</div>
        ${cluster.top_terms ? `<div class="cluster-terms">${cluster.top_terms.map(t => `<span class="cluster-term">${escapeHtml(t)}</span>`).join('')}</div>` : ''}
        ${cluster.dominant_unit ? `<div style="font-size:11px;color:var(--text-3);margin-bottom:6px">Dominant: ${escapeHtml(cluster.dominant_unit)}</div>` : ''}
        <div class="cluster-docs">
          ${docs.slice(0, 5).map(d => `
            <div class="cluster-doc-item">
              <span class="unit-badge ${unitClass(d.unit)}" style="font-size:10px">${escapeHtml(d.unit)}</span>
              <span style="font-size:12px;color:var(--text-2)">${escapeHtml(d.topic)}</span>
            </div>`).join('')}
          ${docs.length > 5 ? `<div style="font-size:11px;color:var(--text-3);margin-top:4px">+${docs.length - 5} more</div>` : ''}
        </div>
      </div>`;
  }

  html += `</div>`;
  $('cluster-results').innerHTML = html;
}

// ------------------------------------------------------------------ //
//  EVALUATION                                                          //
// ------------------------------------------------------------------ //

async function doEvaluate() {
  const query = $('eval-input').value.trim();
  if (!query) return;

  const data = await post('/evaluate', { query, relevant_ids: [] });
  if (!data) return;

  const m = data.metrics;
  let html = `
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-label">Precision</div>
        <div class="metric-value">${(m.precision * 100).toFixed(1)}%</div>
        <div class="metric-desc">Of retrieved, how many relevant?</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Recall</div>
        <div class="metric-value">${(m.recall * 100).toFixed(1)}%</div>
        <div class="metric-desc">Of relevant, how many retrieved?</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">F1 Score</div>
        <div class="metric-value">${(m.f1_score * 100).toFixed(1)}%</div>
        <div class="metric-desc">Harmonic mean of P & R</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">F2 Score</div>
        <div class="metric-value">${(m.f2_score * 100).toFixed(1)}%</div>
        <div class="metric-desc">Recall weighted 2×</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">F0.5 Score</div>
        <div class="metric-value">${(m.f0_5_score * 100).toFixed(1)}%</div>
        <div class="metric-desc">Precision weighted 2×</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Avg Precision</div>
        <div class="metric-value">${(m.average_precision * 100).toFixed(1)}%</div>
        <div class="metric-desc">Area under P-R curve</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">P@5</div>
        <div class="metric-value">${(m.precision_at_5 * 100).toFixed(1)}%</div>
        <div class="metric-desc">Precision at rank 5</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">NDCG</div>
        <div class="metric-value">${(m.ndcg * 100).toFixed(1)}%</div>
        <div class="metric-desc">Normalized DCG</div>
      </div>
    </div>
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:8px">
      <div style="font-size:13px;color:var(--text-3)">True Positives: <strong style="color:var(--green)">${m.true_positives}</strong></div>
      <div style="font-size:13px;color:var(--text-3)">False Positives: <strong style="color:var(--coral)">${m.false_positives}</strong></div>
      <div style="font-size:13px;color:var(--text-3)">False Negatives: <strong style="color:var(--amber)">${m.false_negatives}</strong></div>
      <div style="font-size:13px;color:var(--text-3)">Retrieved: <strong>${m.retrieved_count}</strong></div>
      <div style="font-size:13px;color:var(--text-3)">Relevant (auto): <strong>${m.relevant_count}</strong></div>
    </div>`;

  $('eval-results').innerHTML = html;
}

// ------------------------------------------------------------------ //
//  RECOMMENDATIONS                                                     //
// ------------------------------------------------------------------ //

async function doRecommend() {
  const query = $('rec-input').value.trim();
  const mode = $('rec-mode').value;

  const data = await post('/recommend', { query, mode });
  if (!data) return;

  const recs = data.recommendations || [];
  const area = $('rec-results');

  if (!recs.length) {
    area.innerHTML = `<div class="no-results">No recommendations found.</div>`;
    return;
  }

  const header = `<div class="results-header"><div class="results-count">Found <strong>${recs.length}</strong> recommendation(s) · Mode: ${mode}</div></div>`;
  const cards = recs.map((r, i) => `
    <div class="result-card" onclick="openDoc(${r.id})">
      <div class="result-top">
        <div class="result-rank">#${i + 1}</div>
        <div class="result-title">${escapeHtml(r.topic)}</div>
        <div class="result-score">${r.score?.toFixed(3) || r.similarity?.toFixed(3) || '—'}</div>
      </div>
      <div class="result-meta">
        <span class="unit-badge ${unitClass(r.unit)}">${escapeHtml(r.unit)}</span>
        <span class="model-badge">${escapeHtml(r.reason || '')}</span>
      </div>
    </div>`).join('');
  area.innerHTML = header + cards;
}

// ------------------------------------------------------------------ //
//  ALL NOTES TAB                                                       //
// ------------------------------------------------------------------ //

let allDocs = [];
let activeFilter = 'all';

async function loadAllNotes() {
  if (allDocs.length) { renderNotes(allDocs); return; }
  const data = await get('/documents');
  if (!data) return;
  allDocs = data.documents || [];
  renderUnitFilters();
  renderNotes(allDocs);
}

function renderUnitFilters() {
  const units = ['all', 'Unit 1', 'Unit 2', 'Unit 3', 'Unit 4', 'Unit 5'];
  $('unit-filter').innerHTML = units.map(u => `
    <button class="filter-btn ${u === 'all' ? 'active' : ''}"
      onclick="filterNotes('${u}')">${u === 'all' ? 'All' : u}</button>`
  ).join('');
}

function filterNotes(unit) {
  activeFilter = unit;
  document.querySelectorAll('.filter-btn').forEach(b => {
    b.classList.toggle('active', b.textContent.trim() === (unit === 'all' ? 'All' : unit));
  });
  const filtered = unit === 'all' ? allDocs : allDocs.filter(d => d.unit === unit);
  renderNotes(filtered);
}

function renderNotes(docs) {
  if (!docs.length) {
    $('all-notes-list').innerHTML = `<div class="empty-state"><div class="empty-icon">📭</div><div class="empty-title">No notes found</div></div>`;
    return;
  }
  $('all-notes-list').innerHTML = `
    <div class="notes-grid">
      ${docs.map(d => `
        <div class="note-card" onclick="openDoc(${d.id})">
          <div class="result-meta" style="margin-bottom:6px">
            <span class="unit-badge ${unitClass(d.unit)}">${escapeHtml(d.unit)}</span>
          </div>
          <div class="note-topic">${escapeHtml(d.topic)}</div>
          <div class="note-preview">${escapeHtml(d.content_preview)}</div>
        </div>`).join('')}
    </div>`;
}

// ------------------------------------------------------------------ //
//  DOCUMENT MODAL                                                      //
// ------------------------------------------------------------------ //

async function openDoc(docId) {
  const data = await get(`/document/${docId}`);
  if (!data || !data.document) return;

  const doc = data.document;
  const similar = data.similar || [];

  let html = `
    <div class="modal-title">${escapeHtml(doc.topic)}</div>
    <div class="modal-meta">
      <span class="unit-badge ${unitClass(doc.unit)}">${escapeHtml(doc.unit)}</span>
      &nbsp;&nbsp;${escapeHtml(doc.subject || '')}
    </div>
    <div class="modal-body">${escapeHtml(doc.content)}</div>`;

  if (similar.length) {
    html += `
      <div class="modal-sims">
        <div class="modal-sims-label">Similar Notes</div>
        <div class="rec-list">
          ${similar.map(s => `
            <div class="rec-card" onclick="openDoc(${s.id})">
              <div class="rec-topic">${escapeHtml(s.topic)}</div>
              <div class="rec-unit">${escapeHtml(s.unit)}</div>
              <div class="rec-sim">sim: ${s.similarity?.toFixed(3) || '—'}</div>
            </div>`).join('')}
        </div>
      </div>`;
  }

  $('modal-content').innerHTML = html;
  $('modal-overlay').classList.add('open');
}

function closeModal() {
  $('modal-overlay').classList.remove('open');
}

// ------------------------------------------------------------------ //
//  FILE UPLOAD                                                         //
// ------------------------------------------------------------------ //

let selectedFile = null;

function handleFileSelect(input) {
  if (input.files && input.files[0]) {
    setFile(input.files[0]);
  }
}

function handleDrop(event) {
  event.preventDefault();
  $('drop-zone').classList.remove('dragover');
  const file = event.dataTransfer.files[0];
  if (file) setFile(file);
}

function setFile(file) {
  const allowed = ['pdf', 'pptx', 'ppt'];
  const ext = file.name.split('.').pop().toLowerCase();
  if (!allowed.includes(ext)) {
    alert('Only PDF and PPTX files are supported.');
    return;
  }
  selectedFile = file;
  $('file-icon').textContent = ext === 'pdf' ? '📄' : '📊';
  $('file-name-display').textContent = file.name;
  $('file-size-display').textContent = (file.size / 1024).toFixed(1) + ' KB';
  $('file-preview').style.display = 'flex';
  $('upload-btn').disabled = false;
}

function clearFile() {
  selectedFile = null;
  $('file-input').value = '';
  $('file-preview').style.display = 'none';
  $('upload-btn').disabled = true;
}

async function doUpload() {
  if (!selectedFile) return;

  const subject = $('upload-subject').value.trim() || 'Uploaded Notes';

  // Show progress
  $('upload-progress').style.display = 'block';
  $('prog-label').textContent = 'Extracting text from file...';
  $('upload-btn').disabled = true;

  const formData = new FormData();
  formData.append('file', selectedFile);
  formData.append('subject', subject);

  showSpinner();
  try {
    const res = await fetch('/upload', { method: 'POST', body: formData });
    const data = await res.json();

    $('upload-progress').style.display = 'none';

    if (data.error) {
      $('upload-results').innerHTML = `<div class="no-results">❌ ${escapeHtml(data.error)}</div>`;
      $('upload-btn').disabled = false;
      return;
    }

    renderUploadResults(data);
    loadStats(); // refresh sidebar stats
    clearFile();
    loadUploadHistory();

  } catch (e) {
    $('upload-progress').style.display = 'none';
    $('upload-results').innerHTML = `<div class="no-results">❌ Upload failed. Make sure app.py is running.</div>`;
    $('upload-btn').disabled = false;
  } finally {
    hideSpinner();
  }
}

function renderUploadResults(data) {
  const docs = data.docs || [];
  let html = `
    <div class="upload-success-banner">
      <div class="upload-success-icon">✅</div>
      <div>
        <div class="upload-success-title">${escapeHtml(data.filename)} indexed successfully!</div>
        <div class="upload-success-sub">${data.sections_extracted} sections extracted · Subject: ${escapeHtml(data.subject)} · Index now has ${data.new_index_stats?.total_documents} documents</div>
      </div>
    </div>
    <div class="results-header">
      <div class="results-count">Extracted <strong>${docs.length}</strong> section(s) — all searchable now</div>
    </div>`;

  for (const doc of docs) {
    html += `
      <div class="extracted-doc-card" onclick="openDoc(${doc.id})">
        <div class="extracted-meta">
          <span class="source-badge">${escapeHtml(doc.source?.split('.').pop() || 'file')}</span>
          <span class="unit-badge ${unitClass(doc.unit)}">${escapeHtml(doc.unit)}</span>
          <span class="page-badge">${doc.page ? (doc.source?.endsWith('.pdf') || doc.source?.endsWith('.PDF') ? 'Page ' : 'Slide ') + doc.page : ''}</span>
        </div>
        <div class="note-topic">${escapeHtml(doc.topic)}</div>
        <div class="note-preview">${escapeHtml(doc.content_preview)}</div>
      </div>`;
  }

  $('upload-results').innerHTML = html;
}

async function loadUploadHistory() {
  const data = await get('/uploaded_docs');
  if (!data || !data.upload_history?.length) return;

  let html = `
    <div class="history-section">
      <div class="history-header">Upload History</div>`;
  for (const f of data.upload_history) {
    const icon = f.type === '.pdf' ? '📄' : '📊';
    html += `
      <div class="history-file">
        <span style="font-size:18px">${icon}</span>
        <span class="history-name">${escapeHtml(f.filename)}</span>
        <span class="history-count">${f.sections_extracted} sections</span>
      </div>`;
  }
  html += `</div>`;
  $('uploaded-history').innerHTML = html;
}

// ------------------------------------------------------------------ //
//  INIT                                                                //
// ------------------------------------------------------------------ //

document.addEventListener('DOMContentLoaded', () => {
  loadStats();
});