const API = '';
let selectedKB = null;
let searchKBSet = new Set(); // Multi-KB search selection

// Helper: unwrap unified {code, message, data} response
function unwrap(resp) {
    if (resp && typeof resp === 'object' && 'code' in resp && 'data' in resp) {
        return resp.data;
    }
    return resp; // fallback for endpoints that don't use new format (e.g., search)
}
let kbCheckedSet = new Set();
let kbBatchMode = false;

// ==================== Health Check ====================
async function checkHealth() {
    try {
        const r = await fetch(`${API}/api/health`);
        const raw = await r.json();
        const d = unwrap(raw);
        const dot = document.getElementById('statusDot');
        const text = document.getElementById('statusText');
        if (r.ok && d) {
            dot.className = 'status-dot';
            text.textContent = `ES: ${d.es_status} | Graph: ${d.graph_enabled ? '✓' : '✗'}`;
        } else {
            dot.className = 'status-dot offline';
            text.textContent = raw.message || '服务异常';
        }
    } catch {
        document.getElementById('statusDot').className = 'status-dot offline';
        document.getElementById('statusText').textContent = '无法连接';
    }
}

// ==================== Knowledge Base ====================
// Map kb_id → display_name for lookups
let kbDisplayNames = {};

async function loadKBs() {
    try {
        const folderFilter = document.getElementById('folderFilter');
        const selectedFolder = folderFilter ? folderFilter.value : '';
        const url = selectedFolder ? `${API}/api/knowledgebase?folder=${encodeURIComponent(selectedFolder)}` : `${API}/api/knowledgebase`;
        const r = await fetch(url);
        const raw = await r.json();
        const d = unwrap(raw) || {};
        const list = document.getElementById('kbList');
        const kbs = d.knowledgebases || [];

        // Populate folder dropdown (only on full load, not filtered)
        if (!selectedFolder && folderFilter) {
            const folders = new Set();
            kbs.forEach(kb => { if (kb.folder && kb.folder !== '/') folders.add(kb.folder); });
            const prevVal = folderFilter.value;
            folderFilter.innerHTML = '<option value="">📁 全部文件夹</option>' +
                Array.from(folders).sort().map(f => `<option value="${escapeHtml(f)}">${escapeHtml(f)}</option>`).join('');
            folderFilter.value = prevVal;
        }

        if (kbs.length === 0) {
            list.innerHTML = '<li class="empty-state" style="padding:20px"><span style="color:var(--text-muted)">暂无知识库，请先创建</span></li>';
            document.getElementById('kbBatchToggle').style.display = 'none';
            document.getElementById('kbToolbar').style.display = 'none';
            return;
        }

        // Build display name map
        kbDisplayNames = {};
        kbs.forEach(kb => {
            kbDisplayNames[kb.kb_id] = kb.display_name || kb.kb_id;
        });

        // Clean up checked set
        const currentKBIds = new Set(kbs.map(kb => kb.kb_id));
        for (const id of kbCheckedSet) {
            if (!currentKBIds.has(id)) kbCheckedSet.delete(id);
        }

        document.getElementById('kbBatchToggle').style.display = 'inline-flex';
        if (!kbBatchMode) {
            document.getElementById('kbToolbar').style.display = 'none';
        }

        list.innerHTML = kbs.map(kb => {
            const name = escapeHtml(kb.display_name || kb.kb_id);
            const folder = kb.folder && kb.folder !== '/' ? `<span style="font-size:10px;color:var(--text-muted);margin-left:4px">${escapeHtml(kb.folder)}</span>` : '';
            const checked = kbCheckedSet.has(kb.kb_id) ? 'checked' : '';
            return `
            <li class="kb-item ${selectedKB === kb.kb_id ? 'active' : ''}"
                data-kbid="${kb.kb_id}">
                <input type="checkbox" class="kb-checkbox" data-action="check" ${checked}>
                <span class="name">📂 ${name}${folder}</span>
                <span class="count">${kb.doc_count} 条</span>
                <button class="browse-btn" data-action="browse" title="浏览分块">🔎</button>
                <button class="delete-btn" data-action="delete" title="删除">✕</button>
            </li>
        `}).join('');

        updateKBBatchUI(kbs.length);
        renderSearchKBChips(kbs);

        // Event delegation for KB list
        list.onclick = function(e) {
            const li = e.target.closest('.kb-item');
            if (!li) return;
            const kbId = li.getAttribute('data-kbid');

            const checkbox = e.target.closest('[data-action="check"]');
            if (checkbox) { e.stopPropagation(); toggleKBCheck(kbId, checkbox.checked, kbs.length); return; }

            const browseBtn = e.target.closest('[data-action="browse"]');
            if (browseBtn) { e.stopPropagation(); openChunkModal(kbId); return; }

            const deleteBtn = e.target.closest('[data-action="delete"]');
            if (deleteBtn) { e.stopPropagation(); deleteKB(kbId); return; }

            selectKB(kbId);
        };
    } catch (e) {
        toast('加载知识库失败: ' + e.message, 'error');
    }
}

async function createKB() {
    const input = document.getElementById('kbNameInput');
    const name = input.value.trim();
    if (!name) return toast('请输入知识库名称', 'error');

    try {
        const r = await fetch(`${API}/api/knowledgebase`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ kb_id: name })
        });
        const raw = await r.json();
        const d = unwrap(raw) || {};
        toast(`知识库 "${name}" ${raw.message === 'created' ? '创建成功' : '已存在'}`, 'success');
        input.value = '';
        selectedKB = d.kb_id;
        loadKBs();
    } catch (e) {
        toast('创建失败: ' + e.message, 'error');
    }
}

function selectKB(id) {
    selectedKB = id;
    // Also add to search selection
    searchKBSet.add(id);
    loadKBs();
    const name = kbDisplayNames[id] || id;
    toast(`已选择知识库: ${name}`, 'info');
}

async function deleteKB(id) {
    const name = kbDisplayNames[id] || id;
    document.getElementById('confirmMsg').innerHTML =
        `确定要删除知识库 <strong>"${escapeHtml(name)}"</strong> 及其所有数据吗？<br>此操作不可撤销！`;
    document.getElementById('confirmOverlay').classList.add('active');

    const btn = document.getElementById('confirmDeleteBtn');
    const newBtn = btn.cloneNode(true);
    btn.parentNode.replaceChild(newBtn, btn);
    newBtn.disabled = false;
    newBtn.textContent = '🗑️ 确认删除';
    newBtn.onclick = async function() {
        newBtn.disabled = true;
        newBtn.textContent = '删除中...';
        try {
            await fetch(`${API}/api/knowledgebase/${id}`, { method: 'DELETE' });
            closeConfirm();
            if (selectedKB === id) selectedKB = null;
            toast(`已删除: ${name}`, 'success');
            loadKBs();
        } catch (e) {
            closeConfirm();
            toast('删除失败: ' + e.message, 'error');
        }
    };
}

// ==================== Batch Delete ====================
function toggleBatchMode() {
    kbBatchMode = !kbBatchMode;
    const card = document.getElementById('kbCard');
    const toolbar = document.getElementById('kbToolbar');
    const toggle = document.getElementById('kbBatchToggle');

    if (kbBatchMode) {
        card.classList.add('kb-batch-mode');
        toolbar.style.display = 'flex';
        toggle.textContent = '✕ 取消';
        toggle.classList.add('active');
    } else {
        card.classList.remove('kb-batch-mode');
        toolbar.style.display = 'none';
        toggle.textContent = '🗑️ 批量管理';
        toggle.classList.remove('active');
        // Clear selections
        kbCheckedSet.clear();
        document.querySelectorAll('#kbList .kb-checkbox').forEach(cb => cb.checked = false);
        updateKBBatchUI(0);
    }
}

function toggleKBCheck(kbId, isChecked, totalCount) {
    if (isChecked) {
        kbCheckedSet.add(kbId);
    } else {
        kbCheckedSet.delete(kbId);
    }
    updateKBBatchUI(totalCount);
}

function toggleKBSelectAll() {
    const selectAll = document.getElementById('kbSelectAll');
    const checkboxes = document.querySelectorAll('#kbList .kb-checkbox');
    checkboxes.forEach(cb => {
        const kbId = cb.closest('.kb-item').getAttribute('data-kbid');
        cb.checked = selectAll.checked;
        if (selectAll.checked) {
            kbCheckedSet.add(kbId);
        } else {
            kbCheckedSet.delete(kbId);
        }
    });
    updateKBBatchUI(checkboxes.length);
}

function updateKBBatchUI(totalCount) {
    const count = kbCheckedSet.size;
    const btn = document.getElementById('kbBatchDeleteBtn');
    const countSpan = document.getElementById('kbSelectedCount');
    const selectAll = document.getElementById('kbSelectAll');
    const label = document.getElementById('kbSelectAllLabel');

    countSpan.textContent = count;
    btn.classList.toggle('visible', count > 0);
    selectAll.checked = count > 0 && count === totalCount;
    label.textContent = selectAll.checked ? '取消全选' : '全选';
}

async function batchDeleteKBs() {
    const ids = Array.from(kbCheckedSet);
    if (ids.length === 0) return;

    const names = ids.map(id => kbDisplayNames[id] || id);
    const preview = names.length <= 3
        ? names.map(n => `"${n}"`).join('、')
        : names.slice(0, 3).map(n => `"${n}"`).join('、') + ` 等 ${names.length} 个`;

    document.getElementById('confirmMsg').innerHTML =
        `确定要批量删除 <strong>${preview}</strong> 知识库及其所有数据吗？<br>此操作不可撤销！`;
    document.getElementById('confirmOverlay').classList.add('active');

    const btn = document.getElementById('confirmDeleteBtn');
    const newBtn = btn.cloneNode(true);
    btn.parentNode.replaceChild(newBtn, btn);
    newBtn.disabled = false;
    newBtn.textContent = `🗑️ 确认删除 (${ids.length})`;
    newBtn.onclick = async function() {
        newBtn.disabled = true;
        newBtn.textContent = '批量删除中...';
        try {
            const r = await fetch(`${API}/api/knowledgebase/batch_delete`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ kb_ids: ids }),
            });
            const raw = await r.json();
            const d = unwrap(raw) || {};
            closeConfirm();
            if (r.ok) {
                toast(`✅ 批量删除完成：成功 ${d.deleted} 个，失败 ${d.failed} 个`, 'success');
                kbCheckedSet.clear();
                if (ids.includes(selectedKB)) selectedKB = null;
                kbBatchMode = true; toggleBatchMode(); // exit batch mode
                loadKBs();
            } else {
                toast(`⚠️ 批量删除失败: ${raw.message || '未知错误'}`, 'error');
            }
        } catch (e) {
            closeConfirm();
            toast('❌ 批量删除失败: ' + e.message, 'error');
        }
    };
}

// ==================== File Upload ====================
const fileInput = document.getElementById('fileInput');
const uploadZone = document.getElementById('uploadZone');

['dragover', 'dragenter'].forEach(e => {
    uploadZone.addEventListener(e, ev => { ev.preventDefault(); uploadZone.classList.add('dragover'); });
});

['dragleave', 'drop'].forEach(e => {
    uploadZone.addEventListener(e, ev => { ev.preventDefault(); uploadZone.classList.remove('dragover'); });
});

uploadZone.addEventListener('drop', ev => {
    if (ev.dataTransfer.files.length) uploadFiles(ev.dataTransfer.files);
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length) uploadFiles(fileInput.files);
});

async function uploadFiles(files) {
    if (!selectedKB) return toast('请先选择一个知识库', 'error');

    const bar = document.getElementById('uploadProgress');
    const fill = document.getElementById('uploadFill');
    const status = document.getElementById('uploadStatus');

    bar.classList.add('active');

    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        fill.style.width = `${((i) / files.length) * 100}%`;
        status.textContent = `上传中: ${file.name} (${i + 1}/${files.length})`;

        const formData = new FormData();
        formData.append('kb_id', selectedKB);
        formData.append('file', file);

        try {
            const r = await fetch(`${API}/api/document/upload`, {
                method: 'POST',
                body: formData,
            });
            const raw = await r.json();
            const d = unwrap(raw) || {};

            if (r.ok) {
                toast(`✅ ${file.name}: ${d.chunks} 个 chunks`, 'success');
            } else {
                toast(`⚠️ ${file.name}: ${raw.message || '上传异常'}`, 'error');
            }
        } catch (e) {
            toast(`❌ ${file.name} 上传失败: ${e.message}`, 'error');
        }
    }

    fill.style.width = '100%';
    status.textContent = `全部完成！共 ${files.length} 个文件`;
    setTimeout(() => { bar.classList.remove('active'); status.textContent = ''; }, 3000);
    fileInput.value = '';
    loadKBs();
}

// ==================== Multi-KB Search Selection ====================
function toggleKBSelector() {
    const el = document.getElementById('kbSelector');
    el.classList.toggle('collapsed');
}

function updateKBSummary() {
    const summary = document.getElementById('kbSelectedSummary');
    if (!summary) return;
    const n = searchKBSet.size;
    summary.textContent = n > 0 ? `(已选 ${n} 个)` : '';
}

function renderSearchKBChips(kbs) {
    const container = document.getElementById('searchKBChips');
    const selectAllBtn = document.getElementById('searchKBSelectAll');
    const clearAllBtn = document.getElementById('searchKBClearAll');
    if (!container) return;

    // Clean up stale IDs
    const currentIds = new Set(kbs.map(kb => kb.kb_id));
    for (const id of searchKBSet) {
        if (!currentIds.has(id)) searchKBSet.delete(id);
    }

    if (kbs.length === 0) {
        container.innerHTML = '<span class="kb-chip-empty">请先创建知识库</span>';
        selectAllBtn.style.display = 'none';
        clearAllBtn.style.display = 'none';
        return;
    }

    container.innerHTML = kbs.map(kb => {
        const name = escapeHtml(kb.display_name || kb.kb_id);
        const isSelected = searchKBSet.has(kb.kb_id);
        return `<span class="kb-chip ${isSelected ? 'active' : ''}" 
                      data-kbid="${kb.kb_id}" 
                      onclick="toggleSearchKB('${kb.kb_id}')" 
                      title="${name} (${kb.doc_count} 条)">
                    📂 ${name}
                    <span class="kb-chip-count">${kb.doc_count}</span>
                </span>`;
    }).join('');

    // Toggle button visibility
    const allSelected = searchKBSet.size === kbs.length;
    selectAllBtn.style.display = allSelected ? 'none' : 'inline-block';
    clearAllBtn.style.display = searchKBSet.size > 0 ? 'inline-block' : 'none';
    updateKBSummary();
}

function toggleSearchKB(kbId) {
    if (searchKBSet.has(kbId)) {
        searchKBSet.delete(kbId);
    } else {
        searchKBSet.add(kbId);
    }
    // Update chip active state without full reload
    document.querySelectorAll('#searchKBChips .kb-chip').forEach(chip => {
        const id = chip.getAttribute('data-kbid');
        chip.classList.toggle('active', searchKBSet.has(id));
    });
    // Update buttons
    const totalChips = document.querySelectorAll('#searchKBChips .kb-chip').length;
    const allSelected = searchKBSet.size === totalChips;
    document.getElementById('searchKBSelectAll').style.display = allSelected ? 'none' : 'inline-block';
    document.getElementById('searchKBClearAll').style.display = searchKBSet.size > 0 ? 'inline-block' : 'none';
    updateKBSummary();
}

function toggleSearchKBAll() {
    document.querySelectorAll('#searchKBChips .kb-chip').forEach(chip => {
        const id = chip.getAttribute('data-kbid');
        searchKBSet.add(id);
        chip.classList.add('active');
    });
    document.getElementById('searchKBSelectAll').style.display = 'none';
    document.getElementById('searchKBClearAll').style.display = 'inline-block';
    toast(`已选择全部 ${searchKBSet.size} 个知识库`, 'info');
    updateKBSummary();
}

function clearSearchKBAll() {
    searchKBSet.clear();
    document.querySelectorAll('#searchKBChips .kb-chip').forEach(chip => {
        chip.classList.remove('active');
    });
    document.getElementById('searchKBSelectAll').style.display = 'inline-block';
    document.getElementById('searchKBClearAll').style.display = 'none';
    toast('已清空检索范围', 'info');
    updateKBSummary();
}

// ==================== Search ====================
function toggleOpt(el) { el.classList.toggle('active'); }

// ==================== Parameter Panel ====================
let searchParams = { topK: 5, threshold: 0.20, vecWeight: 0.30 };

function toggleParamPanel() {
    const panel = document.getElementById('paramPanel');
    const toggle = document.getElementById('toggleParams');
    panel.classList.toggle('active');
    toggle.classList.toggle('active');
}

function updateParam(key, val) {
    if (key === 'topK') {
        searchParams.topK = parseInt(val);
        document.getElementById('valTopK').textContent = val;
    } else if (key === 'threshold') {
        searchParams.threshold = parseInt(val) / 100;
        document.getElementById('valThreshold').textContent = searchParams.threshold.toFixed(2);
    } else if (key === 'vecWeight') {
        searchParams.vecWeight = parseInt(val) / 100;
        document.getElementById('valVecWeight').textContent = searchParams.vecWeight.toFixed(2);
    }
    localStorage.setItem('ragflow_params', JSON.stringify(searchParams));
}

function resetParams() {
    searchParams = { topK: 5, threshold: 0.20, vecWeight: 0.30 };
    document.getElementById('paramTopK').value = 5;
    document.getElementById('paramThreshold').value = 20;
    document.getElementById('paramVecWeight').value = 30;
    document.getElementById('valTopK').textContent = '5';
    document.getElementById('valThreshold').textContent = '0.20';
    document.getElementById('valVecWeight').textContent = '0.30';
    localStorage.removeItem('ragflow_params');
    toast('参数已重置', 'info');
}

function loadSavedParams() {
    try {
        const saved = JSON.parse(localStorage.getItem('ragflow_params'));
        if (saved) {
            searchParams = { ...searchParams, ...saved };
            document.getElementById('paramTopK').value = searchParams.topK;
            document.getElementById('paramThreshold').value = Math.round(searchParams.threshold * 100);
            document.getElementById('paramVecWeight').value = Math.round(searchParams.vecWeight * 100);
            document.getElementById('valTopK').textContent = searchParams.topK;
            document.getElementById('valThreshold').textContent = searchParams.threshold.toFixed(2);
            document.getElementById('valVecWeight').textContent = searchParams.vecWeight.toFixed(2);
        }
    } catch {}
}

// ==================== Search History ====================
const MAX_HISTORY = 10;

function getHistory() {
    try { return JSON.parse(localStorage.getItem('ragflow_history') || '[]'); }
    catch { return []; }
}

function addToHistory(query) {
    let history = getHistory().filter(h => h !== query);
    history.unshift(query);
    if (history.length > MAX_HISTORY) history = history.slice(0, MAX_HISTORY);
    localStorage.setItem('ragflow_history', JSON.stringify(history));
    renderHistory();
}

function renderHistory() {
    const history = getHistory();
    const panel = document.getElementById('historyPanel');
    const tags = document.getElementById('historyTags');
    if (history.length === 0) { panel.style.display = 'none'; return; }
    panel.style.display = 'block';
    tags.innerHTML = history.map((q, i) => {
        const escaped = escapeHtml(q).replace(/'/g, "\\'");
        return `<span class="history-tag" title="${escapeHtml(q)}">
            <span class="history-tag-text" onclick="useHistory('${escaped}')">${escapeHtml(q)}</span>
            <span class="history-tag-delete" onclick="event.stopPropagation();removeHistory(${i})" title="删除">✕</span>
        </span>`;
    }).join('');
}

function useHistory(query) {
    document.getElementById('searchInput').value = query;
    doSearch();
}

function removeHistory(index) {
    let history = getHistory();
    history.splice(index, 1);
    localStorage.setItem('ragflow_history', JSON.stringify(history));
    renderHistory();
}

function clearHistory() {
    localStorage.removeItem('ragflow_history');
    renderHistory();
    toast('搜索历史已清空', 'info');
}

async function doSearch() {
    const q = document.getElementById('searchInput').value.trim();
    if (!q) return toast('请输入问题', 'error');
    if (searchKBSet.size === 0) return toast('请先选择检索的知识库', 'error');

    addToHistory(q);

    const btn = document.getElementById('searchBtn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> 检索中...';

    const useGraph = document.getElementById('toggleGraph').classList.contains('active');
    const useCRAG = document.getElementById('toggleCRAG').classList.contains('active');
    const useWebSearch = document.getElementById('toggleWebSearch').classList.contains('active');

    // Choose endpoint based on whether GraphRAG/CRAG/WebSearch is enabled
    const endpoint = (useGraph || useCRAG || useWebSearch) ? '/api/graph_retrieval' : '/api/retrieval';

    const body = {
        question: q,
        kb_ids: Array.from(searchKBSet),
        top_k: searchParams.topK,
        similarity_threshold: searchParams.threshold,
        vector_similarity_weight: searchParams.vecWeight,
    };

    if (endpoint === '/api/graph_retrieval') {
        body.enable_graph = useGraph;
        body.enable_crag = useCRAG;
        body.enable_web_search = useWebSearch;
    }

    try {
        const r = await fetch(`${API}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const d = await r.json();
        renderResults(d);
    } catch (e) {
        toast('检索失败: ' + e.message, 'error');
        document.getElementById('resultsArea').innerHTML =
            '<div class="empty-state"><div class="icon">❌</div><div>检索失败，请检查服务状态</div></div>';
    } finally {
        btn.disabled = false;
        btn.innerHTML = '🔍 检索';
        loadPerfStats();
    }
}

function renderResults(data) {
    const area = document.getElementById('resultsArea');
    const cragBar = document.getElementById('cragBar');

    // CRAG indicator
    if (data.crag_score && data.crag_score !== 'disabled' && data.crag_score !== '') {
        const s = data.crag_score.toLowerCase();
        const action = data.crag_action || '';
        const isFallback = action.includes('FALLBACK');

        cragBar.className = `crag-bar ${s}${isFallback ? ' fallback' : ''}`;
        const icons = { correct: '🟢', incorrect: '🔴', ambiguous: '🟡', web_only: '🌐', error: '⚠️' };
        document.getElementById('cragLabel').textContent = `${icons[s] || '❓'} CRAG: ${data.crag_score}`;
        document.getElementById('cragInfo').textContent = data.crag_reason || '';
        document.getElementById('cragLatency').textContent = data.crag_latency_ms ? `${data.crag_latency_ms}ms` : '';

        // Show action detail
        const cragActionEl = document.getElementById('cragAction');
        if (action) {
            const actionMap = {
                'PASS_THROUGH': { icon: '✅', text: '本地知识充足，直接使用本地检索结果' },
                'SCORCHED_EARTH': { icon: '🌐', text: '本地无相关内容，已切换为外网搜索结果' },
                'SCORCHED_EARTH_FALLBACK': { icon: '⚠️', text: '本地无相关内容，外网搜索也失败了，已降级返回本地结果（请检查网络连接或 Tavily API）' },
                'DUAL_AUGMENT': { icon: '🔄', text: '本地信息不完整，已补充外网搜索结果' },
                'DUAL_AUGMENT_FALLBACK': { icon: '⚠️', text: '本地信息不完整，提炼与外网搜索均失败，已降级返回原始结果' },
                'WEB_SEARCH_DISABLED': { icon: '🚫', text: '本地无相关内容，但网络检索已关闭，仅返回本地结果（开启🌐网络检索可获取更多结果）' },
                'REFINE_ONLY': { icon: '✨', text: '网络检索已关闭，仅对本地知识进行了提炼优化' },
                'REFINE_ONLY_FALLBACK': { icon: '⚠️', text: '网络检索已关闭且本地提炼失败，返回原始结果' },
                'WEB_SEARCH_DIRECT': { icon: '🌐', text: '已直接执行网络检索，结果已追加到本地结果之后' },
                'WEB_SEARCH_DIRECT_EMPTY': { icon: '⚠️', text: '网络检索未返回任何结果' },
                'UNKNOWN_FALLBACK': { icon: '❓', text: '未知状态，返回原始结果' },
            };
            const key = Object.keys(actionMap).find(k => action.startsWith(k));
            const mapped = key ? actionMap[key] : null;
            if (mapped) {
                cragActionEl.innerHTML = `<span class="crag-action-icon">${mapped.icon}</span> ${mapped.text}`;
            } else {
                cragActionEl.textContent = action;
            }
            cragActionEl.style.display = 'block';
        } else {
            cragActionEl.style.display = 'none';
        }
    } else {
        cragBar.className = 'crag-bar';
    }

    // Chunks
    const chunks = data.chunks || [];
    if (chunks.length === 0) {
        area.innerHTML = '<div class="empty-state"><div class="icon">🔍</div><div>未找到相关内容</div></div>';
        return;
    }

    area.innerHTML = chunks.map((c, i) => {
        const type = c.doc_type_kwd || c.knowledge_graph_kwd || '';
        let cls = '';
        let badge = '';

        if (type === 'knowledge_graph') {
            cls = 'graph';
            badge = '<span class="source" style="color:var(--accent)">🔗 知识图谱</span>';
        } else if (type === 'web_search' || c.knowledge_graph_kwd === 'web') {
            cls = 'web';
            badge = '<span class="source" style="color:var(--blue)">🌐 外网搜索</span>';
        } else if (type === 'refined' || c.knowledge_graph_kwd === 'refined') {
            cls = 'refined';
            badge = '<span class="source" style="color:var(--yellow)">✨ 提炼知识</span>';
        } else {
            badge = `<span class="source">📄 ${c.docnm_kwd || '本地文档'}</span>`;
        }

        const score = c.similarity ? `相关度: ${(c.similarity * 100).toFixed(1)}%` : '';
        const rerank = c.rerank_score ? ` | 精排: ${(c.rerank_score * 100).toFixed(1)}%` : '';
        const content = (c.content_with_weight || '').substring(0, 800);

        return `
            <div class="result-item ${cls}" style="animation-delay:${i * 0.05}s">
                <div class="result-header">
                    ${badge}
                    <span class="score">${score}${rerank}</span>
                </div>
                <div class="result-content">${escapeHtml(content)}</div>
            </div>
        `;
    }).join('');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ==================== Toast ====================
function toast(msg, type = 'info') {
    const container = document.getElementById('toastContainer');
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = msg;
    container.appendChild(el);
    setTimeout(() => el.remove(), 3000);
}

// ==================== Chunk Browser ====================
let chunkBrowserKB = null;
let chunkBrowserPage = 1;
let chunkDocList = [];       // [{doc_name, chunk_count, checked}]
let allDocsSelected = true;

async function openChunkModal(kbId) {
    chunkBrowserKB = kbId;
    chunkBrowserPage = 1;
    allDocsSelected = true;
    document.getElementById('chunkModal').classList.add('active');
    document.getElementById('chunkModalTitle').textContent = `分块浏览 — ${kbId}`;

    // Load documents first, then chunks
    await loadDocList();
    await loadChunks();
}

function closeChunkModal() {
    document.getElementById('chunkModal').classList.remove('active');
}

async function loadDocList() {
    const list = document.getElementById('docList');
    list.innerHTML = '<div style="padding:20px;text-align:center;color:var(--text-muted);font-size:13px">加载中...</div>';

    try {
        const r = await fetch(`${API}/api/documents/${chunkBrowserKB}`);
        const raw = await r.json();
        const d = unwrap(raw) || {};
        chunkDocList = (d.documents || []).map(doc => ({...doc, checked: true}));

        if (chunkDocList.length === 0) {
            list.innerHTML = '<div style="padding:20px;text-align:center;color:var(--text-muted);font-size:13px">暂无文档</div>';
            return;
        }

        renderDocList();
        updateHeaderMeta();
    } catch (e) {
        list.innerHTML = `<div style="padding:20px;text-align:center;color:var(--red);font-size:13px">加载失败</div>`;
    }
}

function renderDocList() {
    const list = document.getElementById('docList');
    list.innerHTML = chunkDocList.map((doc, i) => {
        const ext = doc.doc_name.split('.').pop().toLowerCase();
        const icons = {pdf:'📕', md:'📝', docx:'📃', xlsx:'📊', txt:'📄', html:'🌐', json:'⚙️', pptx:'📊'};
        const icon = icons[ext] || '📄';
        return `
            <div class="doc-item ${doc.checked ? 'checked' : ''}" data-doc-idx="${i}">
                <input type="checkbox" ${doc.checked ? 'checked' : ''}>
                <div class="doc-info">
                    <span class="doc-name" title="${escapeHtml(doc.doc_name)}">${icon} ${escapeHtml(doc.doc_name)}</span>
                </div>
                <span class="chunk-badge">${doc.chunk_count}</span>
                <button class="doc-delete-btn" data-doc-delete="${i}" title="删除文档">✕</button>
            </div>
        `;
    }).join('');

    // Bind events via delegation
    list.onclick = function(e) {
        // Delete button?
        const delBtn = e.target.closest('.doc-delete-btn');
        if (delBtn) {
            e.stopPropagation();
            const idx = parseInt(delBtn.getAttribute('data-doc-delete'));
            deleteDoc(chunkDocList[idx].doc_name);
            return;
        }
        // Checkbox?
        const checkbox = e.target.closest('input[type="checkbox"]');
        if (checkbox) {
            e.stopPropagation();
            const item = checkbox.closest('.doc-item');
            const idx = parseInt(item.getAttribute('data-doc-idx'));
            toggleDoc(idx);
            return;
        }
        // Whole row?
        const item = e.target.closest('.doc-item');
        if (item) {
            const idx = parseInt(item.getAttribute('data-doc-idx'));
            toggleDoc(idx);
        }
    };

    // Update select-all button text
    const allChecked = chunkDocList.every(d => d.checked);
    document.getElementById('selectAllBtn').textContent = allChecked ? '取消全选' : '全选';
}

function updateHeaderMeta() {
    const selected = chunkDocList.filter(d => d.checked);
    const totalChunks = selected.reduce((sum, d) => sum + d.chunk_count, 0);
    document.getElementById('chunkModalMeta').textContent =
        `${selected.length}/${chunkDocList.length} 文档 · ~${totalChunks} 分块`;
}

function toggleDoc(idx) {
    chunkDocList[idx].checked = !chunkDocList[idx].checked;
    chunkBrowserPage = 1;
    renderDocList();
    updateHeaderMeta();
    loadChunks();
}

function toggleSelectAll() {
    const allChecked = chunkDocList.every(d => d.checked);
    chunkDocList.forEach(d => d.checked = !allChecked);
    chunkBrowserPage = 1;
    renderDocList();
    updateHeaderMeta();
    loadChunks();
}

async function deleteDoc(docName) {
    // Show custom confirm dialog instead of native confirm()
    document.getElementById('confirmMsg').innerHTML =
        `确定要删除文档 <strong>"${escapeHtml(docName)}"</strong> 及其所有分块吗？<br>此操作不可撤销！`;
    document.getElementById('confirmOverlay').classList.add('active');

    // Wire up the confirm button
    const btn = document.getElementById('confirmDeleteBtn');
    // Remove old listener by replacing node
    const newBtn = btn.cloneNode(true);
    btn.parentNode.replaceChild(newBtn, btn);
    newBtn.disabled = false;
    newBtn.textContent = '🗑️ 确认删除';
    newBtn.onclick = async function() {
        newBtn.disabled = true;
        newBtn.textContent = '删除中...';
        try {
            const r = await fetch(`${API}/api/document/${chunkBrowserKB}/${encodeURIComponent(docName)}`, {
                method: 'DELETE',
            });
            const raw = await r.json();
            const d = unwrap(raw) || {};
            closeConfirm();
            if (r.ok) {
                toast(`✅ 已删除 "${docName}"（${d.deleted_chunks} 个分块）`, 'success');
                await loadDocList();
                chunkBrowserPage = 1;
                await loadChunks();
                loadKBs();
            } else {
                toast(`⚠️ 删除失败: ${raw.message || '未知错误'}`, 'error');
            }
        } catch (e) {
            closeConfirm();
            toast(`❌ 删除失败: ${e.message}`, 'error');
        }
    };
}

function closeConfirm() {
    document.getElementById('confirmOverlay').classList.remove('active');
}

// Cache chunk data for click events (avoids inline JSON escaping issues)
let chunkDataCache = [];

async function loadChunks() {
    const body = document.getElementById('chunkPanelBody');
    const pag = document.getElementById('chunkPagination');
    body.innerHTML = '<div class="empty-state"><div class="icon">⏳</div><div>加载中...</div></div>';
    pag.innerHTML = '';

    const selectedDocs = chunkDocList.filter(d => d.checked).map(d => d.doc_name);

    if (selectedDocs.length === 0) {
        body.innerHTML = '<div class="empty-state"><div class="icon">👈</div><div>请在左侧勾选至少一个文档</div></div>';
        return;
    }

    try {
        const docParam = encodeURIComponent(selectedDocs.join(','));
        const r = await fetch(`${API}/api/chunks/${chunkBrowserKB}?page=${chunkBrowserPage}&page_size=20&doc_names=${docParam}`);
        const raw = await r.json();
        const d = unwrap(raw) || {};

        if (!d.chunks || d.chunks.length === 0) {
            body.innerHTML = '<div class="empty-state"><div class="icon">📭</div><div>该文档暂无分块数据</div></div>';
            return;
        }

        // Store chunk data in cache so we don't need inline JSON
        chunkDataCache = d.chunks;

        const gridHTML = d.chunks.map((c, i) => {
            const idx = (chunkBrowserPage - 1) * 20 + i + 1;
            const preview = escapeHtml(c.content_preview || '');
            const docName = escapeHtml(c.docnm_kwd || '未知');
            return `
                <div class="chunk-card" data-chunk-idx="${i}">
                    <div class="chunk-card-head">
                        <span class="doc-name" title="${docName}">📄 ${docName}</span>
                        <span class="meta">#${idx} · ${c.char_count} 字符</span>
                    </div>
                    <div class="chunk-card-preview">${preview}</div>
                    <div class="chunk-card-footer">
                        <span class="type-badge">${c.doc_type_kwd}</span>
                        <button class="view-btn">查看详情 →</button>
                    </div>
                </div>
            `;
        }).join('');

        body.innerHTML = '<div class="chunk-grid">' + gridHTML + '</div>';

        // Bind click via delegation — no inline escaping issues
        body.querySelector('.chunk-grid').onclick = function(e) {
            const card = e.target.closest('.chunk-card');
            if (!card) return;
            const ci = parseInt(card.getAttribute('data-chunk-idx'));
            const c = chunkDataCache[ci];
            if (!c) return;
            const idx = (chunkBrowserPage - 1) * 20 + ci + 1;
            showChunkDetail(c.content_full, c.docnm_kwd || '未知', c.char_count, idx);
        };

        // Pagination
        pag.innerHTML = `
            <button class="btn btn-secondary" ${d.page <= 1 ? 'disabled' : ''}
                onclick="chunkBrowserPage--;loadChunks()">← 上一页</button>
            <span class="page-info">第 ${d.page} / ${d.total_pages} 页 · 共 ${d.total} 条</span>
            <button class="btn btn-secondary" ${d.page >= d.total_pages ? 'disabled' : ''}
                onclick="chunkBrowserPage++;loadChunks()">下一页 →</button>
        `;
    } catch (e) {
        body.innerHTML = `<div class="empty-state"><div class="icon">❌</div><div>加载失败: ${e.message}</div></div>`;
    }
}

function showChunkDetail(content, docName, charCount, idx) {
    document.getElementById('chunkDetailTitle').textContent = `#${idx} — ${docName} (${charCount} 字符)`;
    document.getElementById('chunkDetailBody').textContent = content;
    document.getElementById('chunkDetailOverlay').classList.add('active');
}

function closeChunkDetail() {
    document.getElementById('chunkDetailOverlay').classList.remove('active');
}

document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
        closeChunkDetail();
        closeChunkModal();
    }
});

// ==================== Performance Stats ====================
const STAGE_LABELS = {
    'es_retrieval': '🔍 ES检索',
    'reranker': '🏆 Reranker',
    'graph_search': '🔗 图谱检索',
    'crag_routing': '🧠 CRAG路由',
    'total_retrieval': '∑ 普通检索总耗',
    'total_graph_retrieval': '∑ 图谱检索总耗',
};

const STAGE_COLORS = {
    'es_retrieval': '#3b82f6',
    'reranker': '#f59e0b',
    'graph_search': '#6366f1',
    'crag_routing': '#10b981',
    'total_retrieval': '#94a3b8',
    'total_graph_retrieval': '#94a3b8',
};

async function loadPerfStats() {
    try {
        const r = await fetch(`${API}/api/stats`);
        const raw = await r.json();
        const data = unwrap(raw) || {};
        renderPerfStats(data);
    } catch {}
}

function renderPerfStats(data) {
    const el = document.getElementById('perfStats');
    const stages = Object.keys(data);
    if (stages.length === 0) {
        el.innerHTML = '<div style="text-align:center;color:var(--text-muted);padding:8px">暂无数据</div>';
        return;
    }

    el.innerHTML = stages.map(stage => {
        const s = data[stage];
        const label = STAGE_LABELS[stage] || stage;
        const color = STAGE_COLORS[stage] || 'var(--text-muted)';
        const isTotal = stage.startsWith('total_');
        return `
            <div style="margin-bottom:8px;${isTotal ? 'border-top:1px solid var(--border);padding-top:8px;margin-top:4px' : ''}">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <span style="color:${color};font-weight:500">${label}</span>
                    <span style="color:var(--text-primary);font-weight:600">${s.last_ms}ms</span>
                </div>
                <div style="display:flex;gap:8px;color:var(--text-muted);font-size:11px;margin-top:2px">
                    <span>avg ${s.avg_ms}</span>
                    <span>p50 ${s.p50_ms}</span>
                    <span>p95 ${s.p95_ms}</span>
                    <span>max ${s.max_ms}</span>
                    <span>×${s.count}</span>
                </div>
            </div>
        `;
    }).join('');
}

async function resetPerfStats() {
    try {
        await fetch(`${API}/api/stats/reset`, { method: 'POST' });
        document.getElementById('perfStats').innerHTML =
            '<div style="text-align:center;color:var(--text-muted);padding:8px">已重置</div>';
        toast('性能统计已重置', 'info');
    } catch {}
}

// ==================== Init ====================
loadSavedParams();
renderHistory();
checkHealth();
loadKBs();
setInterval(checkHealth, 30000);
