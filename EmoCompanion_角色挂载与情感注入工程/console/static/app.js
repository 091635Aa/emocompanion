// EmoCompanion · 统一控制台 —— 前端逻辑
const $ = (id) => document.getElementById(id);
let SID = null;
let EMOTIONS = [];
let MOUNT = {};
let ROLES = {};
let CATEGORIES = [];
const player = $("player");

// ==================== 工具 ====================
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
}
function fmtTime(ts) {
  const d = new Date(ts * 1000);
  return `${String(d.getMonth()+1).padStart(2,"0")}-${String(d.getDate()).padStart(2,"0")} ${String(d.getHours()).padStart(2,"0")}:${String(d.getMinutes()).padStart(2,"0")}`;
}
function setPill(id, txt, state) {
  const el = $(id);
  el.textContent = txt;
  el.classList.remove("ok","bad","warn");
  if (state) el.classList.add(state);
}
async function post(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  if (!r.ok) {
    let msg = await r.text().catch(() => "");
    throw new Error(msg.slice(0, 300) || r.statusText);
  }
  return r.json();
}
function debounce(fn, ms=300) {
  let t; return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

// 点击涟漪效果
function bindRipple(root=document) {
  root.querySelectorAll("button, .qa-chip, .session, .db-btn, .nav-tab").forEach((el) => {
    if (el._ripple) return; el._ripple = true;
    el.addEventListener("click", (e) => {
      const r = document.createElement("span"); r.className = "ripple";
      const rect = el.getBoundingClientRect();
      const size = Math.max(rect.width, rect.height);
      r.style.width = r.style.height = size + "px";
      r.style.left = (e.clientX - rect.left - size / 2) + "px";
      r.style.top = (e.clientY - rect.top - size / 2) + "px";
      el.appendChild(r); setTimeout(() => r.remove(), 600);
    });
  });
}

// Toast 提示
function toast(msg, type="info", dur=3000) {
  const box = $("toastBox");
  const t = document.createElement("div"); t.className = "toast " + type; t.textContent = msg;
  box.appendChild(t);
  setTimeout(() => { t.classList.add("toast-out"); t.addEventListener("animationend", () => t.remove()); }, dur);
}

// 全屏加载
function showLoading(txt="处理中…") { $("loadingText").textContent = txt; $("loadingMask").classList.remove("hidden"); }
function hideLoading() { $("loadingMask").classList.add("hidden"); }

// 安全 fetch, 带自动重试
async function safeFetch(url, opts={}, retries=1) {
  let lastErr;
  for (let i = 0; i <= retries; i++) {
    try {
      const r = await fetch(url, opts);
      return r;
    } catch (e) { lastErr = e; if (i < retries) await new Promise((res) => setTimeout(res, 800)); }
  }
  throw lastErr;
}

// ==================== 初始化 ====================
async function init() {
  refreshHealth();
  loadMounts();
  loadCategories();
  loadSessions();
  loadQuick();
  bindEvents();
  bindStudio();
  bindLogs();
  bindDebug();
  bindRipple();
  bindShortcuts();
  setInterval(refreshHealth, 15000);
  setInterval(pollLoadStatus, 2000);
  setInterval(pollPanels, 3000);
  pollLoadStatus();
  newChat();
  toast("EmoCompanion控制台已就绪", "ok", 2500);
}

// 面板实时轮询
function pollPanels() {
  if (DBG_ON) refreshDebug();
  if (LOG_ON) refreshLogs();
  const logsTab = $("tab-logs"); if (!logsTab.classList.contains("hidden")) refreshLogsMain();
}

// 快捷键绑定
function bindShortcuts() {
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      $("roleDrawer").classList.add("hidden");
      $("kbdHelp").classList.add("hidden");
      if (DBG_ON) toggleDebug();
      if (LOG_ON) toggleLogPanel();
    }
    if (e.ctrlKey || e.metaKey) {
      if (e.key === "d" || e.key === "D") { e.preventDefault(); toggleDebug(); }
      if (e.key === "l" || e.key === "L") { e.preventDefault(); toggleLogPanel(); }
      if (e.key === "/") { e.preventDefault(); $("kbdHelp").classList.remove("hidden"); }
    }
  });
  $("closeKbd").addEventListener("click", () => $("kbdHelp").classList.add("hidden"));
}

async function loadCategories() {
  try {
    const j = await (await fetch("/api/logs/categories")).json();
    CATEGORIES = j.categories || [];
    ["logCat","logCatMain"].forEach((id) => {
      const sel = $(id); sel.innerHTML = '<option value="">全部分类</option>';
      CATEGORIES.forEach((c) => { const o = document.createElement("option"); o.value = c; o.textContent = c; sel.appendChild(o); });
    });
  } catch {}
}

async function loadQuick() {
  try {
    const j = await (await fetch("/api/quick")).json();
    const bar = $("qaBar");
    bar.querySelectorAll(".qa-chip").forEach((c) => c.remove());
    (j.chips || []).forEach((c) => {
      const b = document.createElement("button");
      b.className = "qa-chip"; b.dataset.q = c.q; b.textContent = c.label;
      b.addEventListener("click", () => { $("input").value = c.q; send(); });
      bar.appendChild(b);
    });
  } catch {}
}

async function refreshHealth() {
  try {
    const j = await safeFetch("/api/health", {}, 1);
    const data = await j.json();
    setPill("textState", "文本引擎: " + (data.text_engine === "online" ? "在线" : "离线"),
      data.text_engine === "online" ? "ok" : "bad");
    const ttsReady = data.tts && data.tts.tts === "ready";
    setPill("ttsState", "TTS: " + (ttsReady ? "就绪" : (data.tts && data.tts.tts) || "?"),
      ttsReady ? "ok" : "bad");
    $("ttsState").classList.toggle("pulse", ttsReady);
  } catch {
    setPill("textState", "文本引擎: 无响应", "bad");
    setPill("ttsState", "TTS: 无响应", "bad");
    $("ttsState").classList.remove("pulse");
  }
  if (DBG_ON) refreshDebug();
}

async function loadMounts() {
  try {
    const j = await (await fetch("/api/mounts")).json();
    MOUNT = j.mount || {};
    EMOTIONS = j.emotions || [];
    $("adapterSel").value = MOUNT.adapter || "emotion";
    $("emotionModeSel").value = MOUNT.emotion_mode || "auto";
    $("wantTts").checked = MOUNT.want_tts !== false;
    $("toneVar").value = MOUNT.tone_variation ?? 0.35;
    fillManualEmo();
    fillStudioEmo();
    fillRoleSel(j.roles || ["EmoCompanion"]);
  } catch {}
}

function fillManualEmo() {
  const sel = $("manualEmo"); sel.innerHTML = "";
  (EMOTIONS || []).forEach((e) => { const o = document.createElement("option"); o.value = e; o.textContent = e; sel.appendChild(o); });
  if (MOUNT.emotion) sel.value = MOUNT.emotion;
}

function fillRoleSel(names) {
  const list = names || Object.keys(ROLES || {});
  const sel = $("roleSel"); const cur = sel.value || MOUNT.role || list[0] || "EmoCompanion";
  sel.innerHTML = "";
  list.forEach((n) => { const o = document.createElement("option"); o.value = n; o.textContent = n; sel.appendChild(o); });
  if (list.includes(cur)) sel.value = cur;
  $("roleTag").textContent = sel.value || "EmoCompanion";
}

// ==================== 会话管理 ====================
async function loadSessions() {
  try {
    const j = await (await fetch("/api/sessions")).json();
    renderSessions(j.sessions || []);
  } catch {}
}

function renderSessions(list) {
  const box = $("sessionList"); box.innerHTML = "";
  list.forEach((s) => {
    const d = document.createElement("div");
    d.className = "session" + (s.id === SID ? " active" : "");
    d.innerHTML = `
      <div class="sess-txt"><div class="t">${escapeHtml(s.title)}</div><div class="m">${s.n_msgs} 条 · ${fmtTime(s.updated)}</div></div>
      <button class="sess-del" title="删除">✕</button>`;
    d.addEventListener("click", (e) => { if (e.target.classList.contains("sess-del")) return; openSession(s.id); });
    d.querySelector(".sess-del").addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm("确定删除该对话?")) return;
      await fetch(`/api/sessions/${s.id}`, { method: "DELETE" });
      if (SID === s.id) SID = null;
      loadSessions();
      toast("会话已删除", "ok", 1500);
    });
    box.appendChild(d); bindRipple(d);
  });
}

async function newChat() {
  const j = await post("/api/sessions", { title: "新对话" });
  SID = j.id;
  $("chat").innerHTML = '';
  $("chatTitle").textContent = "新对话";
  loadSessions();
  $("input").focus();
}

async function openSession(sid) {
  const j = await (await fetch(`/api/sessions/${sid}`)).json();
  SID = sid;
  $("chatTitle").textContent = j.title || "对话";
  const chat = $("chat"); chat.innerHTML = '';
  (j.messages || []).forEach((m) => renderMsg(m, false));
  chat.scrollTop = chat.scrollHeight;
  loadSessions();
}

// ==================== 消息渲染 ====================
function renderMsg(m, animate) {
  const chat = $("chat");
  const d = document.createElement("div");
  d.className = "msg " + (m.role === "user" ? "user" : "assistant");
  let meta = "";
  if (m.role === "assistant") {
    const emo = m.emotion || "";
    const src = m.emotion_info ? m.emotion_info.source || "" : "";
    meta = `<div class="meta"><span class="emo-tag">${escapeHtml(emo)}</span>` + (src ? `<span>来源: ${escapeHtml(src)}</span>` : "") + `</div>`;
  }
  let audio = "";
  if (m.role === "assistant" && m.audio) {
    audio = `<button class="play-btn" data-audio="${m.audio}">▶ 播放语音</button>`;
  }
  const copyBtn = `<button class="play-btn copy-btn" data-copy="${escapeHtml(m.content).replace(/"/g,'&quot;')}">📋 复制</button>`;
  d.innerHTML = `<div class="avatar">${m.role === "user" ? "我" : "缘"}</div><div class="col"><div class="bubble">${escapeHtml(m.content)}</div>${meta}${audio}${copyBtn}</div>`;
  if (animate) d.style.animation = "none";
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
  if (audio) {
    d.querySelector(".play-btn[data-audio]").addEventListener("click", (e) => { player.src = e.target.dataset.audio; player.play(); });
  }
  d.querySelector(".copy-btn").addEventListener("click", async (e) => {
    try {
      await navigator.clipboard.writeText(m.content);
      toast("已复制到剪贴板", "ok", 1500);
    } catch { toast("复制失败", "err", 1500); }
  });
  bindRipple(d);
  return d;
}

// ==================== 发送(流式) ====================
async function send() {
  const input = $("input");
  const content = input.value.trim();
  if (!content) return;
  if (!SID) { const j = await post("/api/sessions", { title: "新对话" }); SID = j.id; }
  const btn = $("sendBtn");
  btn.disabled = true; btn.textContent = "发送中";
  renderMsg({ role: "user", content }, false);
  input.value = "";
  const st = $("genStats");
  const wantTts = $("wantTts").checked;
  st.textContent = "对方正在输入…"; st.classList.remove("err","ok");

  const chat = $("chat");
  const col = document.createElement("div"); col.className = "col";
  const bub = document.createElement("div"); bub.className = "bubble typing-hd";
  bub.innerHTML = "对方正在输入<i></i><i></i><i></i>";
  const bubbleEl = document.createElement("div");
  bubbleEl.className = "msg assistant"; bubbleEl.innerHTML = `<div class="avatar">缘</div>`;
  bubbleEl.appendChild(col); col.appendChild(bub);
  chat.appendChild(bubbleEl); chat.scrollTop = chat.scrollHeight;

  let replyText = "";
  try {
    const body = {
      content, want_tts: wantTts, backend: "gguf",
      emotion_mode: $("emotionModeSel").value,
      emotion: $("emotionModeSel").value === "manual" ? $("manualEmo").value : undefined,
      role: $("roleSel").value || "EmoCompanion",
      adapter: $("adapterSel").value,
      tone_variation: Number($("toneVar").value) || 0.35,
    };
    const resp = await safeFetch(`/api/sessions/${SID}/stream`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    }, 0);
    if (!resp.ok || !resp.body) throw new Error("流式请求失败: HTTP " + resp.status);
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "", final = null;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, idx); buf = buf.slice(idx + 2);
        for (const line of chunk.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          let evt;
          try { evt = JSON.parse(line.slice(6)); } catch { continue; }
          if (evt.delta) {
            bub.classList.remove("typing-hd");
            replyText += evt.delta;
            const span = document.createElement("span"); span.className = "token-in"; span.textContent = evt.delta;
            bub.appendChild(span); chat.scrollTop = chat.scrollHeight;
          } else if (evt.text_done) {
            replyText = evt.reply || replyText;
            bub.textContent = replyText; bub.classList.remove("typing-hd");
            chat.scrollTop = chat.scrollHeight;
            if (wantTts) { st.textContent = "对方正在发送语音"; st.classList.add("typing-voice"); }
          } else if (evt.sentence_audio) {
            const pb = document.createElement("button"); pb.className = "play-btn";
            const stTag = evt.style ? "·" + evt.style : "";
            const rtTag = evt.rate && Math.abs(evt.rate - 1) > 0.01 ? `(x${Number(evt.rate).toFixed(2)})` : "";
            pb.textContent = `▶ 整段语音${stTag}${rtTag}`;
            pb.addEventListener("click", () => { player.src = evt.audio; player.play(); });
            col.appendChild(pb);
            if (wantTts) { player.src = evt.audio; player.play(); st.textContent = "就绪"; st.classList.remove("typing-voice"); }
            chat.scrollTop = chat.scrollHeight;
          } else if (evt.done) { final = evt; }
          else if (evt.error) { throw new Error(evt.error); }
        }
      }
    }
    if (final) {
      const emo = final.emotion || "";
      const metaDiv = document.createElement("div"); metaDiv.className = "meta";
      metaDiv.innerHTML = `<span class="emo-tag">${escapeHtml(emo)}</span>` +
        (final.emotion_info ? `<span>来源: ${escapeHtml(final.emotion_info.source)}</span>` : "") +
        (final.audio ? `<span>● 整段语音可播放</span>` : "");
      col.appendChild(metaDiv);
      if (final.audio) {
        const pb = document.createElement("button"); pb.className = "play-btn"; pb.textContent = "▶ 播放整段";
        pb.addEventListener("click", () => { player.src = final.audio; player.play(); });
        col.appendChild(pb);
      }
      const ts = final.text_stats || {};
      const t = final.tts_meta || {};
      let ttsTxt = t.error ? `失败：${String(t.error).slice(0,120)}` : (final.audio ? `已合成${t.n_sentences ? '(' + t.n_sentences + '句)' : ''}` : "跳过");
      st.textContent = `情感=${final.emotion}(${final.emotion_info && final.emotion_info.source}) ｜ 文本${ts.latency_s ? ts.latency_s + 's' : ''} ｜ TTS: ${ttsTxt}`;
      st.classList.remove("typing-voice"); if (t.error) st.classList.add("err"); else st.classList.add("ok");
      toast("EmoCompanion已回复", "ok", 2000);
    }
    chat.scrollTop = chat.scrollHeight;
    loadSessions();
  } catch (err) {
    bub.textContent = "[发送失败]"; bub.classList.add("err");
    st.textContent = "❌ " + err.message; st.classList.remove("typing-voice"); st.classList.add("err");
    toast("发送失败: " + err.message, "err", 4000);
  } finally {
    btn.disabled = false; btn.textContent = "发送";
    if (DBG_ON) { refreshDebug(); refreshLogs(); }
  }
}

// ==================== 角色设定 ====================
async function openRoleDrawer() {
  const j = await (await fetch("/api/roles")).json();
  ROLES = j.roles || {};
  const r = ROLES[j.active] || ROLES["EmoCompanion"] || {};
  $("roleName").value = r.name || "EmoCompanion";
  $("roleDesc").value = r.desc || "";
  $("rolePersona").value = r.persona || "";
  $("roleCatch").value = (r.catchphrases || []).join("，");
  renderTraits(r.traits || {});
  renderEmoKws(r.emotion_keywords || {});
  $("roleDrawer").classList.remove("hidden");
}
function renderTraits(traits) {
  const box = $("traitList"); box.innerHTML = "";
  const names = { warmth: "温暖", playfulness: "俏皮", sassiness: "傲娇", energy_baseline: "能量基线", formality: "正式度" };
  Object.entries(traits || {}).forEach(([k, v]) => {
    const d = document.createElement("div"); d.className = "trait";
    d.innerHTML = `<div class="nm"><span>${names[k] || k}</span><b id="tv_${k}">${Number(v).toFixed(2)}</b></div><input type="range" min="0" max="1" step="0.01" value="${Number(v).toFixed(2)}" data-trait="${k}"/>`;
    d.querySelector("input").addEventListener("input", (e) => { $("tv_" + k).textContent = Number(e.target.value).toFixed(2); });
    box.appendChild(d);
  });
}
function renderEmoKws(kws) {
  const box = $("emoKwList"); box.innerHTML = "";
  Object.entries(kws || {}).forEach(([k, v]) => {
    const d = document.createElement("div"); d.className = "emo-kw";
    d.innerHTML = `<span class="e">${escapeHtml(k)}</span><input data-emokw="${k}" value="${escapeHtml(String(v))}"/>`;
    box.appendChild(d);
  });
}
async function saveRole() {
  const st = $("roleSaveStat"); st.textContent = "保存中…"; st.classList.remove("err");
  try {
    const traits = {}; document.querySelectorAll("[data-trait]").forEach((i) => { traits[i.dataset.trait] = Number(i.value); });
    const kws = {}; document.querySelectorAll("[data-emokw]").forEach((i) => { kws[i.dataset.emokw] = i.value; });
    const body = {
      name: $("roleName").value, desc: $("roleDesc").value, persona: $("rolePersona").value,
      traits, catchphrases: $("roleCatch").value.split(/[,，]/).map((s) => s.trim()).filter(Boolean), emotion_keywords: kws,
    };
    await post("/api/mounts", {});
    const r = await safeFetch("/api/roles", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }, 0);
    if (!r.ok) throw new Error(await r.text());
    st.textContent = "✅ 已保存角色设定";
    ROLES = (await (await fetch("/api/roles")).json()).roles;
    fillRoleSel();
    toast("角色设定已保存", "ok", 2000);
  } catch (err) { st.textContent = "❌ " + err.message; st.classList.add("err"); toast("保存失败: " + err.message, "err", 4000); }
}

// ==================== 调试模式 ====================
let DBG_ON = false;
function toggleDebug() {
  DBG_ON = !DBG_ON;
  $("debugBtn").classList.toggle("on", DBG_ON);
  $("debugPanel").classList.toggle("hidden", !DBG_ON);
  if (DBG_ON) { refreshDebug(); refreshDbLogTail(); }
}
function bindDebug() {
  $("debugBtn").addEventListener("click", toggleDebug);
  $("dbRefresh").addEventListener("click", () => { refreshDebug(); refreshDbLogTail(); });
  $("dbCopyPaths").addEventListener("click", async () => {
    try {
      const j = await (await fetch("/api/debug")).json();
      await navigator.clipboard.writeText(JSON.stringify(j.paths || {}, null, 2));
      toast("路径已复制", "ok", 1500);
    } catch { toast("复制失败", "err", 1500); }
  });
}
async function refreshDebug() {
  if (!DBG_ON || !$("dbAuto").checked) return;
  try {
    const j = await safeFetch("/api/debug", {}, 0);
    const data = await j.json();
    const g = data.gpu, m = data.mount || {}, te = data.text_debug || {}, r = data.role || {};
    $("dbRole").textContent = r.active || "-";
    $("dbEmotions").textContent = (data.emotions || []).join(" ");
    $("dbStyles").textContent = (data.styles || []).join(" ");
    $("dbCatch").textContent = (r.catchphrases || []).join("，") || "-";
    $("dbCurEmo").textContent = (m.emotion || "-") + (m.emotion_mode ? "(" + m.emotion_mode + ")" : "");
    const anchors = (te.meta && te.meta.anchors) || [];
    $("dbAnchors").textContent = anchors.join(" ") || "-";
    const ds = te.deai_summary || {};
    $("dbPos").textContent = (ds.pos_tok ?? "-") + "/" + (ds.pos_phr ?? "-");
    $("dbHol").textContent = (ds.hol_tok ?? "-") + "/" + (ds.hol_phr ?? "-");
    $("dbOoc").textContent = ds.ooc_phr ?? "-";
    const mt = te.meta || {};
    $("dbMeta").textContent = (mt.beta != null ? "β" + mt.beta : "-") + (mt.T != null ? " · T" + mt.T : "") + (mt.n_vocab ? " · v" + mt.n_vocab : "");
    $("dbModel").textContent = te.model || "-";
    const st = te.stats || {};
    $("dbCalls").textContent = (st.calls ?? "-") + " / " + (st.tokens ?? "-");
    $("dbSecs").textContent = st.seconds ?? "-";
    $("dbAvg").textContent = (st.avg_tok_s ?? 0) + " tok/s";
    $("dbBackend").textContent = (data.tts && data.tts.backend) || "-";
    const ld = data.load || {};
    $("dbLoad").textContent = (ld.state && ld.message) ? ld.state + " · " + ld.message : "未加载";
    $("dbLoad").className = "v" + (ld.state === "ready" ? " ok" : (ld.state === "error" ? " warn" : ""));
    $("dbGpuName").textContent = g ? (g.name || "-") : "未检测到";
    $("dbVram").textContent = g ? `${(g.used_mb / 1024).toFixed(1)} / ${(g.total_mb / 1024).toFixed(1)} GB` : "-";
    $("dbVramBar").style.width = g ? Math.min(100, g.used_mb / g.total_mb * 100) + "%" : "0%";
    $("dbUtil").textContent = g ? g.util_gpu + "%" : "-";
    $("dbRam").textContent = data.process_ram_mb ?? "-";
    $("dbUptime").textContent = data.uptime_s ?? "-";
    const ss = data.session_stats || {};
    $("dbSessTotal").textContent = ss.total_sessions ?? "-";
    $("dbMsgTotal").textContent = ss.total_messages ?? "-";
    $("dbTextEngine").textContent = data.text_engine === "online" ? "在线" : "离线";
    const ls = data.log_stats || {};
    $("dbLogLevels").textContent = JSON.stringify(ls.levels || {});
    $("dbLogCats").textContent = JSON.stringify(ls.categories || {});
    $("dbPaths").textContent = JSON.stringify(data.paths || {});
    $("dbRefreshTime").textContent = "更新于 " + new Date().toLocaleTimeString();
    renderTimeline(data);
  } catch (e) {
    console.error("debug refresh fail", e);
  }
}
async function refreshDbLogTail() {
  if (!DBG_ON) return;
  try {
    const j = await safeFetch("/api/logs?limit=30", {}, 0);
    const data = await j.json();
    renderLogList(data.logs || [], "dbLogTail", null, data.total);
  } catch {}
}
function renderTimeline(data) {
  const box = $("dbTimeline"); box.innerHTML = "";
  const items = [];
  if (data.text_engine) items.push({ t: new Date().toLocaleTimeString(), msg: `文本引擎 ${data.text_engine === "online" ? "在线" : "离线"}`, ok: data.text_engine === "online" });
  const ld = data.load || {};
  items.push({ t: "-", msg: `TTS 模型 ${ld.state || "未知"}: ${ld.message || ""}`, ok: ld.state === "ready" });
  if (data.process_ram_mb) items.push({ t: "-", msg: `进程内存 ${data.process_ram_mb} MB`, ok: true });
  items.forEach((it) => {
    const d = document.createElement("div"); d.className = "db-tl-item";
    d.innerHTML = `<span class="db-tl-dot" style="background:${it.ok ? 'var(--good)' : 'var(--warn)'}"></span><span class="db-tl-time">${it.t}</span><span class="db-tl-msg">${escapeHtml(it.msg)}</span>`;
    box.appendChild(d);
  });
}

// ==================== 日志面板 ====================
let LOG_ON = false;
function toggleLogPanel() {
  LOG_ON = !LOG_ON;
  $("logBtn").classList.toggle("on", LOG_ON);
  $("logPanel").classList.toggle("hidden", !LOG_ON);
  if (LOG_ON) refreshLogs();
}
function bindLogs() {
  $("logBtn").addEventListener("click", toggleLogPanel);
  $("logRefresh").addEventListener("click", refreshLogs);
  $("logRefreshMain").addEventListener("click", refreshLogsMain);
  ["logCat","logLevel","logSearch"].forEach((id) => $(id).addEventListener("change", refreshLogs));
  $("logSearch").addEventListener("input", debounce(refreshLogs, 300));
  ["logCatMain","logLevelMain","logSearchMain"].forEach((id) => $(id).addEventListener("change", refreshLogsMain));
  $("logSearchMain").addEventListener("input", debounce(refreshLogsMain, 300));
}
async function refreshLogs() {
  await fetchLogs("logCat", "logLevel", "logSearch", "logList", "logCount");
}
async function refreshLogsMain() {
  await fetchLogs("logCatMain", "logLevelMain", "logSearchMain", "logListMain", "logCountMain");
}
async function fetchLogs(catId, levelId, searchId, listId, countId) {
  const params = new URLSearchParams();
  const cat = $(catId).value, level = $(levelId).value, search = $(searchId).value.trim();
  if (cat) params.set("category", cat);
  if (level) params.set("level", level);
  if (search) params.set("search", search);
  params.set("limit", "200");
  try {
    const j = await (await fetch("/api/logs?" + params.toString())).json();
    renderLogList(j.logs || [], listId, countId, j.total);
  } catch {}
}
function renderLogList(logs, listId, countId, total) {
  const box = $(listId); box.innerHTML = "";
  if (countId) $(countId).textContent = `共 ${total} 条`;
  if (!logs.length) { box.innerHTML = '<div class="log-empty">暂无日志</div>'; return; }
  logs.forEach((l) => {
    const d = document.createElement("div"); d.className = "log-row";
    d.innerHTML = `<span class="log-time">${escapeHtml(l.t || '')}</span><span class="log-level ${escapeHtml(l.level)}">${escapeHtml(l.level)}</span><span class="log-cat">${escapeHtml(l.category)}</span><span class="log-msg">${escapeHtml(l.msg)}</span>`;
    box.appendChild(d);
  });
  box.scrollTop = box.scrollHeight;
}

// ==================== 加载模型 ====================
async function loadModel() {
  const btn = $("loadBtn"); const bar = $("loadBar");
  btn.disabled = true; bar.classList.remove("hidden", "err"); bar.textContent = "正在加载 gguf 外挂情感模型…（请稍候）";
  showLoading("正在加载 TTS 模型…");
  try {
    const r = await post("/api/tts/load", { backend: "gguf" });
    bar.textContent = (r.state === "ready" ? "✅ " : "⚠️ ") + (r.message || "") + (r.elapsed ? "（耗时 " + r.elapsed + "s）" : "");
    if (r.state !== "ready") bar.classList.add("err");
    toast(r.state === "ready" ? "TTS 模型加载完成" : "TTS 模型加载异常", r.state === "ready" ? "ok" : "warn", 3000);
  } catch (e) { bar.classList.remove("hidden"); bar.classList.add("err"); bar.textContent = "❌ 加载失败: " + e.message; toast("加载失败: " + e.message, "err", 4000); }
  finally { btn.disabled = false; hideLoading(); if (DBG_ON) refreshDebug(); }
}
async function pollLoadStatus() {
  try {
    const j = await (await fetch("/api/tts/load")).json();
    if (j && j.state === "loading") {
      $("loadBar").textContent = "加载中… " + (j.message || ""); $("loadBar").classList.remove("hidden", "err");
    } else if ($("loadBar").classList.contains("hidden") === false && $("loadBar").textContent.startsWith("加载中")) {
      $("loadBar").textContent = (j.state === "ready" ? "✅ " : "⚠️ ") + (j.message || "") + (j.elapsed ? "（耗时 " + j.elapsed + "s）" : "");
      if (j.state !== "ready") $("loadBar").classList.add("err");
    }
  } catch {}
}

// ==================== 语音合成工作室 ====================
function fillStudioEmo() {
  const sel = $("ssEmotion"); sel.innerHTML = "";
  (EMOTIONS || []).forEach((e) => { const o = document.createElement("option"); o.value = e; o.textContent = e; sel.appendChild(o); });
  sel.value = MOUNT.emotion && sel.querySelector(`option[value="${MOUNT.emotion}"]`) ? MOUNT.emotion : "开心";
}
async function studioSynth(cfg) {
  const st = $("ssStatus"), btn = $("ssBtn");
  btn.disabled = true; st.textContent = "合成中…"; st.classList.remove("err", "ok");
  showLoading("正在合成语音…");
  try {
    const r = await safeFetch("/api/tts/synthesize", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: cfg.text, emotion: cfg.emotion, backend: cfg.backend ?? "gguf", adapter: cfg.adapter ?? "emotion", rate: Number($("ssRate").value) || 1.0 }),
    }, 0);
    if (!r.ok) throw new Error((await r.text()).slice(0, 400) || "HTTP " + r.status);
    const meta = JSON.parse(r.headers.get("X-TTS-Meta") || "{}");
    const blob = await r.blob(); const url = URL.createObjectURL(blob);
    const au = $("ssAudio"); au.src = url; au.load();
    $("ssResult").classList.remove("hidden");
    $("ssMeta").innerHTML =
      `<div>情感 <b>${meta.emotion ?? cfg.emotion}</b> ｜ 后端 <b>${meta.backend ?? "gguf"}</b></div>` +
      `<div>采样率 <span>${meta.sr ?? 24000}Hz</span> ｜ 音频 ≈ <span>${(meta.audio_seconds ?? 0).toFixed(2)}s</span> ｜ 语速 <span>${meta.rate}</span></div>`;
    au.play(); st.textContent = "✅ 已合成并播放"; st.classList.add("ok");
    toast("语音合成完成", "ok", 2000);
  } catch (err) { st.textContent = "❌ " + err.message; st.classList.add("err"); toast("合成失败: " + err.message, "err", 4000); }
  finally { btn.disabled = false; hideLoading(); }
}
function bindStudio() {
  document.querySelectorAll(".nav-tab").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll(".nav-tab").forEach((x) => x.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach((x) => x.classList.add("hidden"));
      b.classList.add("active"); $("tab-" + b.dataset.tab).classList.remove("hidden");
      if (b.dataset.tab === "logs") refreshLogsMain();
    });
  });
  $("ssTemp").addEventListener("input", (e) => { $("ssTempOut").textContent = Number(e.target.value).toFixed(2); });
  $("ssTopK").addEventListener("input", (e) => { $("ssTopKOut").textContent = e.target.value; });
  $("ssRate").addEventListener("input", (e) => { $("ssRateOut").textContent = "x" + Number(e.target.value).toFixed(2); });
  $("ssBtn").addEventListener("click", () => {
    const text = $("ssText").value.trim();
    if (!text) { $("ssStatus").textContent = "❌ 请输入文本"; $("ssStatus").classList.add("err"); return; }
    studioSynth({ text, emotion: $("ssEmotion").value });
  });
  $("ssDetectBtn").addEventListener("click", async () => {
    const text = $("ssDetectText").value.trim(); const st = $("ssStatus");
    if (!text) { st.textContent = "❌ 请输入文本"; st.classList.add("err"); return; }
    st.textContent = "识别情感中…"; st.classList.remove("err", "ok");
    try {
      const j = await post("/api/tts/detect", { text, user_msg: text });
      const emo = j.label || "平静"; $("ssEmotion").value = emo;
      st.textContent = `自动识别：${emo}（${j.source}，置信 ${(j.confidence ?? 0).toFixed(2)}）`;
      await studioSynth({ text, emotion: emo });
    } catch (err) { st.textContent = "❌ " + err.message; st.classList.add("err"); }
  });
}

// ==================== 事件绑定 ====================
function bindEvents() {
  $("newChat").addEventListener("click", newChat);
  $("sendBtn").addEventListener("click", send);
  $("input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
  $("roleBtn").addEventListener("click", openRoleDrawer);
  $("roleClose").addEventListener("click", () => $("roleDrawer").classList.add("hidden"));
  $("roleDrawer").addEventListener("click", (e) => { if (e.target.id === "roleDrawer") $("roleDrawer").classList.add("hidden"); });
  $("roleSave").addEventListener("click", saveRole);
  $("loadBtn").addEventListener("click", loadModel);

  $("adapterSel").addEventListener("change", (e) => post("/api/mounts", { adapter: e.target.value }).catch(() => {}));
  $("emotionModeSel").addEventListener("change", (e) => {
    $("manualEmoWrap").classList.toggle("hidden", e.target.value !== "manual");
    post("/api/mounts", { emotion_mode: e.target.value }).catch(() => {});
  });
  $("manualEmo").addEventListener("change", (e) => post("/api/mounts", { emotion: e.target.value }).catch(() => {}));
  $("wantTts").addEventListener("change", (e) => post("/api/mounts", { want_tts: e.target.checked }).catch(() => {}));
  $("toneVar").addEventListener("change", (e) => post("/api/mounts", { tone_variation: Number(e.target.value) }).catch(() => {}));
  $("roleSel").addEventListener("change", (e) => { $("roleTag").textContent = e.target.value; post("/api/mounts", { role: e.target.value }).catch(() => {}); });
}

init();
