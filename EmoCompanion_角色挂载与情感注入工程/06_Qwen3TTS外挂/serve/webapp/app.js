// EmoCompanion · 一体化对话台 —— 前端逻辑
const $ = (id) => document.getElementById(id);
let SID = null;              // 当前会话
let EMOTIONS = [];
let MOUNT = {};
let ROLES = {};
const player = $("player");

// ==================== 聊天滚动（平滑回底 / 上翻暂停 / 新消息提示） ====================
let AT_BOTTOM = true;    // 聊天区是否已停留在底部
let PENDING_NEW = 0;     // 用户上翻期间的未读新消息计数
function nearBottom() {
  const el = $("chat");
  return el.scrollTop + el.clientHeight >= el.scrollHeight - 28;
}
function scrollToBottom(smooth) {
  const el = $("chat");
  el.scrollTo({ top: el.scrollHeight, behavior: smooth ? "smooth" : "auto" });
  AT_BOTTOM = true; PENDING_NEW = 0; updateScrollBtn();
}
function markNew() { PENDING_NEW++; updateScrollBtn(); }
function updateScrollBtn() {
  const btn = $("scrollBottomBtn");
  btn.classList.toggle("hidden", AT_BOTTOM && PENDING_NEW === 0);
  if (AT_BOTTOM && PENDING_NEW === 0) return;
  const count = $("sbnCount"), lab = $("sbnLabel");
  if (PENDING_NEW > 0) {
    count.textContent = PENDING_NEW; count.classList.remove("hidden"); lab.textContent = "新消息";
  } else {
    count.classList.add("hidden"); lab.textContent = "回到底部";
  }
}
// 追加内容后：若在底部则跟随滚动，否则仅累积未读并提示，不打扰上翻阅读
function maybeStick(smooth) {
  if (!AT_BOTTOM) { markNew(); return; }
  scrollToBottom(!!smooth);
}
function bindChatScroll() {
  $("chat").addEventListener("scroll", () => {
    AT_BOTTOM = nearBottom();
    updateScrollBtn();
  });
  $("scrollBottomBtn").addEventListener("click", () => scrollToBottom(true));
}

// ==================== 工具 ====================
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}
function fmtTime(ts) {
  const d = new Date(ts * 1000);
  return `${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}
function setPill(id, txt, state) {
  const el = $(id);
  el.textContent = txt;
  el.classList.toggle("ok", state === "ok");
  el.classList.toggle("bad", state === "bad");
}
async function post(url, body) {
  const r = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!r.ok) {
    let msg = await r.text().catch(() => "");
    throw new Error(msg.slice(0, 300) || r.statusText);
  }
  return r.json();
}

// ==================== 初始化 ====================
async function init() {
  refreshHealth();
  loadMounts();
  loadSessions();
  loadQuick();
  bindEvents();
  bindStudio();
  setInterval(refreshHealth, 15000);
  newChat();
}

// 快速问答：从后端动态下发，避免在前端硬编码
async function loadQuick() {
  try {
    const j = await (await fetch("/api/quick")).json();
    const bar = $("qaBar");
    bar.querySelectorAll(".qa-chip").forEach((c) => c.remove());
    (j.chips || []).forEach((c) => {
      const b = document.createElement("button");
      b.className = "qa-chip";
      b.dataset.q = c.q;
      b.textContent = c.label;
      b.addEventListener("click", () => { $("input").value = c.q; send(); });
      bar.appendChild(b);
    });
  } catch { /* 保留默认 */ }
}

async function refreshHealth() {
  try {
    const j = await (await fetch("/api/health")).json();
    setPill("textState", "文本引擎: " + (j.text_engine === "online" ? "在线" : "离线"),
      j.text_engine === "online" ? "ok" : "bad");
    setPill("ttsState", "TTS: " + (j.tts && j.tts.tts === "ready" ? "就绪" : (j.tts && j.tts.tts) || "?") +
      " · " + (j.mount && j.mount.tts_backend === "gguf" ? "gguf" : "tf"),
      j.tts && j.tts.tts === "ready" ? "ok" : "bad");
  } catch {
    setPill("textState", "文本引擎: 无响应", "bad");
    setPill("ttsState", "TTS: 无响应", "bad");
  }
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
    fillStyleSel(j.speaking_styles || []);
    // /api/mounts 的 roles 是名称列表 ["EmoCompanion"]，不是角色配置字典；
    // 直接用它填充角色下拉，避免 Object.keys 把数组当下标产生 "0"
    fillRoleSel(j.roles || ["EmoCompanion"]);
  } catch { /* 静默 */ }
}

function fillManualEmo() {
  const sel = $("manualEmo");
  sel.innerHTML = "";
  (EMOTIONS || []).forEach((e) => {
    const o = document.createElement("option");
    o.value = e; o.textContent = e; sel.appendChild(o);
  });
  if (MOUNT.emotion) sel.value = MOUNT.emotion;
}

function fillStyleSel(styles) {
  const sel = $("styleSel");
  sel.innerHTML = "";
  (styles && styles.length ? styles : ["自然"]).forEach((s) => {
    const o = document.createElement("option");
    o.value = s; o.textContent = s; sel.appendChild(o);
  });
  sel.value = (MOUNT.style && styles.includes(MOUNT.style)) ? MOUNT.style : "自然";
}

// names: 角色名列表；缺省回退到 ROLES 字典的键（角色设定抽屉拉取后）
function fillRoleSel(names) {
  const list = names || Object.keys(ROLES || {});
  const sel = $("roleSel");
  const cur = sel.value || MOUNT.role || list[0] || "EmoCompanion";
  sel.innerHTML = "";
  list.forEach((n) => {
    const o = document.createElement("option");
    o.value = n; o.textContent = n; sel.appendChild(o);
  });
  if (list.includes(cur)) sel.value = cur;
  $("roleTag").textContent = sel.value || "EmoCompanion";
}

// ==================== 会话管理 ====================
async function loadSessions() {
  try {
    const j = await (await fetch("/api/sessions")).json();
    renderSessions(j.sessions || []);
  } catch { /* 静默 */ }
}

function renderSessions(list) {
  const box = $("sessionList");
  box.innerHTML = "";
  list.forEach((s) => {
    const d = document.createElement("div");
    d.className = "session" + (s.id === SID ? " active" : "");
    d.innerHTML = `
      <div class="sess-txt">
        <div class="t">${escapeHtml(s.title)}</div>
        <div class="m">${fmtTime(s.updated)} · ${s.n_msgs} 条${s.words ? " · " + s.words + "字" : ""}${s.n_audio ? " · " + s.n_audio + "声" : ""}</div>
        ${s.last ? `<div class="sess-last">${escapeHtml(s.last)}</div>` : ""}
      </div>
      <button class="sess-del" title="删除">✕</button>`;
    d.addEventListener("click", (e) => {
      if (e.target.classList.contains("sess-del")) return;
      openSession(s.id);
    });
    d.querySelector(".sess-del").addEventListener("click", async (e) => {
      e.stopPropagation();
      await fetch(`/api/sessions/${s.id}`, { method: "DELETE" });
      if (SID === s.id) SID = null;
      loadSessions();
    });
    box.appendChild(d);
  });
}

async function newChat() {
  const j = await post("/api/sessions", { title: "新对话" });
  SID = j.id;
  $("chat").innerHTML = "";
  $("chatTitle").textContent = "新对话";
  AT_BOTTOM = true; updateScrollBtn();
  loadSessions();
  $("input").focus();
}

async function openSession(sid) {
  const j = await (await fetch(`/api/sessions/${sid}`)).json();
  SID = sid;
  $("chatTitle").textContent = j.title || "对话";
  const chat = $("chat");
  chat.innerHTML = "";
  (j.messages || []).forEach((m) => renderMsg(m, false));
  scrollToBottom(false);
  loadSessions();
}

// ==================== 消息渲染 ====================
function curveLabels(m) {
  // 把 tts_meta 的逐句 情感/语速 折线渲染成小标签，如 [开心 ×1.1][平静 ×0.9]
  const tm = m && m.tts_meta;
  const styles = (tm && tm.styles) || [];
  const rates = (tm && tm.rates) || [];
  if (!styles.length) return "";
  const tags = styles.map((s, i) => {
    const r = rates[i];
    const rTxt = r && Math.abs(r - 1) > 0.01 ? " ×" + Number(r).toFixed(2) : "";
    return `<span class="curve-tag" title="${escapeHtml(s)}">${escapeHtml(s)}${rTxt}</span>`;
  });
  return `<div class="curve">${tags.join("")}</div>`;
}
function renderMsg(m, animate) {
  const chat = $("chat");
  const d = document.createElement("div");
  d.className = "msg " + (m.role === "user" ? "user" : "assistant");
  const tTxt = m.ts ? fmtTime(m.ts) : "";
  const gen = m.text_stats && m.text_stats.latency_s ? `生成&nbsp;${Number(m.text_stats.latency_s).toFixed(1)}s` : "";
  let meta = `<div class="meta">` +
    `<span class="msg-time">${tTxt}</span>` +
    (gen ? `<span class="gen-time">${gen}</span>` : "") +
    (m.role === "assistant" && m.emotion ? `<span class="emo-tag">${escapeHtml(m.emotion)}</span>` : "") +
    (m.role === "assistant" && m.emotion_info && m.emotion_info.source ?
      `<span>${escapeHtml(m.emotion_info.source)}</span>` : "") +
    `</div>`;
  let audio = "";
  if (m.role === "assistant" && m.audio) {
    audio = `<button class="play-btn" data-audio="${m.audio}">▶ 播放语音</button>`;
  }
  d.innerHTML = `
    <div class="avatar">${m.role === "user" ? "我" : "缘"}</div>
    <div class="col">
      <div class="bubble">${escapeHtml(m.content)}</div>
      ${curveLabels(m)}
      ${meta}${audio}
    </div>`;
  if (animate) d.style.animation = "none";
  chat.appendChild(d);
  maybeStick();
  if (audio) {
    d.querySelector(".play-btn").addEventListener("click", (e) => {
      player.src = e.target.dataset.audio;
      player.play();
    });
  }
  return d;
}

function showTyping() {
  const chat = $("chat");
  const d = document.createElement("div");
  d.className = "msg assistant";
  d.id = "typing";
  d.innerHTML = `
    <div class="avatar">缘</div>
    <div class="col"><div class="bubble typing"><i></i><i></i><i></i></div></div>`;
  chat.appendChild(d);
  maybeStick();
}
function hideTyping() { const t = $("typing"); if (t) t.remove(); }

// ==================== 发送 ====================
async function send() {
  const input = $("input");
  const content = input.value.trim();
  if (!content) return;
  if (!SID) { const j = await post("/api/sessions", { title: "新对话" }); SID = j.id; }
  const btn = $("sendBtn");
  btn.disabled = true;
  renderMsg({ role: "user", content }, false);
  input.value = "";
  const st = $("genStats");
  const wantTts = $("wantTts").checked;
  st.textContent = "对方正在输入…"; st.classList.remove("err");
  try {
    const body = {
      content,
      want_tts: wantTts,
      backend: "gguf",
      emotion_mode: $("emotionModeSel").value,
      emotion: $("emotionModeSel").value === "manual" ? $("manualEmo").value : undefined,
      role: $("roleSel").value || "EmoCompanion",
      adapter: $("adapterSel").value,
      tone_variation: Number($("toneVar").value) || 0.35,
    };
    // 助手气泡：先"对方正在输入…"，文本就绪后整段显示，随后逐句播放语音
    const chat = $("chat");
    const col = document.createElement("div");
    col.className = "col";
    const bub = document.createElement("div");
    bub.className = "bubble typing-hd";
    bub.innerHTML = "对方正在输入<i></i><i></i><i></i>";
    const bubbleEl = document.createElement("div");
    bubbleEl.className = "msg assistant";
    bubbleEl.innerHTML = `<div class="avatar">缘</div>`;
    bubbleEl.appendChild(col);
    col.appendChild(bub);
    chat.appendChild(bubbleEl);
    maybeStick();

    const resp = await fetch(`/api/sessions/${SID}/stream`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
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
          if (evt.text_done) {
            bub.textContent = evt.reply || "";
            bub.classList.remove("typing-hd");
            maybeStick(true);
            // 文本阶段完成：若需 TTS，状态条由"对方正在输入…"切换为"对方正在发送语音…"
            // （wantTts=false 时保持"对方正在输入…"，不出现语音文案）
            if (wantTts) {
              st.textContent = "对方正在发送语音";
              st.classList.add("typing-voice");
            }
          } else if (evt.sentence_audio) {
            // 逐句语音就绪：生成播放按钮并立即播放（标注该句风格/语速）
            const pb = document.createElement("button");
            pb.className = "play-btn";
            const stTag = evt.style ? "·" + evt.style : "";
            const rtTag = evt.rate && Math.abs(evt.rate - 1) > 0.01 ? `(x${Number(evt.rate).toFixed(2)})` : "";
            pb.textContent = `▶ 第${evt.index + 1}句${stTag}${rtTag}`;
            pb.addEventListener("click", () => { player.src = evt.audio; player.play(); });
            col.appendChild(pb);
            if (wantTts) {
              player.src = evt.audio; player.play();
              // 逐句语音已就绪：状态条恢复为"就绪"
              st.textContent = "就绪";
              st.classList.remove("typing-voice");
            }
            maybeStick();
          } else if (evt.done) { final = evt; }
          else if (evt.error) { throw new Error(evt.error); }
        }
      }
    }
    if (final) {
      const emo = final.emotion || "";
      const ts = final.text_stats || {};
      const metaDiv = document.createElement("div");
      metaDiv.className = "meta";
      metaDiv.innerHTML = `<span class="msg-time">${fmtTime(Date.now() / 1000)}</span>` +
        (ts.latency_s ? `<span class="gen-time">生成&nbsp;${Number(ts.latency_s).toFixed(1)}s</span>` : "") +
        `<span class="emo-tag">${escapeHtml(emo)}</span>` +
        (final.emotion_info ? `<span>${escapeHtml(final.emotion_info.source)}</span>` : "") +
        (final.audio ? `<span>● 整段语音可播放</span>` : "");
      col.appendChild(metaDiv);
      // 语音情感/语速曲线标注（整段合成时 styles/rates 来自 tts_meta）
      if (final.tts_meta && final.tts_meta.styles && final.tts_meta.styles.length) {
        const cv = document.createElement("div");
        cv.className = "curve";
        cv.innerHTML = curveLabels({ tts_meta: final.tts_meta });
        col.appendChild(cv);
      }
      if (final.audio) {
        const pb = document.createElement("button");
        pb.className = "play-btn"; pb.textContent = "▶ 播放整段";
        pb.addEventListener("click", () => { player.src = final.audio; player.play(); });
        col.appendChild(pb);
      }
      let t = final.tts_meta || {};
      let ttsTxt;
      if (t.error) ttsTxt = `失败：${String(t.error).slice(0, 120)}`;
      else if (final.audio) ttsTxt = `已合成${t.n_sentences ? "(" + t.n_sentences + "句)" : ""}`;
      else ttsTxt = "跳过";
      st.textContent = `情感=${final.emotion}(${final.emotion_info && final.emotion_info.source}) ｜ 文本${ts.latency_s ? ts.latency_s + "s" : ""} ｜ TTS: ${ttsTxt}`;
      st.classList.remove("typing-voice");
      if (t.error) st.classList.add("err");
    }
    scrollToBottom(false);
    loadSessions();
  } catch (err) {
    hideTyping();
    st.textContent = "❌ " + err.message; st.classList.remove("typing-voice"); st.classList.add("err");
  } finally { btn.disabled = false; if (DBG_ON) refreshDebug(); }
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
  const box = $("traitList");
  box.innerHTML = "";
  const names = { warmth: "温暖", playfulness: "俏皮", sassiness: "傲娇", energy_baseline: "能量基线", formality: "正式度" };
  Object.entries(traits || {}).forEach(([k, v]) => {
    const d = document.createElement("div");
    d.className = "trait";
    d.innerHTML = `
      <div class="nm"><span>${names[k] || k}</span><b id="tv_${k}">${Number(v).toFixed(2)}</b></div>
      <input type="range" min="0" max="1" step="0.01" value="${Number(v).toFixed(2)}" data-trait="${k}"/>`;
    d.querySelector("input").addEventListener("input", (e) => {
      $("tv_" + k).textContent = Number(e.target.value).toFixed(2);
    });
    box.appendChild(d);
  });
}
function renderEmoKws(kws) {
  const box = $("emoKwList");
  box.innerHTML = "";
  Object.entries(kws || {}).forEach(([k, v]) => {
    const d = document.createElement("div");
    d.className = "emo-kw";
    d.innerHTML = `<span class="e">${escapeHtml(k)}</span><input data-emokw="${k}" value="${escapeHtml(String(v))}"/>`;
    box.appendChild(d);
  });
}
async function saveRole() {
  const st = $("roleSaveStat");
  st.textContent = "保存中…"; st.classList.remove("err");
  try {
    const traits = {};
    document.querySelectorAll("[data-trait]").forEach((i) => { traits[i.dataset.trait] = Number(i.value); });
    const kws = {};
    document.querySelectorAll("[data-emokw]").forEach((i) => { kws[i.dataset.emokw] = i.value; });
    const body = {
      name: $("roleName").value,
      desc: $("roleDesc").value,
      persona: $("rolePersona").value,
      traits,
      catchphrases: $("roleCatch").value.split(/[,，]/).map((s) => s.trim()).filter(Boolean),
      emotion_keywords: kws,
    };
    await post("/api/mounts", {});
    const r = await fetch("/api/roles", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error(await r.text());
    st.textContent = "✅ 已保存角色设定";
    ROLES = (await (await fetch("/api/roles")).json()).roles;
    fillRoleSel();
  } catch (err) {
    st.textContent = "❌ " + err.message; st.classList.add("err");
  }
}

// ==================== 调试模式 ====================
let DBG_ON = false;
function toggleDebug() {
  DBG_ON = !DBG_ON;
  $("debugBtn").classList.toggle("on", DBG_ON);
  $("debugPanel").classList.toggle("hidden", !DBG_ON);
  if (DBG_ON) refreshDebug();
}
async function refreshDebug() {
  const ids = ["dbEmotions","dbStyles","dbCatch","dbAdapters","dbCurEmo","dbAnchors",
    "dbPos","dbHol","dbOoc","dbMeta","dbModel","dbModelTTS","dbCalls","dbSecs","dbAvg","dbBackend","dbTTSSamp","dbLoad",
    "dbGpuName","dbVram","dbUtil","dbRam"];
  if (!DBG_ON) return;
  try {
    const j = await (await fetch("/api/debug")).json();
    const g = j.gpu, m = j.mount || {}, te = j.text_debug || {}, r = j.role || {};
    // 相关标签 · 角色情感
    $("dbEmotions").textContent = (j.emotions || []).join(" ");
    $("dbStyles").textContent = (j.styles || []).join(" ");
    $("dbCatch").textContent = (r.catchphrases || []).join("，") || "-";
    $("dbAdapters").textContent = (j.tts ? (j.tts.adapters || []).join("/") : "-") +
      (j.tts && j.tts.backend ? " · " + j.tts.backend + " 后端" : "");
    $("dbCurEmo").textContent = (m.emotion || "-") + (m.emotion_mode ? "(" + m.emotion_mode + ")" : "");
    // 相关标签 · 文本引擎
    const anchors = (te.meta && te.meta.anchors) || [];
    $("dbAnchors").textContent = anchors.join(" ") || "-";
    const ds = te.deai_summary || {};
    $("dbPos").textContent = ds.pos_tok + "/" + ds.pos_phr;
    $("dbHol").textContent = ds.hol_tok + "/" + ds.hol_phr;
    $("dbOoc").textContent = ds.ooc_phr;
    const mt = te.meta || {};
    $("dbMeta").textContent = (mt.beta != null ? "β" + mt.beta : "-") +
      (mt.T != null ? " · T" + mt.T : "") + (mt.n_vocab ? " · v" + mt.n_vocab : "");
    // 生成速度
    $("dbModel").textContent = te.model || "-";
    const tg = j.tts && j.tts.gguf ? j.tts.gguf : {};
    $("dbModelTTS").textContent = tg.model ? (tg.model + (tg.mmproj ? " · " + tg.mmproj : "")) : "-";
    const st = te.stats || {};
    $("dbCalls").textContent = (st.calls ?? "-") + " / " + (st.tokens ?? "-");
    $("dbSecs").textContent = st.seconds ?? "-";
    $("dbAvg").textContent = (st.avg_tok_s ?? 0) + " tok/s";
    $("dbBackend").textContent = (j.tts && j.tts.backend) || "-";
    $("dbTTSSamp").textContent =
      "T" + (m.temperature ?? "-") + " · top_k" + (m.top_k ?? "-") +
      " · top_p" + (m.top_p ?? "-") + " · " + (m.adapter ?? "-");
    $("dbLoad").textContent = (j.load && j.load.state && j.load.message) ?
      j.load.state + " · " + j.load.message : "未加载";
    $("dbLoad").className = "v" + (j.load && j.load.state === "ready" ? " ok"
      : (j.load && j.load.state === "error" ? " warn" : ""));
    // 显卡
    $("dbGpuName").textContent = g ? (g.name || "-") : "未检测到";
    $("dbVram").textContent = g ? `${(g.used_mb / 1024).toFixed(1)} / ${(g.total_mb / 1024).toFixed(1)} GB` : "-";
    $("dbVramBar").style.width = g ? Math.min(100, g.used_mb / g.total_mb * 100) + "%" : "0%";
    $("dbUtil").textContent = g ? g.util_gpu + "%" : "-";
    $("dbRam").textContent = j.process_ram_mb ?? "-";
  } catch (e) {
    ids.forEach((id) => { const el = $(id); if (el) el.textContent = "获取失败"; });
  }
}

// ==================== 加载模型 ====================
async function loadModel() {
  const btn = $("loadBtn");
  const bar = $("loadBar");
  const backend = "gguf";
  btn.disabled = true;
  bar.classList.remove("hidden", "err");
  bar.textContent = "正在加载 gguf 外挂情感模型…（请稍候）";
  try {
    const r = await post("/api/tts/load", { backend });
    bar.textContent = (r.state === "ready" ? "✅ " : "⚠️ ") + (r.message || "") +
      (r.elapsed ? "（耗时 " + r.elapsed + "s）" : "");
    if (r.state !== "ready") bar.classList.add("err");
  } catch (e) {
    bar.classList.remove("hidden"); bar.classList.add("err");
    bar.textContent = "❌ 加载失败: " + e.message;
  } finally {
    btn.disabled = false;
    if (DBG_ON) refreshDebug();
  }
}
async function pollLoadStatus() {
  try {
    const j = await (await fetch("/api/tts/load")).json();
    if (j && j.state === "loading") {
      $("loadBar").textContent = "加载中… " + (j.message || ""); 
      $("loadBar").classList.remove("hidden", "err");
    } else if ($("loadBar").classList.contains("hidden") === false &&
               $("loadBar").textContent.startsWith("加载中")) {
      $("loadBar").textContent = (j.state === "ready" ? "✅ " : "⚠️ ") + (j.message || "") +
        (j.elapsed ? "（耗时 " + j.elapsed + "s）" : "");
      if (j.state !== "ready") $("loadBar").classList.add("err");
    }
  } catch { /* ignore */ }
}

// ==================== 语音合成工作室 ====================
function fillStudioEmo() {
  const sel = $("ssEmotion");
  sel.innerHTML = "";
  (EMOTIONS || []).forEach((e) => {
    const o = document.createElement("option");
    o.value = e; o.textContent = e; sel.appendChild(o);
  });
  sel.value = MOUNT.emotion && sel.querySelector(`option[value="${MOUNT.emotion}"]`) ? MOUNT.emotion : "开心";
}

// 直接用后端 /api/tts/synthesize 合成（外挂 GGUF）
async function studioSynth(cfg) {
  const st = $("ssStatus"), btn = $("ssBtn");
  btn.disabled = true; st.textContent = "合成中…"; st.classList.remove("err", "ok");
  try {
    const r = await fetch("/api/tts/synthesize", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: cfg.text, emotion: cfg.emotion,
        backend: cfg.backend ?? "gguf",
        adapter: cfg.adapter ?? "emotion",
        rate: Number($("ssRate").value) || 1.0,
      }),
    });
    if (!r.ok) throw new Error((await r.text()).slice(0, 400) || "HTTP " + r.status);
    const meta = JSON.parse(r.headers.get("X-TTS-Meta") || "{}");
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const au = $("ssAudio");
    au.src = url; au.load();
    $("ssResult").classList.remove("hidden");
    $("ssMeta").innerHTML =
      `<div>情感 <b>${meta.emotion ?? cfg.emotion}</b> ｜ 后端 <b>${meta.backend ?? "gguf"}</b> ｜ adapter <span>${meta.adapter ?? "voice+emotion"}</span></div>` +
      `<div>采样率 <span>${meta.sr ?? 24000}Hz</span> ｜ 音频 ≈ <span>${(meta.audio_seconds ?? 0).toFixed(2)}s</span> ｜ 耗时 <span>${(meta.wall_s ?? 0).toFixed(2)}s</span></div>` +
      `<div>温度 <span>${meta.temperature}</span> ｜ top_k <span>${meta.top_k}</span> ｜ 语速 <span>${meta.rate}</span> ｜ ${meta.strategy ?? "trained_lora_preset"}</div>`;
    au.play();
    st.textContent = "✅ 已合成并播放"; st.classList.add("ok");
  } catch (err) { st.textContent = "❌ " + err.message; st.classList.add("err"); }
  finally { btn.disabled = false; }
}

function bindStudio() {
  // Tab 切换（对话台 / 语音合成）
  document.querySelectorAll(".nav-tab").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll(".nav-tab").forEach((x) => x.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach((x) => x.classList.add("hidden"));
      b.classList.add("active");
      $("tab-" + b.dataset.tab).classList.remove("hidden");
    });
  });

  // 滑块输出同步
  $("ssTemp").addEventListener("input", (e) => { $("ssTempOut").textContent = Number(e.target.value).toFixed(2); });
  $("ssTopK").addEventListener("input", (e) => { $("ssTopKOut").textContent = e.target.value; });
  $("ssRate").addEventListener("input", (e) => { $("ssRateOut").textContent = "x" + Number(e.target.value).toFixed(2); });

  // 手动合成
  $("ssBtn").addEventListener("click", () => {
    const text = $("ssText").value.trim();
    if (!text) { $("ssStatus").textContent = "❌ 请输入文本"; $("ssStatus").classList.add("err"); return; }
    studioSynth({ text, emotion: $("ssEmotion").value });
  });

  // 自动情感识别并合成
  $("ssDetectBtn").addEventListener("click", async () => {
    const text = $("ssDetectText").value.trim();
    const st = $("ssStatus");
    if (!text) { st.textContent = "❌ 请输入文本"; st.classList.add("err"); return; }
    st.textContent = "识别情感中…"; st.classList.remove("err", "ok");
    try {
      const j = await post("/api/tts/detect", { text, user_msg: text });
      const emo = j.label || "平静";
      $("ssEmotion").value = emo;
      st.textContent = `自动识别：${emo}（${j.source}，置信 ${(j.confidence ?? 0).toFixed(2)}）`;
      await studioSynth({ text, emotion: emo });
    } catch (err) { st.textContent = "❌ " + err.message; st.classList.add("err"); }
  });
}

// ==================== 事件绑定 ====================
function bindEvents() {
  bindChatScroll();
  $("newChat").addEventListener("click", newChat);
  $("sendBtn").addEventListener("click", send);
  $("input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });
  $("roleBtn").addEventListener("click", openRoleDrawer);
  $("roleClose").addEventListener("click", () => $("roleDrawer").classList.add("hidden"));
  $("roleDrawer").addEventListener("click", (e) => { if (e.target.id === "roleDrawer") $("roleDrawer").classList.add("hidden"); });
  $("roleSave").addEventListener("click", saveRole);
  $("debugBtn").addEventListener("click", toggleDebug);
  $("loadBtn").addEventListener("click", loadModel);
  setInterval(pollLoadStatus, 2000);
  pollLoadStatus();

  // 挂载切换
  $("adapterSel").addEventListener("change", (e) => post("/api/mounts", { adapter: e.target.value }));
  $("emotionModeSel").addEventListener("change", (e) => {
    $("manualEmoWrap").classList.toggle("hidden", e.target.value !== "manual");
    post("/api/mounts", { emotion_mode: e.target.value });
  });
  $("manualEmo").addEventListener("change", (e) => post("/api/mounts", { emotion: e.target.value }));
  $("styleSel").addEventListener("change", (e) => { MOUNT.style = e.target.value; post("/api/mounts", { style: e.target.value }); });
  $("wantTts").addEventListener("change", (e) => post("/api/mounts", { want_tts: e.target.checked }));
  $("toneVar").addEventListener("change", (e) => post("/api/mounts", { tone_variation: Number(e.target.value) }));
  $("roleSel").addEventListener("change", (e) => { $("roleTag").textContent = e.target.value; post("/api/mounts", { role: e.target.value }); });

  // 快速问答
  document.querySelectorAll(".qa-chip").forEach((b) => {
    b.addEventListener("click", () => { $("input").value = b.dataset.q; send(); });
  });
}

init();
