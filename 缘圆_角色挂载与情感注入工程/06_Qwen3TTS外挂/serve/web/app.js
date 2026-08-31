// 缘圆 前端后端一体化 —— 前端逻辑
const $ = (id) => document.getElementById(id);

// ---------- Tab 切换 ----------
document.querySelectorAll(".tab").forEach((b) => {
  b.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    $("tab-" + b.dataset.tab).classList.add("active");
  });
});

function setPill(id, txt, state) {
  const el = $(id);
  el.textContent = txt;
  el.classList.toggle("ok", state === "ok");
  el.classList.toggle("bad", state === "bad");
}

// ---------- 状态探活 ----------
async function refreshHealth() {
  try {
    const r = await fetch("/api/tts/health");
    const j = await r.json();
    setPill("ttsState", "TTS: " + (j.tts || "?"), j.tts === "ready" ? "ok" : "bad");
  } catch { setPill("ttsState", "TTS: 无响应", "bad"); }
  try {
    const r = await fetch("/api/text/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: [{ role: "user", content: "ping" }], max_new: 2 }),
    });
    setPill("textState", "文本引擎: " + (r.ok ? "在线" : r.status), r.ok ? "ok" : "bad");
  } catch { setPill("textState", "文本引擎: 离线", "bad"); }
}

// ---------- 语音合成 ----------
async function loadTtsModels() {
  try {
    const r = await fetch("/api/tts/models");
    const j = await r.json();
    const va = $("voiceAdapter");
    Object.entries(j.adapters || {}).forEach(([name]) => {
      const o = document.createElement("option");
      o.value = name; o.textContent = name + " 外挂";
      va.appendChild(o);
    });
    const em = $("emotion");
    (j.emotions || []).forEach((e) => {
      const o = document.createElement("option");
      o.value = e; o.textContent = e;
      em.appendChild(o);
    });
  } catch { /* 模型列表拉取失败时静默 */ }
}

$("ttsForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = $("ttsBtn"), st = $("ttsStatus");
  btn.disabled = true; st.textContent = "合成中…"; st.classList.remove("err");
  try {
    const r = await fetch("/api/tts/synthesize", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: $("ttsText").value, emotion: $("emotion").value, backend: $("ttsBackend").value }),
    });
    if (!r.ok) {
      const err = await r.text();
      throw new Error((r.status === 503 ? "TTS 组件缺失" : "合成失败") + ": " + err.slice(0, 400));
    }
    const meta = JSON.parse(r.headers.get("X-TTS-Meta") || "{}");
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const au = $("ttsAudio");
    au.src = url; au.load(); au.play();
    $("ttsResult").classList.remove("hidden");
    $("ttsMeta").textContent = `情感=${meta.emotion} | 后端=${meta.backend ?? "gguf"} | 采样率=${meta.sr}Hz | 音频≈${meta.audio_seconds}s | 耗时=${meta.seconds ?? meta.wall_s}s | RTF=${meta.rtf ?? "n/a"} | 策略=${meta.strategy}`;
    st.textContent = "✅ 完成"; st.classList.remove("err");
  } catch (err) {
    st.textContent = "❌ " + err.message; st.classList.add("err");
  } finally { btn.disabled = false; }
});

// ---------- 文本生成 ----------
function defaultMsgs() {
  return JSON.stringify(
    [{ role: "user", content: "你好缘圆，今天过得怎么样？" }], null, 2);
}
$("textMsgs").value = defaultMsgs();

$("textForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = $("textBtn"), st = $("textStatus");
  btn.disabled = true; st.textContent = "生成中…"; st.classList.remove("err");
  let messages;
  try { messages = JSON.parse($("textMsgs").value); }
  catch (err) { st.textContent = "❌ messages JSON 无效"; st.classList.add("err"); btn.disabled = false; return; }
  try {
    const useTts = $("autoTts").checked;
    const url = useTts ? "/api/pipeline/talk" : "/api/text/chat";
    const r = await fetch(url, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages, role: $("textRole").value,
        max_new: Number($("textMaxNew").value) || 128,
        temperature: Number($("textTemp").value) || 0.9,
        top_p: Number($("textTopP").value) || 0.9,
        backend: ($("pipeBackend")?.value) || "gguf",
      }),
    });
    const out = $("textOutput");
    if (!r.ok) {
      const err = await r.text();
      throw new Error("对话/语音管线错误: " + err.slice(0, 600));
    }
    // ---- 自动转语音（对话完成即交互：文本 + 自动情感 + 音频）----
    if (useTts) {
      const pipe = JSON.parse(r.headers.get("X-Pipe") || "{}");
      const reply = pipe.reply || "";
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const ta = $("pipeAudio");
      ta.src = url; ta.load();
      const t = pipe.tts || {};
      $("pipeMeta").textContent =
        `自动情感=${pipe.emotion} ｜ 精度=${pipe.confidence} ｜ 来源=${pipe.source} ｜ 后端=${pipe.backend || "gguf"}` +
        (t.rtf != null ? ` ｜ RTF=${t.rtf} ｜ 音频≈${t.audio_seconds}s` : " ｜ RTF=n/a");
      $("pipeRow").classList.remove("hidden");
      ta.play();
    } else {
      const j = await r.json();
      const reply = (typeof j.text === "string" ? j.text
        : j.choices?.[0]?.message?.content ?? JSON.stringify(j));
      const div = document.createElement("div");
      div.className = "msg assistant";
      div.innerHTML = `<div class="who">缘圆 · ${j.role ?? "default"}</div>` +
        escapeHtml(reply);
      out.appendChild(div);
    }
    st.textContent = "✅ 已生成"; st.classList.remove("err");
  } catch (err) {
    st.textContent = "❌ " + err.message; st.classList.add("err");
  } finally { btn.disabled = false; }
});

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// ---------- 初始化 ----------
loadTtsModels();
refreshHealth();
setInterval(refreshHealth, 15000);