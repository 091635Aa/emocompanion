/* 运行大屏逻辑（V6 风格） */
const $ = (id) => document.getElementById(id);
const 状态色 = { "待执行": "", "执行中": "run", "完成": "done", "失败": "err", "跳过": "" };
const 维度色 = { "开心": "#4ECDC4", "悲伤": "#4D96FF", "愤怒": "#FF5D8F",
                 "中性": "#8b98b3", "复杂混合": "#A06BE0", "待定": "#5d6b85" };

async function api(path, body) {
  const opts = body === undefined
    ? { headers: { "Accept": "application/json" } }
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, opts);
  let json = null;
  try { json = await r.json(); } catch (e) {}
  if (!json) throw new Error(`HTTP ${r.status}`);
  if (json.状态 === "error") throw new Error(json.错误 || "接口错误");
  return json.数据;
}

/* Token/秒 趋势图（纯 canvas 折线） */
const 历史点 = [];   // {label, tps}
function 画图() {
  const canvas = $("tokenChart");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  if (!历史点.length) {
    ctx.fillStyle = "#5d6b85"; ctx.font = "12px Microsoft YaHei";
    ctx.fillText("暂无生成记录", 10, h / 2);
    return;
  }
  const 数据 = 历史点.slice(-20);
  const maxV = Math.max(1, ...数据.map(d => d.tps));
  const pad = 26;
  ctx.strokeStyle = "rgba(78,205,196,.12)";
  ctx.lineWidth = 1;
  for (let i = 1; i <= 4; i++) {
    const y = pad + (h - pad * 2) * i / 4;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }
  ctx.strokeStyle = "#4ECDC4"; ctx.lineWidth = 2;
  ctx.beginPath();
  数据.forEach((d, i) => {
    const x = pad + (w - pad * 2) * i / Math.max(1, 数据.length - 1);
    const y = h - pad - (h - pad * 2) * d.tps / maxV;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#4ECDC4";
  数据.forEach((d, i) => {
    const x = pad + (w - pad * 2) * i / Math.max(1, 数据.length - 1);
    const y = h - pad - (h - pad * 2) * d.tps / maxV;
    ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
  });
  ctx.fillStyle = "#8b98b3"; ctx.font = "10px Consolas";
  ctx.fillText(`峰值 ${maxV.toFixed(1)} token/s`, pad, 12);
}

/* 稳定度 / 情感命中率 双线趋势图 */
const 稳定点 = [];   // {稳定度, 命中率}
function 画稳定图() {
  const canvas = $("stabilityChart");
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  if (!稳定点.length) {
    ctx.fillStyle = "#5d6b85"; ctx.font = "12px Microsoft YaHei";
    ctx.fillText("等待采样…", 10, h / 2);
    return;
  }
  const 数据 = 稳定点.slice(-60);
  const pad = 26;
  // 网格
  ctx.strokeStyle = "rgba(255,255,255,.06)"; ctx.lineWidth = 1;
  for (let i = 1; i <= 4; i++) {
    const y = pad + (h - pad * 2) * i / 4;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }
  const 线 = (键, 颜色) => {
    ctx.strokeStyle = 颜色; ctx.lineWidth = 2;
    ctx.beginPath();
    数据.forEach((d, i) => {
      const x = pad + (w - pad * 2) * i / Math.max(1, 数据.length - 1);
      const y = h - pad - (h - pad * 2) * Math.min(100, Math.max(0, d[键])) / 100;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  };
  线("稳定度", "#37d67a");
  线("命中率", "#FF5D8F");
}

/* 开关渲染与切换 */
const 开关定义 = [
  { 名称: "API", 键: "启用API", 说明: "对外 OpenAI 接口" },
  { 名称: "RAG", 键: "启用RAG", 说明: "向量检索注入" },
  { 名称: "LoRA", 键: "启用LoRA", 说明: "适配器外挂" },
  { 名称: "记忆", 键: "启用记忆", 说明: "超长期记忆" },
  { 名称: "策略", 键: "动态策略", 说明: "A/B/C" },
];

async function 渲染开关() {
  try {
    const s = await api("/api/v1/switches");
    const 区 = $("开关区");
    区.innerHTML = 开关定义.map(d => {
      const v = s[d.键];
      const 显示 = d.名称 === "策略" ? (v || "跟随配置") : (v === null ? "跟随配置" : (v ? "开" : "关"));
      const on = d.名称 === "策略" ? !!v : (v === true);
      return `<div class="switch-card ${on ? "on" : ""}" data-名称="${d.名称}" title="${d.说明}">
        <span class="s-name">${d.名称}</span><span class="s-state">${显示}</span>
      </div>`;
    }).join("");
    区.querySelectorAll(".switch-card").forEach(el => {
      el.addEventListener("click", async () => {
        const 名称 = el.dataset.名称;
        const s = await api("/api/v1/switches");
        let 值;
        if (名称 === "策略") {
          const 顺序 = ["A", "B", "C"];
          const 当前 = s.动态策略 || "跟随";
          值 = 当前 === "A" ? "B" : 当前 === "B" ? "C" : 当前 === "C" ? null : "A";
          await api("/api/v1/switch", { 名称: "策略", 值 });
        } else {
          值 = !(s[d.键] === true);
          await api("/api/v1/switch", { 名称, 值 });
        }
        await 渲染开关();
      });
    });
  } catch (e) {}
}

/* 流程条（模块化管线 + 连线） */
async function 渲染流程() {
  try {
    const f = await api("/api/v1/flow/status");
    $("流程条").innerHTML = f.节点.map((n, i) =>
      `<span class="st ${状态色[n.状态]}" title="${n.详情 || ""}">${n.节点}·${n.状态}</span>` +
      (i < f.节点.length - 1 ? `<i class="pl"></i>` : "")
    ).join("");
  } catch (e) {}
}

/* 参数直观调整 */
async function 渲染参数() {
  try {
    const p = await api("/api/v1/params");
    $("参数行").style.display = p.推荐.λ != null ? "" : "none";
    if (p.推荐.λ == null) return;
    const 编辑 = (k) => {
      const 推荐 = p.推荐[k];
      const 当前 = p.覆盖[k] ?? 推荐;
      return `<input type="number" step="0.01" min="0.01" max="1" id="参数输入-${k}"
        value="${当前}" title="${p.说明[k]}" style="width:76px;font-size:12px;padding:4px 8px">`;
    };
    $("参数λ").innerHTML = `<span class="small" style="color:#4ECDC4">λ</span> ${编辑("λ")}`;
    $("参数γ").innerHTML = `<span class="small" style="color:#A06BE0">γ</span> ${编辑("γ")}`;
    $("参数τ").innerHTML = `<span class="small" style="color:#FF9F43">τ</span> ${编辑("τ")}`;
    ["λ", "γ", "τ"].forEach(k => {
      const el = $(`参数输入-${k}`);
      el.onchange = async () => {
        try {
          await api("/api/v1/params", { 名称: k, 值: parseFloat(el.value) });
          await 渲染参数();
        } catch (e) { alert(`${k} 调整失败：${e.message}`); }
      };
    });
  } catch (e) {}
}
$("参数重置").addEventListener("click", async () => {
  try { await api("/api/v1/params/reset", {}); await 渲染参数(); } catch (e) {}
});

/* 日志 */
async function 渲染日志() {
  try {
    const r = await api("/api/v1/logs?数量=40");
    $("日志区").innerHTML = r.日志.map(x =>
      `<div class="lg"><span class="t">${x.时间}</span><span class="lvl">${x.级别}</span><span class="m">${x.消息}</span></div>`
    ).join("") || '<div class="muted">暂无日志</div>';
  } catch (e) {}
}

/* 情感向量 */
function 渲染情感(情感) {
  const 分布 = 情感.维度分布 || {};
  const 总 = 情感.样本数 || 1;
  const 全部维度 = ["开心", "悲伤", "愤怒", "中性", "复杂混合"];
  $("情感条").innerHTML = 全部维度.map(d => {
    const n = 分布[d] || 0;
    return `<div class="bar-row">
      <span class="name">${d}</span>
      <div class="track"><div style="width:${(n / 总 * 100).toFixed(1)}%;background:${维度色[d]}"></div></div>
      <span class="num">${n}</span>
    </div>`;
  }).join("");
  $("k-情感sub").textContent = 情感.最近维度 ? `最近: ${情感.最近维度} | 密度 ${情感.最近密度 ?? "—"}` : "暂无样本";
}

/* 记忆 */
async function 渲染记忆() {
  try {
    const st = await api("/api/v1/memory/status");
    $("记忆统计").textContent = `启用:${st.启用 ? "开" : "关"} | 共 ${st.总条数} 条 | 写入 ${st.写入数} | 检索 ${st.检索数}`;
    const 列表 = await api("/api/v1/memory/list?数量=8");
    $("记忆区").innerHTML = (列表.记忆列表 || []).map(m =>
      `<div class="mem"><span class="tag">[${m.情感维度}]</span> <span class="txt">${m.提示词} → ${m.摘要}</span></div>`
    ).join("") || '<div class="muted">记忆库为空（开启记忆开关后生成会自动写入）</div>';
  } catch (e) {}
}

/* 打标监控 */
async function 渲染打标() {
  try {
    const t = await api("/api/v1/label/status");
    $("打标统计").textContent = t.运行中 ? "运行中…" : (t.历史.length ? `已完成 ${t.历史.length} 次` : "");
    $("打标当前").innerHTML = t.运行中
      ? `<div class="bar"><div style="width:${t.总数 ? (t.进度 / t.总数 * 100).toFixed(0) : 0}%"></div></div>
         <div class="muted small mt8">${t.批次名} ${t.进度}/${t.总数}：${t.当前提示 || ""}</div>`
      : (t.结果 && t.结果.状态 === "完成"
        ? `<div class="banner ok small">最近批次 ${t.结果.批次名 || t.批次名 || ""} 完成 · ${t.结果.条目数} 条 → ${(t.结果.输出文件 || []).join(" ")}</div>`
        : '<div class="muted small">暂无打标任务</div>');
    $("打标历史").innerHTML = (t.历史 || []).slice().reverse().slice(0, 10).map(h =>
      `<div class="lh"><span>${h.批次名} · ${h.条目数 ?? ""}条</span><span class="${h.错误 ? "fail" : "ok"}">${h.错误 ? "失败" : h.完成时间 || ""}</span></div>`
    ).join("") || "";
  } catch (e) {}
}

/* 主刷新 */
async function 刷新() {
  try {
    const m = await api("/api/v1/monitor");
    const 状态 = await api("/api/v1/model/status");
    $("k-服务").textContent = "运行中";
    $("k-服务sub").textContent = `启动于 ${m.服务启动时间}`;
    $("k-模型").textContent = 状态.已加载模型名 || "未加载";
    $("k-模型sub").textContent = 状态.加载状态;
    const 显存 = m.显存;
    if (显存) {
      $("k-显存").textContent = `${显存.显存MB} MB`;
      $("k-显存sub").textContent = `共 ${显存.总显存MB} MB · ${显存.模型名 || ""}`;
    } else { $("k-显存").textContent = "—"; $("k-显存sub").textContent = "GPU 未检测"; }
    const 内存 = m.系统内存;
    if (内存) { $("k-内存").textContent = `${内存.占用%}%`; $("k-内存sub").textContent = `可用 ${内存.可用MB} / ${内存.总MB} MB`; }
    else { $("k-内存").textContent = "—"; }
    $("k-token").textContent = m.总token数.toLocaleString();
    $("k-tokensub").textContent = `生成 ${m.生成次数} 次`;
    $("k-tps").textContent = m.token每秒.toFixed(1);
    历史点.push({ tps: m.token每秒 });
    历史点.splice(0, Math.max(0, 历史点.length - 40));
    画图();
    $("k-稳定").textContent = m.稳定度;
    const 情感 = m.情感统计;
    $("k-情感").textContent = 情感.最近命中率 != null ? (情感.最近命中率 * 100).toFixed(1) + "%" : "—";
    渲染情感(情感);
    // 稳定度 / 情感命中率 采样
    稳定点.push({ 稳定度: m.稳定度, 命中率: (情感.最近命中率 != null ? 情感.最近命中率 * 100 : 0) });
    稳定点.splice(0, Math.max(0, 稳定点.length - 120));
    画稳定图();
  } catch (e) {
    $("k-服务").textContent = "不可用";
    $("k-服务").className = "kpi rose";
  }
}

(async function 启动() {
  await Promise.all([刷新(), 渲染开关(), 渲染流程(), 渲染日志(), 渲染记忆(), 渲染打标(), 渲染参数()]);
  setInterval(() => {
    刷新();
    渲染开关();
    渲染流程();
    渲染日志();
    渲染记忆();
    渲染打标();
    渲染参数();
  }, 2500);
})();
