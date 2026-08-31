/* 专业控制台逻辑 */
const $ = (id) => document.getElementById(id);
const 状态色 = { "待执行": "", "执行中": "run", "完成": "done", "失败": "err", "跳过": "skip" };
let 已加载模型名 = null;

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

/* 页签切换 */
document.querySelectorAll("#页签 button").forEach(b => {
  b.addEventListener("click", () => {
    document.querySelectorAll("#页签 button").forEach(x => x.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    $(`panel-${b.dataset.panel}`).classList.add("active");
  });
});

/* ── 模型面板 ── */
async function 刷新模型() {
  try {
    const st = await api("/api/v1/model/status");
    已加载模型名 = st.已加载模型名;
    $("模型徽章").textContent = st.加载状态 === "已加载" ? `已加载: ${st.已加载模型名}` : st.加载状态;
    $("模型徽章").className = "badge " + (st.加载状态 === "已加载" ? "ok" : st.加载状态 === "加载中" ? "run" : "");
    $("显存徽章").textContent = st.显存MB != null ? `显存 ${st.显存MB} MB` : "GPU 未检测";

    const 描述 = st.模型描述;
    const 参数 = 描述 ? 描述.参数 : null;
    $("状态卡片").innerHTML = `
      <dt>加载状态</dt><dd><b>${st.加载状态}</b>${st.加载错误 ? "（" + st.加载错误 + "）" : ""}</dd>
      <dt>已加载模型</dt><dd>${st.已加载模型名 || "—"}</dd>
      <dt>加载耗时</dt><dd>${st.加载耗时 != null ? st.加载耗时 + "s" : "—"}</dd>
      <dt>hidden_size</dt><dd>${参数 ? 描述.hidden_size : "—"}</dd>
      <dt>vocab_size</dt><dd>${参数 ? 描述.vocab_size : "—"}</dd>
      <dt>λ / γ / τ</dt><dd>${参数 ? `${参数.λ} / ${参数.γ} / ${参数.τ}（${参数.来源}）` : "—"}</dd>
      <dt>归一化基准</dt><dd>${描述 ? 描述.归一化基准 : "—"}</dd>
      <dt>量化 / 动态策略</dt><dd>${描述 ? `${描述.量化} / ${描述.动态策略}` : "—"}</dd>
      <dt>RAG / LoRA / 长上下文</dt><dd>${描述 ? `${描述.rag} / ${描述.lora || "无"} / ${描述.长上下文}` : "—"}</dd>
      <dt>自动测试</dt><dd>${描述 ? (描述.自动测试 ? "开启" : "关闭") : "—"}</dd>`;

    $("加载按钮").disabled = st.加载状态 === "加载中" || st.加载状态 === "已加载";
    $("卸载按钮").disabled = st.加载状态 !== "已加载";
  } catch (e) {
    $("服务徽章").textContent = "服务不可用"; $("服务徽章").className = "badge err";
  }
}

async function 刷新模型库() {
  try {
    const 列表 = await api("/api/v1/models");
    const tbody = $("模型库表").querySelector("tbody");
    tbody.innerHTML = 列表.map(m => {
      const p = m.参数;
      return `<tr>
        <td>${m.模型名}</td><td class="small">${m.路径}</td><td>${m.类型}</td>
        <td>${m.量化}</td><td>${m.动态策略}</td><td>${m.hidden_size}</td><td>${m.vocab_size}</td>
        <td>${p.λ} / ${p.γ} / ${p.τ}</td><td>${p.来源}</td><td>${m.归一化基准}</td>
        <td>${m.rag}</td><td>${m.lora || "—"}</td><td>${m.长上下文}</td><td>${m.自动测试}</td>
      </tr>`;
    }).join("") || '<tr><td colspan="14" class="muted">暂无注册模型</td></tr>';
  } catch (e) {}
}

async function 刷新模型下拉() {
  try {
    const 数据 = await api("/api/v1/scan");
    $("模型下拉").innerHTML = 数据.可用模型.map(m =>
      `<option value="${m.模型名}" data-路径="${m.路径}">${m.模型名} ${m.已注册 ? "(已注册)" : ""} · dim=${m.hidden_size}</option>`
    ).join("");
  } catch (e) { $("模型下拉").innerHTML = '<option>扫描失败</option>'; }
}

$("注册按钮").addEventListener("click", async () => {
  const 选项 = $("模型下拉").selectedOptions[0];
  if (!选项) return;
  try {
    const 描述 = await api("/api/v1/model/register", {
      模型名: 选项.value, 路径: 选项.dataset.路径, 类型: "标准", 量化: "4bit",
      动态策略: "B", rag: false, 长上下文: false, 自动测试: true,
    });
    $("注册日志").textContent = JSON.stringify(描述, null, 2);
    await 刷新模型库();
  } catch (e) { $("注册日志").textContent = "注册失败：" + e.message; }
});

$("加载按钮").addEventListener("click", async () => {
  try {
    await api("/api/v1/model/load", { 模型名: $("模型下拉").value });
    await 刷新模型();
  } catch (e) { alert("加载失败：" + e.message); }
});

$("卸载按钮").addEventListener("click", async () => {
  try { await api("/api/v1/model/unload"); await 刷新模型(); }
  catch (e) { alert("卸载失败：" + e.message); }
});

$("下载按钮").addEventListener("click", () => {
  const 仓库 = prompt("输入 HuggingFace 仓库名（公共仓库无需密钥，如 Qwen/Qwen2.5-1.5B-Instruct）：");
  if (!仓库) return;
  const 镜像 = confirm("使用 hf-mirror 镜像加速？") ? "https://hf-mirror.com" : null;
  api("/api/v1/model/download", { 目标名: 仓库.split("/").pop(), HF仓库: 仓库, 镜像 }).then(() => {
    alert("下载任务已启动，完成后自动注册。可在模型列表查看。");
  }).catch(e => alert("下载启动失败：" + e.message));
});

/* ── 生成面板 ── */
$("生成按钮").addEventListener("click", async () => {
  const 提示 = $("生成提示").value.trim();
  if (!提示) return;
  $("生成按钮").disabled = true;
  try {
    const 结果 = await api("/api/v1/generate", { 模型名: 已加载模型名, 提示词: 提示, 最大token: parseInt($("生成token").value) || 128 });
    $("生成原始").textContent = JSON.stringify(结果, null, 2);
    await 刷新历史();
  } catch (e) { $("生成原始").textContent = "生成失败：" + e.message; }
  $("生成按钮").disabled = false;
});

async function 刷新历史() {
  try {
    const h = await api("/api/v1/generate/history");
    const tbody = $("历史表").querySelector("tbody");
    tbody.innerHTML = h.历史.map(r => `<tr>
      <td class="small">${r.提示词}</td><td class="small">${(r.文本 || "").slice(0, 60)}</td>
      <td>${r.平均熵.toFixed(3)}</td><td>${r.重复率.toFixed(3)}</td><td>${r.情感命中率.toFixed(3)}</td>
      <td>${r.λ}</td><td>${r.γ}</td><td>${r.τ}</td><td>${r.步数}</td><td>${r.耗时}s</td>
    </tr>`).join("") || '<tr><td colspan="10" class="muted">暂无生成记录</td></tr>';
  } catch (e) {}
}

/* ── 测试面板 ── */
$("测试按钮").addEventListener("click", async () => {
  try {
    await api("/api/v1/test/activate", { 范围: $("测试范围").value, 模型名: 已加载模型名 });
  } catch (e) { alert("激活测试失败：" + e.message); }
});

async function 刷新测试() {
  try {
    const t = await api("/api/v1/test/status");
    $("测试状态").textContent = t.运行中 ? "运行中…" : (t.最近结果 ? "最近完成" : "—");
    if (t.最近结果 && !t.运行中) {
      const r = t.最近结果;
      $("测试结果").innerHTML = `<div class="banner ${r.失败 === 0 ? "ok" : "err"}">
        通过 ${r.通过} / ${(r.通过 || 0) + (r.失败 || 0)}（${r.总耗时 || "—"}s）${r.模型名 ? "· 模型: " + r.模型名 : ""}${r.自动 ? " · 自动触发" : ""}</div>` +
        (r.明细 || []).map(d => `<div class="small mb12"><b>${d.通过 ? "PASS" : "FAIL"}</b> · ${d.名称}（${d.耗时}s）<br>${d.说明 || ""}</div>`).join("");
    }
    // 报告列表
    if (t.报告列表.length && !$("报告下拉").options.length) {
      $("报告下拉").innerHTML = t.报告列表.map(f => `<option>${f}</option>`).join("");
      刷新报告();
    }
  } catch (e) {}
}

async function 刷新报告() {
  const 文件名 = $("报告下拉").value;
  if (!文件名) return;
  try {
    const r = await api(`/api/v1/test/reports?文件名=${encodeURIComponent(文件名)}`);
    $("报告内容").textContent = JSON.stringify(r.报告, null, 2);
  } catch (e) {}
}
$("报告下拉").addEventListener("change", 刷新报告);

/* ── 打标面板 ── */
$("打标按钮").addEventListener("click", async () => {
  try {
    await api("/api/v1/label/run", { 模型名: 已加载模型名, 批次名: $("批次名").value || "批次1", 最大token: parseInt($("打标token").value) || 128 });
  } catch (e) { alert("打标启动失败：" + e.message); }
});

async function 刷新打标() {
  try {
    const t = await api("/api/v1/label/status");
    if (t.运行中) {
      const p = t.总数 ? Math.round(t.进度 / t.总数 * 100) : 0;
      $("打标进度").style.width = p + "%";
    } else if (t.结果 && t.结果.状态 === "完成") {
      $("打标进度").style.width = "100%";
    }
    $("打标状态").textContent = JSON.stringify(t, null, 2);
    // 标注文件清单
    const 输出 = t.输出目录;
    const tbody = $("标注表").querySelector("tbody");
    try {
      const r = await fetch(`/api/v1/records`);
      // 简单列出打标目录文件：通过 label 状态接口的输出目录 + 服务端 glob 无法直接列出，
      // 这里展示最近输出文件
      const 文件 = (t.结果 && t.结果.输出文件 || []).map(f => `<tr><td>${f.split("\\").pop()}</td><td>输出</td></tr>`).join("");
      tbody.innerHTML = 文件 || '<tr><td colspan="2" class="muted">暂无打标输出</td></tr>';
    } catch (e) { tbody.innerHTML = '<tr><td colspan="2">—</td></tr>'; }
  } catch (e) {}
}

/* ── 流程面板 ── */
async function 刷新流程() {
  try {
    const f = await api("/api/v1/flow/status");
    $("流程条").innerHTML = f.节点.map((n, i) =>
      `<div class="step ${状态色[n.状态]}" title="${n.详情 || ""}"><b>${n.节点}</b>${n.状态}${n.耗时 ? ` · ${n.耗时}s` : ""}</div>` +
      (i < f.节点.length - 1 ? `<div class="arrow">→</div>` : "")
    ).join("");
    $("流程JSON").textContent = JSON.stringify(f, null, 2);
  } catch (e) {}
}

async function 刷新记录() {
  try {
    const r = await api("/api/v1/records");
    const tbody = $("记录表").querySelector("tbody");
    tbody.innerHTML = r.记录列表.map(x => `<tr>
      <td>${x.时间戳 || "—"}</td><td>${x.模型 || "—"}</td><td>${x.量化 || "—"}</td>
      <td>${x.成功率 != null ? (x.成功率 * 100).toFixed(1) + "%" : "—"}</td>
      <td>${x.汇总均值 ? x.汇总均值.平均熵 : "—"}</td><td>${x.汇总均值 ? x.汇总均值.重复率 : "—"}</td>
      <td>${x.汇总均值 ? x.汇总均值.情感命中率 : "—"}</td><td>${x.汇总均值 ? x.汇总均值.平均耗时 : "—"}</td>
    </tr>`).join("") || '<tr><td colspan="8" class="muted">暂无运行记录</td></tr>';
  } catch (e) {}
}

/* ── 运行时开关 ── */
const 开关定义 = [
  { 名称: "API", 键: "启用API", 说明: "对外 OpenAI 兼容接口" },
  { 名称: "RAG", 键: "启用RAG", 说明: "向量检索注入" },
  { 名称: "LoRA", 键: "启用LoRA", 说明: "适配器外挂" },
  { 名称: "记忆", 键: "启用记忆", 说明: "超长期记忆系统" },
  { 名称: "策略", 键: "动态策略", 说明: "A/B/C" },
];

async function 渲染开关() {
  try {
    const s = await api("/api/v1/switches");
    $("开关区").innerHTML = 开关定义.map(d => {
      const v = s[d.键];
      const 显示 = d.名称 === "策略" ? (v || "跟随") : (v === null ? "跟随" : (v ? "开" : "关"));
      const on = d.名称 === "策略" ? !!v : (v === true);
      return `<div class="switch-card ${on ? "on" : ""}" data-名称="${d.名称}" title="${d.说明}">
        <span class="s-name">${d.名称}</span><span class="s-state">${显示}</span></div>`;
    }).join("");
    $("开关区").querySelectorAll(".switch-card").forEach(el => {
      el.addEventListener("click", async () => {
        const 名称 = el.dataset.名称;
        const s = await api("/api/v1/switches");
        let 值;
        if (名称 === "策略") {
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

/* ── 记忆系统 ── */
async function 渲染记忆() {
  try {
    const st = await api("/api/v1/memory/status");
    $("记忆状态").innerHTML = `<span class="dot"></span>${st.启用 ? "开" : "关"}`;
    $("记忆状态").className = "status-pill " + (st.启用 ? "ok" : "");
    $("记忆按钮").textContent = st.启用 ? "关闭记忆" : "开启记忆";
    $("记忆统计").textContent = JSON.stringify({
      启用: st.启用, 总条数: st.总条数, 写入数: st.写入数, 检索数: st.检索数, 维度分布: st.维度分布,
    }, null, 2);
    const 列表 = await api("/api/v1/memory/list?数量=20");
    $("记忆列表").innerHTML = (列表.记忆列表 || []).map(m =>
      `<div class="mem"><span class="tag">[${m.情感维度}]</span> <span class="txt">${m.提示词} → ${m.摘要}</span></div>`
    ).join("") || '<div class="muted">记忆库为空</div>';
  } catch (e) {}
}

$("记忆按钮").addEventListener("click", async () => {
  try {
    const st = await api("/api/v1/memory/status");
    await api("/api/v1/memory/activate", { 开启: !st.启用 });
    await 渲染记忆();
  } catch (e) { alert("切换记忆失败：" + e.message); }
});

$("记忆清空").addEventListener("click", async () => {
  if (!confirm("确定清空全部记忆？")) return;
  try {
    await api("/api/v1/memory/clear", {});
    await 渲染记忆();
  } catch (e) { alert("清空失败：" + e.message); }
});

/* ── 参数调整（λ/γ/τ 直观覆盖） ── */
async function 渲染参数() {
  try {
    const p = await api("/api/v1/params");
    $("参数展示").innerHTML = `
      <dt>推荐基准</dt><dd>λ=${p.推荐.λ ?? "—"}　γ=${p.推荐.γ ?? "—"}　τ=${p.推荐.τ ?? "—"}</dd>
      <dt>当前生效</dt><dd>λ=${p.生效.λ}　γ=${p.生效.γ}　τ=${p.生效.τ}</dd>`;
    $("参数编辑").innerHTML = ["λ", "γ", "τ"].map(k => `
      <div class="row mt8">
        <label style="margin:0;width:22px;font-weight:700">${k}</label>
        <input type="number" step="0.01" min="0.01" max="1" id="参数-${k}"
               value="${p.覆盖[k] ?? ""}" placeholder="跟随推荐 ${p.推荐[k] ?? "—"}" style="width:110px">
        <span class="muted small">${p.说明[k]}</span>
      </div>`).join("");
    ["λ", "γ", "τ"].forEach(k => {
      $(`参数-${k}`).addEventListener("change", async () => {
        const v = $(`参数-${k}`).value.trim();
        try {
          await api("/api/v1/params", { 名称: k, 值: v === "" ? null : parseFloat(v) });
          await 渲染参数();
        } catch (e) { alert(`${k} 调整失败：${e.message}`); }
      });
    });
  } catch (e) {}
}
$("参数重置").addEventListener("click", async () => {
  try { await api("/api/v1/params/reset", {}); await 渲染参数(); } catch (e) { alert(e.message); }
});

/* ── 打标二次复查 ── */
let 复查条目 = [];
const 复查维度 = ["开心", "悲伤", "愤怒", "中性", "复杂混合", "待定"];
const 复查状态 = ["待标注", "已标注", "返工"];

async function 刷新复查批次() {
  try {
    const t = await api("/api/v1/label/tasks");
    const 列表 = t.任务列表 || [];
    $("复查批次").innerHTML = 列表.map(x =>
      `<option value="${x.批次名}">${x.批次名}（${x.条目数}条 · 已标注${x.已标注数} · 复查${x["复查进度%"]}%）</option>`
    ).join("") || '<option value="">暂无标注任务</option>';
    await 刷新复查统计();
  } catch (e) {}
}

async function 加载复查批次() {
  const 批次 = $("复查批次").value;
  if (!批次) return;
  try {
    const d = await api(`/api/v1/label/task/${encodeURIComponent(批次)}`);
    复查条目 = (d.条目 || []).map(x => Object.assign({}, x));
    渲染复查表();
    刷新复查统计();
  } catch (e) { alert("加载失败：" + e.message); }
}

function 渲染复查表() {
  const tbody = $("复查表").querySelector("tbody");
  tbody.innerHTML = 复查条目.map((x, i) => `<tr>
    <td>${i + 1}</td>
    <td class="small">${x.提示词 || ""}</td>
    <td class="small">${(x.回复文本 || "").slice(0, 40)}</td>
    <td>${x.平均熵 ?? "—"}</td><td>${x.重复率 ?? "—"}</td>
    <td><select data-字段="情感维度" data-行="${i}">${复查维度.map(d => `<option ${x.情感维度 === d ? "selected" : ""}>${d}</option>`).join("")}</select></td>
    <td><select data-字段="质量评分" data-行="${i}">${[1, 2, 3, 4, 5].map(s => `<option value="${s}" ${Number(x.质量评分) === s ? "selected" : ""}>${s}</option>`).join("")}</select></td>
    <td><select data-字段="标注状态" data-行="${i}">${复查状态.map(s => `<option ${x.标注状态 === s ? "selected" : ""}>${s}</option>`).join("")}</select></td>
    <td><input data-字段="复查备注" data-行="${i}" value="${x.复查备注 || ""}" style="min-width:90px"></td>
  </tr>`).join("") || '<tr><td colspan="9" class="muted">无条目</td></tr>';
  tbody.querySelectorAll("[data-行]").forEach(el => {
    el.addEventListener("change", () => {
      const 行 = Number(el.dataset.行), 字段 = el.dataset.字段;
      if (字段 === "质量评分") 复查条目[行][字段] = Number(el.value);
      else 复查条目[行][字段] = el.value;
    });
  });
}

async function 保存复查() {
  const 批次 = $("复查批次").value;
  if (!批次 || !复查条目.length) return;
  try {
    const r = await api(`/api/v1/label/task/${encodeURIComponent(批次)}`, { 条目列表: 复查条目 });
    alert(`已保存 ${r.条目数} 条`);
    await 刷新复查批次();
  } catch (e) { alert("保存失败：" + e.message); }
}

async function 刷新复查统计() {
  const 批次 = $("复查批次").value;
  if (!批次) return;
  try {
    const s = await api(`/api/v1/label/task/${encodeURIComponent(批次)}/stats`);
    const 全过 = s["待标注数"] === 0 && s["返工数"] === 0;
    $("复查统计").innerHTML = `<div class="banner ${全过 ? "ok" : ""}">待标注 ${s.待标注数} · 已标注 ${s.已标注数} · 返工 ${s.返工数} · 复查进度 ${s["复查进度%"]}%</div>`;
  } catch (e) {}
}

$("复查加载").addEventListener("click", 加载复查批次);
$("复查保存").addEventListener("click", 保存复查);

/* ── 主循环 ── */
(async function 启动() {
  await 刷新模型下拉();
  await 刷新模型();
  await 刷新模型库();
  await 刷新历史();
  await 刷新流程();
  await 刷新记录();
  await 渲染开关();
  await 渲染记忆();
  await 渲染参数();
  await 刷新复查批次();
  setInterval(() => {
    刷新模型();
    刷新测试();
    刷新打标();
    刷新流程();
    渲染开关();
    渲染记忆();
    渲染参数();
  }, 2500);
})();
