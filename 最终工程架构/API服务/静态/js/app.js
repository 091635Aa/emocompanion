/* 简化版控制台逻辑 */
const $ = (id) => document.getElementById(id);
const 状态色 = { "待执行": "", "执行中": "run", "完成": "done", "失败": "err", "跳过": "skip" };

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

/* ── 状态与流程轮询 ── */
let 已加载模型名 = null;

async function 刷新状态() {
  try {
    const st = await api("/api/v1/model/status");
    已加载模型名 = st.已加载模型名;
    $("模型徽章").textContent = st.加载状态 === "已加载"
      ? `已加载: ${st.已加载模型名}` : (st.加载状态 === "加载中" ? "加载中…" : "未加载模型");
    $("模型徽章").className = "badge " + (st.加载状态 === "已加载" ? "ok" : st.加载状态 === "加载中" ? "run" : "");
    const 显存 = st.显存MB, 总显存 = st.总显存MB;
    if (显存 != null && 总显存) {
      $("显存徽章").textContent = `显存 ${显存} / ${总显存} MB`;
      $("显存条").style.width = Math.min(100, 显存 / 总显存 * 100) + "%";
    } else {
      $("显存徽章").textContent = "GPU 未检测";
      $("显存条").style.width = "0%";
    }
    const 加载中 = st.加载状态 === "加载中";
    $("加载状态").innerHTML = `<span class="dot"></span>${加载中 ? "加载中…" : st.加载状态}`;
    $("加载状态").className = "status-pill " + (st.加载状态 === "已加载" ? "ok" : 加载中 ? "run" : st.加载状态 === "失败" ? "err" : "");
    $("加载提示").textContent = st.加载状态 === "失败" ? `加载失败：${st.加载错误 || "未知错误"}` : `点击「加载模型」开始，大模型加载需要一些时间。`;
    $("加载按钮").disabled = 加载中 || !当前选中模型名 || st.加载状态 === "已加载";
    $("卸载按钮").disabled = st.加载状态 !== "已加载";
    $("发送按钮").disabled = st.加载状态 !== "已加载";
    $("测试按钮").disabled = st.加载状态 !== "已加载";
    $("对话模型标签").textContent = st.已加载模型名 || "—";
    if (st.加载状态 === "已加载" && st.模型描述) {
      const 定制 = st.模型描述.类型 === "定制";
      $("打标卡片").style.display = 定制 ? "" : "none";
      $("打标按钮").disabled = false;
      $("流程类型标签").textContent = 定制 ? "定制流程（含打标）" : "标准流程";
    }
    // 健康徽章
    const h = await api("/api/v1/health");
    $("服务徽章").textContent = "服务运行中";
    $("服务徽章").className = "badge ok";
  } catch (e) {
    $("服务徽章").textContent = "服务不可用";
    $("服务徽章").className = "badge err";
  }
}

async function 刷新流程() {
  try {
    const f = await api("/api/v1/flow/status");
    $("流程条").innerHTML = f.节点.map((n, i) =>
      `<div class="step ${状态色[n.状态]}"><b>${n.节点}</b>${n.状态}</div>` +
      (i < f.节点.length - 1 ? `<div class="arrow">→</div>` : "")
    ).join("");
    $("流程条").querySelectorAll(".step").forEach((el, i) => {
      el.title = f.节点[i].详情 || "";
    });
  } catch (e) {}
}

/* ── 模型注册 ── */
let 当前选中模型名 = null;

async function 刷新模型列表() {
  try {
    const 数据 = await api("/api/v1/scan");
    const 列表 = 数据.可用模型;
    if (!列表.length) { $("模型下拉").innerHTML = '<option value="">未发现模型目录</option>'; return; }
    $("模型下拉").innerHTML = 列表.map(m =>
      `<option value="${m.模型名}" data-路径="${m.路径}">${m.模型名} ${m.已注册 ? "(已注册)" : ""} · dim=${m.hidden_size}</option>`
    ).join("");
    当前选中模型名 = 列表[0].模型名;
    $("加载按钮").disabled = false;
  } catch (e) {
    $("模型下拉").innerHTML = '<option value="">扫描失败</option>';
  }
}

$("模型下拉").addEventListener("change", () => { 当前选中模型名 = $("模型下拉").value; });

$("注册按钮").addEventListener("click", async () => {
  const 选项 = $("模型下拉").selectedOptions[0];
  if (!选项) return;
  const 路径 = 选项.dataset.路径;
  $("注册按钮").disabled = true;
  $("注册提示").textContent = "正在生成模型文件并自动匹配参数…";
  try {
    const 描述 = await api("/api/v1/model/register", {
      模型名: 当前选中模型名, 路径, 类型: "标准", 量化: "4bit", 动态策略: "B",
      rag: false, 长上下文: false, 自动测试: true,
    });
    const p = 描述.参数;
    $("注册提示").innerHTML = `已生成模型文件，自动匹配参数：λ=${p.λ} γ=${p.γ} τ=${p.τ}（${p.来源}），归一化基准 ${描述.归一化基准}。可点击「加载模型」。`;
  } catch (e) {
    $("注册提示").textContent = "注册失败：" + e.message;
  }
  $("注册按钮").disabled = false;
  await 刷新流程();
});

/* ── 加载 / 卸载 ── */
$("加载按钮").addEventListener("click", async () => {
  try {
    await api("/api/v1/model/load", { 模型名: 当前选中模型名 });
    $("加载提示").textContent = "开始加载，请稍候…";
    await 刷新状态(); await 刷新流程();
  } catch (e) { $("加载提示").textContent = "启动加载失败：" + e.message; }
});

$("卸载按钮").addEventListener("click", async () => {
  try {
    await api("/api/v1/model/unload");
    $("加载提示").textContent = "已卸载模型，显存已释放。";
    await 刷新状态(); await 刷新流程();
  } catch (e) { $("加载提示").textContent = "卸载失败：" + e.message; }
});

/* ── 测试 ── */
$("测试按钮").addEventListener("click", async () => {
  $("测试按钮").disabled = true;
  $("测试结果").innerHTML = '<div class="banner run">测试进行中…</div>';
  try {
    await api("/api/v1/test/activate", { 范围: "全部", 模型名: 已加载模型名 });
  } catch (e) { $("测试结果").innerHTML = `<div class="banner err">启动测试失败：${e.message}</div>`; }
});

async function 轮询测试() {
  try {
    const t = await api("/api/v1/test/status");
    if (t.运行中) {
      $("测试结果").innerHTML = '<div class="banner run">测试进行中…</div>';
    } else if (t.最近结果) {
      const r = t.最近结果;
      const 全过 = r.失败 === 0;
      $("测试结果").innerHTML =
        `<div class="banner ${全过 ? "ok" : "err"}">${全过 ? "全部测试通过" : `部分失败（通过 ${r.通过} / ${(r.通过 || 0) + (r.失败 || 0)}）`}（${r.总耗时 || "—"}s）</div>` +
        (r.明细 || []).map(d =>
          `<div class="small">${d.通过 ? "通过" : "失败"} · ${d.名称}：${d.说明 || ""}</div>`
        ).join("");
      $("测试按钮").disabled = false;
    }
  } catch (e) {}
}

/* ── 对话 ── */
function 添加消息(角色, 内容, 附加 = "") {
  const 区 = $("对话区");
  const div = document.createElement("div");
  div.className = "msg " + 角色;
  div.textContent = 内容;
  if (附加) {
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = 附加;
    div.appendChild(meta);
  }
  区.appendChild(div);
  区.scrollTop = 区.scrollHeight;
}

$("发送按钮").addEventListener("click", async () => {
  const 提示 = $("对话输入").value.trim();
  if (!提示) return;
  添加消息("user", 提示);
  $("对话输入").value = "";
  $("发送按钮").disabled = true;
  try {
    const 结果 = await api("/api/v1/generate", { 模型名: 已加载模型名, 提示词: 提示, 最大token: 128 });
    const 细腻度 = 结果.平均熵.toFixed(3);
    const 重复 = 结果.重复率.toFixed(3);
    添加消息("bot", 结果.文本, `细腻度 ${细腻度} · 重复度 ${重复} · 情感命中 ${结果.情感命中率.toFixed(3)} · λ=${结果.λ} γ=${结果.γ} τ=${结果.τ}`);
  } catch (e) {
    添加消息("bot", "生成失败：" + e.message);
  }
  $("发送按钮").disabled = 已加载模型名 ? false : true;
  await 刷新流程();
});

$("对话输入").addEventListener("keydown", (e) => { if (e.key === "Enter") $("发送按钮").click(); });

/* ── 打标 ── */
$("打标按钮").addEventListener("click", async () => {
  try {
    await api("/api/v1/label/run", { 模型名: 已加载模型名, 批次名: $("批次名").value || "批次1", 最大token: 128 });
    $("打标提示").textContent = "打标进行中…";
  } catch (e) { $("打标提示").textContent = "打标启动失败：" + e.message; }
});

async function 轮询打标() {
  try {
    const t = await api("/api/v1/label/status");
    if (t.运行中) {
      const p = t.总数 ? Math.round(t.进度 / t.总数 * 100) : 0;
      $("打标进度").style.width = p + "%";
      $("打标提示").textContent = `打标中 ${t.进度}/${t.总数}：${t.当前提示 || ""}`;
    } else if (t.结果) {
      const r = t.结果;
      if (r.状态 === "完成") {
        $("打标进度").style.width = "100%";
        $("打标提示").innerHTML = `打标完成 → <a href="/pro" target="_blank">查看标注文件</a>（f:\\打标）`;
      } else {
        $("打标提示").textContent = "打标失败：" + (r.错误 || "未知错误");
      }
    }
  } catch (e) {}
}

/* ── 开关（RAG / LoRA / 记忆） ── */
const 开关键 = { 开关RAG: ["RAG", "启用RAG", "RAG"], 开关LoRA: ["LoRA", "启用LoRA", "LoRA"],
                 开关记忆: ["记忆", "启用记忆", "记忆"] };

async function 刷新开关() {
  try {
    const s = await api("/api/v1/switches");
    for (const [id, [, 键, 显示]] of Object.entries(开关键)) {
      const 值 = s[键];
      const 文本 = 值 === null ? `${显示} 跟随` : (值 ? `${显示} 开` : `${显示} 关`);
      const 按钮 = $(id);
      按钮.textContent = 文本;
      按钮.disabled = 已加载模型名 === null;
    }
  } catch (e) {}
}

for (const [id, [名称]] of Object.entries(开关键)) {
  $(id).addEventListener("click", async () => {
    try {
      const s = await api("/api/v1/switches");
      const 键 = 开关键[id][1];
      const 值 = !(s[键] === true);
      await api("/api/v1/switch", { 名称, 值 });
      await 刷新开关();
    } catch (e) { alert("开关切换失败：" + e.message); }
  });
}

/* ── 主循环 ── */
(async function 启动() {
  await 刷新模型列表();
  await 刷新状态();
  await 刷新流程();
  await 刷新开关();
  setInterval(async () => {
    await 刷新状态();
    await 刷新流程();
    await 轮询测试();
    await 轮询打标();
    await 刷新开关();
  }, 2500);
})();
