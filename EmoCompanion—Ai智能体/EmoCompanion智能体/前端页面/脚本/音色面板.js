/* ============================================================
   EmoCompanion智能体 V2 —— 音色面板
   -----------------------------------------------------------------
   在 #音色区 内动态构建：
     标题 / 当前模型选择行（动态克隆 #合成模型 的选项）/
     当前音色徽标（读取 localStorage["当前音色名"]）/
     刷新按钮 / 定制新音色按钮（触发隐藏的多选音频文件输入）/
     内置音色列表（kind==="system"）/ 定制音色列表（其余 kind）。
   暴露全局 API：
     window.音色面板.初始化()
     window.音色面板.刷新()
   依赖（防御式 typeof 调用）：
     window.请求.获取 / 提交 / 上传 / 删除
     window.对话界面.添加消息(类型, 内容, 元信息)
   ============================================================ */
(function () {
  "use strict";

  /* ---------------- 防御式依赖引用 ---------------- */
  const 请求 = (typeof window.请求 !== "undefined" && window.请求) || null;
  const 对话界面 = (typeof window.对话界面 !== "undefined" && window.对话界面) || null;

  /* ---------------- 常量 ---------------- */
  const 默认模型 = "qwen-audio-3.0-tts-plus";   // #合成模型 缺失时的兜底模型
  const 试听文本 = "你好，我是EmoCompanion。很高兴在这里遇见你。";
  const 当前音色ID键 = "当前音色ID";             // localStorage 键：当前音色 id
  const 当前音色名键 = "当前音色名";             // localStorage 键：当前音色显示名
  /* kind -> 中文徽标文字（内置 / 复刻 / 在线） */
  const kind中文表 = {
    system: "内置",
    local_clone: "复刻",
    clone: "在线",
    dashscope: "在线",
  };

  /* ---------------- 运行时状态 ---------------- */
  let 已初始化 = false;   // 防止 主控.js 与文件末尾兜底重复初始化
  let 定制中 = false;     // 定制上传期间锁定按钮
  let 试听中按钮 = null;  // 正在试听的按钮（结束后恢复）

  /* ---------------- 工具 ---------------- */
  /* 添加系统/提示消息：优先走 对话界面，缺失时回退 console */
  function 显示消息(类型, 内容) {
    if (对话界面 && typeof 对话界面.添加消息 === "function") {
      try {
        对话界面.添加消息(类型, 内容);
        return;
      } catch (忽略) { /* 落到 console 兜底 */ }
    }
    console.log("[" + (类型 || "系统") + "]", 内容);
  }

  /* 读取当前模型：优先本面板下拉，其次 #合成模型，最后默认值 */
  function 获取当前模型() {
    const 面板下拉 = document.getElementById("音色-模型");
    if (面板下拉 && 面板下拉.value) return 面板下拉.value;
    const 合成下拉 = document.getElementById("合成模型");
    if (合成下拉 && 合成下拉.value) return 合成下拉.value;
    return 默认模型;
  }

  /* 把 #合成模型 的选项克隆进面板下拉；无源选项时补默认模型 */
  function 同步模型下拉() {
    const 下拉 = document.getElementById("音色-模型");
    if (!下拉) return;
    const 源下拉 = document.getElementById("合成模型");
    const 原选中 = 下拉.value;
    下拉.innerHTML = "";
    let 有选项 = false;
    if (源下拉 && 源下拉.options.length) {
      for (const 选项 of 源下拉.options) {
        const 新选项 = document.createElement("option");
        新选项.value = 选项.value;
        新选项.textContent = 选项.textContent || 选项.value;
        下拉.appendChild(新选项);
      }
      有选项 = true;
      /* 保留用户在本下拉的选择；未选过则跟随合成区的当前模型 */
      const 可保留 = Array.prototype.some.call(下拉.options, (选项) => 选项.value === 原选中);
      下拉.value = (可保留 && 原选中) ? 原选中 : (源下拉.value || 下拉.options[0].value);
    }
    if (!有选项) {
      const 新选项 = document.createElement("option");
      新选项.value = 默认模型;
      新选项.textContent = 默认模型;
      下拉.appendChild(新选项);
      下拉.value = 默认模型;
    }
  }

  /* 更新"当前音色"徽标文字（localStorage 无记录时显示 未选择） */
  function 更新当前音色徽标() {
    const 徽标 = document.getElementById("音色-当前名");
    if (!徽标) return;
    徽标.textContent = localStorage.getItem(当前音色名键) || "未选择";
  }

  /* 高亮当前音色卡片（与 localStorage 中的当前音色ID 比对） */
  function 高亮当前音色() {
    const 音色区 = document.getElementById("音色区");
    if (!音色区) return;
    const 当前ID = localStorage.getItem(当前音色ID键);
    音色区.querySelectorAll(".音色卡片").forEach((卡片) => {
      卡片.classList.toggle("当前", 卡片.dataset.id === 当前ID);
    });
  }

  /* ---------------- UI 构建 ---------------- */
  /* 注入本模块自带样式（不修改 页面样式.css，仅本文件自持） */
  function 注入样式() {
    if (document.getElementById("音色面板样式")) return;
    const 样式 = document.createElement("style");
    样式.id = "音色面板样式";
    样式.textContent = `
      #音色区 .音色面板 { display:flex; flex-direction:column; gap:12px; padding:2px 0 14px; }
      #音色区 .音色工具栏 { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
      #音色区 .音色模型行 { display:flex; align-items:center; gap:6px; font-size:12px; color:var(--次要); }
      #音色区 .音色模型行 select {
        background:var(--卡片2); border:1px solid var(--分隔线); color:var(--正文);
        border-radius:8px; padding:6px 8px; font-family:inherit; outline:none;
      }
      #音色区 .音色模型行 select:focus { border-color:var(--主色); }
      #音色区 .音色徽标 { font-size:13px; color:var(--次要); }
      #音色区 .音色徽标 b { color:var(--主色); font-weight:600; }
      #音色区 .音色分组 { background:var(--卡片); border:1px solid var(--分隔线); border-radius:12px; padding:12px 14px; }
      #音色区 .音色分组-标题 { font-size:14px; color:var(--正文); margin-bottom:10px; }
      #音色区 .音色列表 { display:grid; grid-template-columns:repeat(auto-fill,minmax(250px,1fr)); gap:10px; }
      #音色区 .音色分组-空 { color:var(--次要); font-size:12px; padding:4px 0; }
      #音色区 .音色卡片 {
        background:var(--卡片2); border:1px solid var(--分隔线); border-radius:10px;
        padding:10px 12px; display:flex; flex-direction:column; gap:8px; transition:.15s;
      }
      #音色区 .音色卡片.当前 { border-color:var(--主色); box-shadow:0 0 8px rgba(124,140,255,.35); }
      #音色区 .音色卡片-头 { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
      #音色区 .音色卡片-名 { font-weight:600; word-break:break-all; }
      #音色区 .音色徽章 { font-size:11px; padding:1px 8px; border-radius:999px; background:var(--分隔线); color:var(--次要); white-space:nowrap; }
      #音色区 .音色徽章.kind-system { background:rgba(87,214,179,.16); color:var(--辅色); }
      #音色区 .音色徽章.kind-local_clone { background:rgba(255,200,87,.16); color:var(--警告); }
      #音色区 .音色徽章.kind-clone, #音色区 .音色徽章.kind-dashscope { background:rgba(124,140,255,.16); color:var(--主色); }
      #音色区 .音色语言 { font-size:11px; color:var(--次要); margin-left:auto; white-space:nowrap; }
      #音色区 .音色卡片-操作 { display:flex; gap:6px; flex-wrap:wrap; }
      #音色区 .音色卡片-操作 button { padding:4px 10px; font-size:12px; }
    `;
    document.head.appendChild(样式);
  }

  /* 构建面板骨架；已构建过则直接复用（刷新只重建列表） */
  function 构建面板() {
    const 音色区 = document.getElementById("音色区");
    if (!音色区) return null;
    注入样式();
    if (音色区.querySelector(".音色面板")) return 音色区;
    音色区.innerHTML = `
      <div class="音色面板">
        <h2>音色选择与定制</h2>
        <div class="音色工具栏">
          <label class="音色模型行">当前模型
            <select id="音色-模型"></select>
          </label>
          <span class="音色徽标">当前音色：<b id="音色-当前名">未选择</b></span>
          <button id="音色-刷新" class="次要按钮" type="button">刷新</button>
          <button id="音色-定制" class="主要按钮" type="button">定制新音色</button>
          <input id="音色-文件" type="file" accept="audio/*" multiple hidden>
        </div>
        <div id="内置音色列表" class="音色分组">
          <h3 class="音色分组-标题">内置音色（0）</h3>
          <div class="音色列表"></div>
        </div>
        <div id="定制音色列表" class="音色分组">
          <h3 class="音色分组-标题">定制音色（0）</h3>
          <div class="音色列表"></div>
        </div>
      </div>`;
    return 音色区;
  }

  /* 创建单个音色卡片：名称（当前音色高亮）+ kind 徽标 + 语言 + 操作按钮 */
  function 创建卡片(音色, 是当前) {
    const 卡片 = document.createElement("div");
    卡片.className = "音色卡片" + (是当前 ? " 当前" : "");
    卡片.dataset.id = 音色.id;

    const 头 = document.createElement("div");
    头.className = "音色卡片-头";
    const 名 = document.createElement("span");
    名.className = "音色卡片-名";
    名.textContent = 音色.name || 音色.id || "未命名";
    const 徽章 = document.createElement("span");
    徽章.className = "音色徽章 kind-" + (音色.kind || "other");
    徽章.textContent = kind中文表[音色.kind] || 音色.kind || "其他";
    头.appendChild(名);
    头.appendChild(徽章);
    if (音色.lang) {
      const 语言 = document.createElement("span");
      语言.className = "音色语言";
      语言.textContent = 音色.lang;
      头.appendChild(语言);
    }

    const 操作 = document.createElement("div");
    操作.className = "音色卡片-操作";
    const 动作们 = [
      ["选择", "选择", true],
      ["试听", "试听", true],
      ["重命名", "重命名", true],
      ["删除", "删除", 音色.kind !== "system"],   // 仅非 system 显示删除
    ];
    for (const [动作, 文字, 显示] of 动作们) {
      if (!显示) continue;
      const 按钮 = document.createElement("button");
      按钮.type = "button";
      按钮.className = 动作 === "删除" ? "危险按钮" : "次要按钮";
      按钮.dataset.动作 = 动作;
      按钮.textContent = 文字;
      操作.appendChild(按钮);
    }

    卡片.appendChild(头);
    卡片.appendChild(操作);
    return 卡片;
  }

  /* 清空分组的音色列表（保留标题），返回列表容器 */
  function 重置分组列表(分组) {
    const 列表 = 分组.querySelector(".音色列表") || 分组;
    while (列表.firstChild) 列表.removeChild(列表.firstChild);
    return 列表;
  }

  function 空提示节点() {
    const 节点 = document.createElement("div");
    节点.className = "音色分组-空";
    节点.textContent = "暂无音色";
    return 节点;
  }

  /* ---------------- 刷新：拉取并渲染音色 ---------------- */
  async function 刷新() {
    const 内置分组 = document.getElementById("内置音色列表");
    const 定制分组 = document.getElementById("定制音色列表");
    if (!内置分组 || !定制分组) return;
    if (!请求) {
      显示消息("系统", "请求模块尚未加载，无法刷新音色列表");
      return;
    }

    const 刷新按钮 = document.getElementById("音色-刷新");
    同步模型下拉();       // 跟随合成区最新的模型选项
    更新当前音色徽标();
    if (刷新按钮) {
      刷新按钮.disabled = true;
      刷新按钮.textContent = "刷新中…";
    }

    const 模型 = 获取当前模型();
    try {
      const 数据 = await 请求.获取("/api/音色?model=" + encodeURIComponent(模型));
      const 音色们 = (数据 && Array.isArray(数据.voices)) ? 数据.voices : [];
      const 当前ID = localStorage.getItem(当前音色ID键);

      const 内置列表 = 重置分组列表(内置分组);
      const 定制列表 = 重置分组列表(定制分组);
      let 内置数 = 0, 定制数 = 0;
      for (const 音色 of 音色们) {
        if (!音色 || !音色.id) continue;
        const 是当前 = 音色.id === 当前ID;
        if (音色.kind === "system") {
          内置列表.appendChild(创建卡片(音色, 是当前));
          内置数++;
        } else {
          定制列表.appendChild(创建卡片(音色, 是当前));
          定制数++;
        }
      }
      if (!内置数) 内置列表.appendChild(空提示节点());
      if (!定制数) 定制列表.appendChild(空提示节点());

      const 内置标题 = 内置分组.querySelector(".音色分组-标题");
      if (内置标题) 内置标题.textContent = "内置音色（" + 内置数 + "）";
      const 定制标题 = 定制分组.querySelector(".音色分组-标题");
      if (定制标题) 定制标题.textContent = "定制音色（" + 定制数 + "）";
    } catch (错误) {
      显示消息("系统", "音色列表加载失败：" + (错误.message || 错误));
    } finally {
      if (刷新按钮) {
        刷新按钮.disabled = false;
        刷新按钮.textContent = "刷新";
      }
    }
  }

  /* ---------------- 操作逻辑 ---------------- */
  /* 选择：写入 localStorage，更新徽标与高亮 */
  function 选择音色(音色ID, 音色名) {
    localStorage.setItem(当前音色ID键, 音色ID);
    localStorage.setItem(当前音色名键, 音色名);
    更新当前音色徽标();
    高亮当前音色();
    显示消息("系统", "已选择音色：" + 音色名);
  }

  /* 试听：调用 /api/合成 生成一句试听音频并播放（失败给友好提示） */
  async function 试听(按钮, 音色ID) {
    if (!请求) {
      显示消息("系统", "请求模块尚未加载，无法试听");
      return;
    }
    if (试听中按钮) return;   // 上一次试听尚未结束
    试听中按钮 = 按钮;
    按钮.disabled = true;
    按钮.textContent = "试听中…";
    try {
      const 结果 = await 请求.提交("/api/合成", {
        model: 获取当前模型(),
        voice: 音色ID,
        text: 试听文本,
        format: "wav",
        sample_rate: 24000,
      });
      const 地址 = 结果 && 结果.audio_url;
      if (!地址) throw new Error("接口未返回 audio_url");
      /* 优先挂到 #合成音频 元素播放（顺带可见），元素缺失时用 new Audio 播放 */
      const 播放器 = document.getElementById("合成音频");
      if (播放器) {
        播放器.src = 地址;
        播放器.play().catch(() => { /* 浏览器自动播放策略拦截时忽略 */ });
      } else {
        const 音频 = new Audio(地址);
        音频.play().catch(() => { /* 忽略自动播放拦截 */ });
      }
      if (结果.warning) 显示消息("系统", 结果.warning);
    } catch (错误) {
      const 消息 = String((错误 && 错误.message) || 错误 || "");
      /* 502（密钥/配额/业务空间等）给友好提示，其余透传原始信息 */
      const 疑似限制 = /502|密钥|api.?key|配额|限额|限流|账号|权限|业务空间/i.test(消息);
      显示消息("系统", 疑似限制 ? "试听失败（可能密钥/账号限制）" : "试听失败：" + 消息);
    } finally {
      按钮.disabled = false;
      按钮.textContent = "试听";
      试听中按钮 = null;
    }
  }

  /* 重命名：prompt 输入新别名，提交成功后刷新列表 */
  async function 重命名(音色ID, 原名) {
    if (!请求) {
      显示消息("系统", "请求模块尚未加载，无法重命名");
      return;
    }
    const 输入 = prompt("输入新的显示名称", 原名);
    if (输入 === null) return;   // 用户取消
    const 新名 = String(输入).trim();
    if (!新名) {
      显示消息("系统", "名称不能为空，已取消重命名");
      return;
    }
    if (新名 === 原名) return;
    try {
      await 请求.提交("/api/音色/重命名", { voice_id: 音色ID, alias: 新名 });
      /* 重命名的是当前音色时，同步更新徽标中的名字 */
      if (localStorage.getItem(当前音色ID键) === 音色ID) {
        localStorage.setItem(当前音色名键, 新名);
        更新当前音色徽标();
      }
      显示消息("系统", "已重命名：" + 原名 + " → " + 新名);
      刷新();
    } catch (错误) {
      显示消息("系统", "重命名失败：" + (错误.message || 错误));
    }
  }

  /* 删除：confirm 确认后调用 DELETE，成功后刷新（仅非 system 卡片有删除按钮） */
  async function 删除音色(音色ID, 音色名) {
    if (!请求) {
      显示消息("系统", "请求模块尚未加载，无法删除");
      return;
    }
    if (!confirm("确定删除音色「" + 音色名 + "」吗？此操作不可恢复。")) return;
    try {
      await 请求.删除("/api/定制音色/删除", { 音色ID: 音色ID });
      /* 删除的是当前音色时，清空本地选择 */
      if (localStorage.getItem(当前音色ID键) === 音色ID) {
        localStorage.removeItem(当前音色ID键);
        localStorage.removeItem(当前音色名键);
        更新当前音色徽标();
      }
      显示消息("系统", "已删除音色：" + 音色名);
      刷新();
    } catch (错误) {
      显示消息("系统", "删除失败：" + (错误.message || 错误));
    }
  }

  /* 定制：把选中的音频文件打包为 FormData 上传（按钮锁定直至完成） */
  async function 定制新音色(文件列表) {
    if (定制中) return;
    if (!请求) {
      显示消息("系统", "请求模块尚未加载，无法定制音色");
      return;
    }
    const 文件们 = Array.prototype.filter.call(文件列表 || [], (文件) => 文件 && 文件.size > 0);
    if (!文件们.length) {
      显示消息("系统", "未选择有效的音频文件，已取消定制");
      return;
    }
    const 输入 = prompt("定制音色名称/前缀", "emocompanion");
    if (输入 === null) return;   // 用户取消
    const 名称 = String(输入).trim();
    if (!名称) {
      显示消息("系统", "名称不能为空，已取消定制");
      return;
    }

    const 按钮 = document.getElementById("音色-定制");
    if (按钮) {
      定制中 = true;
      按钮.disabled = true;
      按钮.textContent = "定制中…（需数十秒）";
    }
    try {
      const 表单 = new FormData();
      文件们.forEach((文件) => 表单.append("文件们", 文件));
      表单.append("名称", 名称);
      const 结果 = await 请求.上传("/api/定制音色", 表单);
      处理定制结果(结果);
    } catch (错误) {
      显示消息("系统", "定制请求失败：" + (错误.message || 错误));
    } finally {
      if (按钮) {
        定制中 = false;
        按钮.disabled = false;
        按钮.textContent = "定制新音色";
      }
    }
  }

  /* 从 流程结果 各步骤中提取失败信息（ok=false 时展示失败步骤的错误） */
  function 提取流程错误(流程) {
    const 片段们 = [];
    for (const 键 of Object.keys(流程 || {})) {
      const 步骤 = 流程[键];
      if (!步骤 || typeof 步骤 !== "object") continue;
      const 错误 = 步骤.error || 步骤.错误 || 步骤.message || 步骤.msg || 步骤.detail;
      if (错误) 片段们.push(键 + "：" + 错误);
    }
    return 片段们.join("；");
  }

  /* 汇总选择报告为 "评分 X.XX · 片段 Y.Ys" 摘要（score/duration_s 可选） */
  function 选择报告摘要(报告) {
    const 报告对象 = 报告 || {};
    const 评分 = 报告对象.score;
    let 时长 = 报告对象.duration_s;
    if (时长 == null && 报告对象.end_s != null && 报告对象.start_s != null) {
      时长 = 报告对象.end_s - 报告对象.start_s;   // 兼容只有起止时间的报告
    }
    const 片段们 = [];
    if (评分 != null && !isNaN(评分)) 片段们.push("评分 " + Number(评分).toFixed(2));
    if (时长 != null && !isNaN(时长)) 片段们.push("片段 " + Number(时长).toFixed(1) + "s");
    return 片段们.join(" · ");
  }

  /* 处理 /api/定制音色 的响应：成功展示音色ID/评分/时长，失败展示流程错误 */
  function 处理定制结果(结果) {
    if (!结果 || typeof 结果 !== "object") {
      显示消息("系统", "定制音色：接口未返回有效结果");
      return;
    }
    if (结果.ok) {
      const 摘要 = 选择报告摘要(结果.选择报告);
      显示消息("系统", "定制完成！音色ID：" + (结果.音色ID || "未知") + (摘要 ? "，" + 摘要 : ""));
    } else {
      const 错误文本 = 提取流程错误(结果.流程结果);
      显示消息("系统", 错误文本 ? "定制未完成：" + 错误文本 : "定制未完成（后端未提供详细原因）");
    }
    刷新();   // 无论成败都刷新列表，反映最新状态
  }

  /* ---------------- 事件绑定（卡片操作用事件委托） ---------------- */
  function 绑定事件() {
    const 音色区 = document.getElementById("音色区");
    if (!音色区) return;

    const 刷新按钮 = document.getElementById("音色-刷新");
    if (刷新按钮) 刷新按钮.addEventListener("click", 刷新);

    const 模型下拉 = document.getElementById("音色-模型");
    if (模型下拉) 模型下拉.addEventListener("change", 刷新);

    const 定制按钮 = document.getElementById("音色-定制");
    const 文件输入 = document.getElementById("音色-文件");
    if (定制按钮 && 文件输入) {
      定制按钮.addEventListener("click", () => {
        if (定制中) return;
        文件输入.value = "";   // 允许重复选择同一批文件
        文件输入.click();
      });
      文件输入.addEventListener("change", () => {
        if (文件输入.files && 文件输入.files.length) 定制新音色(文件输入.files);
      });
    }

    音色区.addEventListener("click", (事件) => {
      const 按钮 = 事件.target && 事件.target.closest ? 事件.target.closest("button[data-动作]") : null;
      if (!按钮) return;
      const 卡片 = 按钮.closest(".音色卡片");
      if (!卡片) return;
      const 动作 = 按钮.dataset.动作;
      const 音色ID = 卡片.dataset.id;
      const 名节点 = 卡片.querySelector(".音色卡片-名");
      const 音色名 = 名节点 ? 名节点.textContent : 音色ID;
      if (动作 === "选择") 选择音色(音色ID, 音色名);
      else if (动作 === "试听") 试听(按钮, 音色ID);
      else if (动作 === "重命名") 重命名(音色ID, 音色名);
      else if (动作 === "删除") 删除音色(音色ID, 音色名);
    });
  }

  /* ---------------- 对外 API ---------------- */
  function 初始化() {
    if (已初始化) return;
    if (!document.getElementById("音色区")) return;   // 防御式：容器缺失则跳过
    构建面板();
    同步模型下拉();
    更新当前音色徽标();
    绑定事件();
    已初始化 = true;
    刷新();
  }

  window.音色面板 = { 初始化: 初始化, 刷新: 刷新 };

  /* 页面加载后由 主控.js 防御式调用 初始化；
     若主控未调用（如单独调试本页面），这里兜底自行初始化（标志避免重复）。 */
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      if (!已初始化) 初始化();
    });
  } else {
    初始化();
  }
})();
