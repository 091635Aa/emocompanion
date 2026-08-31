/* ============================================================
   缘圆智能体 V2 —— 主控
   DOMContentLoaded 后：tab 切换 / 引导数据 / 合成表单 /
   对话发送（含斜杠命令）/ 防御性初始化并行模块。
   ============================================================ */
(function () {
  "use strict";

  /* ---------------- tab 切换 ---------------- */
  function 切换标签(标签) {
    document.querySelectorAll(".标签页按钮").forEach(按钮 => {
      按钮.classList.toggle("激活", 按钮.dataset.标签 === 标签);
    });
    document.querySelectorAll(".功能面板").forEach(面板 => {
      面板.classList.toggle("激活", 面板.id === `${标签}区`);
    });
  }

  function 绑定标签切换() {
    document.querySelectorAll(".标签页按钮").forEach(按钮 => {
      按钮.addEventListener("click", () => 切换标签(按钮.dataset.标签));
    });
  }

  /* ---------------- 引导数据 ---------------- */
  function 填充模型下拉(模型们) {
    const 下拉 = document.getElementById("合成模型");
    if (!下拉) return;
    下拉.innerHTML = "";
    for (const 模型 of 模型们 || []) {
      const 选项 = document.createElement("option");
      选项.value = 模型.id;
      选项.textContent = 模型.name || 模型.id;
      下拉.appendChild(选项);
    }
  }

  function 回填最近使用(最近) {
    if (!最近 || !最近.model) return;
    const 下拉 = document.getElementById("合成模型");
    if (!下拉) return;
    for (const 选项 of 下拉.options) {
      if (选项.value === 最近.model) {
        下拉.value = 最近.model;
        break;
      }
    }
  }

  async function 加载引导数据() {
    try {
      const 数据 = await window.请求.获取("/api/引导数据");
      填充模型下拉(数据.models);
      回填最近使用(数据.last_used);
      渲染标签区(数据.tags && 数据.tags.control, 数据.tags && 数据.tags.rich);
      填充场景预设(数据.scene_presets);
      const 模型 = document.getElementById("合成模型");
      if (模型) {
        await 加载合成音色(模型.value || "");
        if (数据.last_used && 数据.last_used.voice) {
          const 音色下拉 = document.getElementById("合成音色");
          if (音色下拉 && 音色下拉.value !== 数据.last_used.voice) 音色下拉.value = 数据.last_used.voice;
        }
      }
    } catch (错误) {
      window.对话界面.添加消息("系统", `引导数据加载失败：${错误.message}`);
    }
    window.状态指示.设置状态("空闲");
  }

  /* 填充合成音色下拉（跟随模型） */
  async function 加载合成音色(模型) {
    const 下拉 = document.getElementById("合成音色");
    if (!下拉) return;
    下拉.innerHTML = "";
    try {
      const 数据 = await window.请求.获取("/api/音色?model=" + encodeURIComponent(模型 || ""));
      const 音色们 = (数据 && Array.isArray(数据.voices)) ? 数据.voices : [];
      for (const 音色 of 音色们) {
        if (!音色 || !音色.id) continue;
        const 选项 = document.createElement("option");
        选项.value = 音色.id;
        选项.textContent = 音色.name || 音色.id;
        下拉.appendChild(选项);
      }
    } catch (错误) { /* 音色加载失败静默 */ }
  }

  /* 渲染情感/效果标签区：点击插入 [标签] 到文本光标处 */
  function 渲染标签区(控制们, 富们) {
    const 区 = document.getElementById("合成标签区");
    if (!区) return;
    区.innerHTML = "";
    const 插入 = (标签) => {
      const 文本域 = document.getElementById("合成文本");
      if (!文本域) return;
      const 起 = 文本域.selectionStart == null ? 文本域.value.length : 文本域.selectionStart;
      const 末 = 文本域.selectionEnd == null ? 文本域.value.length : 文本域.selectionEnd;
      const 插入文本 = "[" + 标签 + "]";
      文本域.value = 文本域.value.slice(0, 起) + 插入文本 + 文本域.value.slice(末);
      文本域.focus();
      const 新位置 = 起 + 插入文本.length;
      文本域.setSelectionRange(新位置, 新位置);
    };
    const 分组 = (条目们) => {
      for (const 条目 of 条目们 || []) {
        const 英文 = Array.isArray(条目) ? 条目[0] : 条目;
        const 中文 = Array.isArray(条目) ? 条目[1] : 英文;
        const 按钮 = document.createElement("button");
        按钮.type = "button";
        按钮.className = "标签按钮";
        按钮.textContent = 中文;
        按钮.title = "插入 [" + 英文 + "]";
        按钮.addEventListener("click", () => 插入(英文));
        区.appendChild(按钮);
      }
    };
    分组(控制们);   // 情绪控制标签
    分组(富们);     // 拟声富语言标签
  }

  /* 填充场景预设下拉：选中后把维度转成指令填入 */
  function 填充场景预设(预设们) {
    const 下拉 = document.getElementById("合成预设");
    if (!下拉) return;
    下拉.innerHTML = '<option value="">场景预设…</option>';
    for (const 预设 of 预设们 || []) {
      const 选项 = document.createElement("option");
      选项.dataset.dims = JSON.stringify(预设.dims || {});
      选项.textContent = 预设.name || "";
      下拉.appendChild(选项);
    }
    下拉.addEventListener("change", () => {
      const 选项 = 下拉.options[下拉.selectedIndex];
      if (!选项 || !选项.dataset.dims) return;
      let dims = {};
      try { dims = JSON.parse(选项.dataset.dims || "{}"); } catch (忽略) { dims = {}; }
      const 指令 = Object.values(dims).filter(Boolean).join("，");
      const 指令输入 = document.getElementById("合成指令");
      if (指令输入 && 指令) 指令输入.value = 指令;
    });
  }

  /* ---------------- 合成表单 ---------------- */
  async function 执行合成() {
    const 文本输入 = document.getElementById("合成文本");
    const 指令输入 = document.getElementById("合成指令");
    const 模型下拉 = document.getElementById("合成模型");
    const 音色下拉 = document.getElementById("合成音色");
    const 按钮 = document.getElementById("合成按钮");
    const 播放器 = document.getElementById("合成音频");
    if (!文本输入 || !按钮) return;

    const 文本 = 文本输入.value.trim();
    if (!文本) {
      window.对话界面.添加消息("系统", "请先输入要合成的文本");
      return;
    }

    const 载荷 = { text: 文本 };
    if (模型下拉 && 模型下拉.value) 载荷.model = 模型下拉.value;
    if (音色下拉 && 音色下拉.value) 载荷.voice = 音色下拉.value;
    if (指令输入 && 指令输入.value.trim()) 载荷.instruction = 指令输入.value.trim();

    按钮.disabled = true;
    按钮.textContent = "合成中…";
    try {
      const 结果 = await window.请求.提交("/api/合成", 载荷);
      if (结果.audio_url) {
        if (播放器) {
          播放器.src = 结果.audio_url;
          播放器.play().catch(() => { /* 浏览器自动播放策略拦截时忽略 */ });
        }
        window.对话界面.添加消息("音频", 结果.audio_url);
      }
      if (结果.warning) {
        window.对话界面.添加消息("系统", 结果.warning);
      }
    } catch (错误) {
      window.对话界面.添加消息("系统", `合成失败：${错误.message}`);
    } finally {
      按钮.disabled = false;
      按钮.textContent = "合成语音";
    }
  }

  function 绑定合成表单() {
    const 按钮 = document.getElementById("合成按钮");
    if (按钮) 按钮.addEventListener("click", 执行合成);

    /* 模型切换时刷新音色下拉 */
    const 模型下拉 = document.getElementById("合成模型");
    if (模型下拉) {
      模型下拉.addEventListener("change", () => 加载合成音色(模型下拉.value || ""));
    }

    /* AI 自动标注：识别语气并插入标签 */
    const AI标注 = document.getElementById("AI标注按钮");
    if (AI标注) AI标注.addEventListener("click", async () => {
      const 文本 = (document.getElementById("合成文本") || {}).value || "";
      if (!文本.trim()) { window.对话界面.添加消息("系统", "请先输入要标注的文本"); return; }
      AI标注.disabled = true; AI标注.textContent = "标注中…";
      try {
        const 结果 = await window.请求.提交("/api/AI/标注", { text: 文本 });
        if (结果 && 结果.text) {
          const 文本域 = document.getElementById("合成文本");
          if (文本域) 文本域.value = 结果.text;
          window.对话界面.添加消息("系统", "AI 标注完成");
        } else {
          window.对话界面.添加消息("系统", "AI 标注未返回结果");
        }
      } catch (错误) {
        window.对话界面.添加消息("系统", "AI 标注失败：" + ((错误 && 错误.message) || 错误));
      } finally {
        AI标注.disabled = false; AI标注.textContent = "AI 自动标注";
      }
    });

    /* AI 优化指令 */
    const AI优化 = document.getElementById("AI优化按钮");
    if (AI优化) AI优化.addEventListener("click", async () => {
      const 指令 = (document.getElementById("合成指令") || {}).value || "";
      if (!指令.trim()) { window.对话界面.添加消息("系统", "请先生成或填写指令"); return; }
      AI优化.disabled = true; AI优化.textContent = "优化中…";
      try {
        const 结果 = await window.请求.提交("/api/AI/优化指令", { instruction: 指令 });
        if (结果 && 结果.instruction) {
          const 指令输入 = document.getElementById("合成指令");
          if (指令输入) 指令输入.value = 结果.instruction;
          window.对话界面.添加消息("系统", "AI 指令优化完成");
        }
      } catch (错误) {
        window.对话界面.添加消息("系统", "AI 优化失败：" + ((错误 && 错误.message) || 错误));
      } finally {
        AI优化.disabled = false; AI优化.textContent = "AI 优化指令";
      }
    });

    /* 余额探测 */
    const 余额按钮 = document.getElementById("余额探测按钮");
    if (余额按钮) 余额按钮.addEventListener("click", async () => {
      余额按钮.disabled = true; 余额按钮.textContent = "探测中…";
      const 提示 = document.getElementById("合成提示");
      try {
        const 结果 = await window.请求.获取("/api/余额/阿里云/探测");
        if (提示) 提示.textContent = (结果 && 结果.message) || "探测完成";
      } catch (错误) {
        if (提示) 提示.textContent = "余额探测失败：" + ((错误 && 错误.message) || 错误);
      } finally {
        余额按钮.disabled = false; 余额按钮.textContent = "余额探测";
      }
    });

    /* 多音字发音纠正 */
    const 纠正添加 = document.getElementById("纠正添加");
    if (纠正添加) 纠正添加.addEventListener("click", async () => {
      const 词输入 = document.getElementById("纠正词");
      const 音输入 = document.getElementById("纠正音");
      const 词 = (词输入 && 词输入.value || "").trim();
      const 音 = (音输入 && 音输入.value || "").trim();
      if (!词 || !音) { window.对话界面.添加消息("系统", "请填写词与拼音"); return; }
      try {
        const 结果 = await window.请求.提交("/api/发音纠正", { word: 词, ph: 音 });
        window.对话界面.添加消息("系统", (结果 && 结果.created) ? "已添加发音纠正：" + 词 : "发音纠正已存在：" + 词);
        if (词输入) 词输入.value = "";
        if (音输入) 音输入.value = "";
      } catch (错误) {
        window.对话界面.添加消息("系统", "添加发音纠正失败：" + ((错误 && 错误.message) || 错误));
      }
    });
  }

  /* ---------------- 斜杠命令 ---------------- */
  /* 请求系统接口并把摘要作为系统消息展示；404 时给友好提示 */
  async function 展示系统接口(接口路径, 摘要字段们, 未提供提示) {
    try {
      const 数据 = await window.请求.获取(接口路径);
      let 摘要 = "";
      for (const 字段 of 摘要字段们) {
        if (数据 && typeof 数据[字段] === "string" && 数据[字段].trim()) {
          摘要 = 数据[字段];
          break;
        }
      }
      if (!摘要) 摘要 = JSON.stringify(数据, null, 2);
      window.对话界面.添加消息("系统", 摘要);
    } catch (错误) {
      const 提示 = 错误.message.includes("404")
        ? 未提供提示
        : `${未提供提示}：${错误.message}`;
      window.对话界面.添加消息("系统", 提示);
    }
  }

  function 处理斜杠命令(文本) {
    const 命令 = 文本.split(/\s+/, 1)[0].toLowerCase();
    if (命令 === "/spec") {
      return 展示系统接口(
        "/api/系统/规格",
        ["规格摘要", "content"],
        "规格接口尚未提供（404），请稍后再试");
    }
    if (命令 === "/goal") {
      return 展示系统接口(
        "/api/系统/目标",
        ["进度摘要", "任务进度", "content"],
        "目标接口尚未提供（404），请稍后再试");
    }
    window.对话界面.添加消息("系统", "未知命令，可用 /spec、/goal");
    return Promise.resolve();
  }

  /* ---------------- 对话发送 ---------------- */
  /* 普通文本：优先走全模态实时对话（qwen3.5-omni-plus-realtime，回复语音固定用定制复刻音色）；
     无通话模块时回退到旧的 TTS 合成朗读。 */
  async function 发送普通文本(文本) {
    if (window.通话控制 && typeof window.通话控制.对话发送 === "function") {
      try {
        await window.通话控制.对话发送(文本);
        return;
      } catch (错误) {
        window.对话界面.添加消息("系统", `对话失败：${(错误 && 错误.message) || 错误}`);
        return;
      }
    }
    /* 兜底：无全模态对话能力时的旧 TTS 合成 */
    window.对话界面.添加消息("文本", 文本);
    try {
      const 结果 = await window.请求.提交("/api/合成", { text: 文本 });
      if (结果.audio_url) {
        window.对话界面.添加消息("音频", 结果.audio_url);
      }
      if (结果.warning) {
        window.对话界面.添加消息("系统", 结果.warning);
      }
    } catch (错误) {
      window.对话界面.添加消息("系统", `语音回复失败：${错误.message}`);
    }
  }

  async function 发送消息() {
    const 输入框 = document.getElementById("输入框");
    if (!输入框) return;
    const 文本 = 输入框.value.trim();
    if (!文本) return;
    输入框.value = "";

    if (文本.startsWith("/")) {
      await 处理斜杠命令(文本);
    } else {
      await 发送普通文本(文本);
    }
    输入框.focus();
  }

  function 绑定输入事件() {
    const 输入框 = document.getElementById("输入框");
    const 发送按钮 = document.getElementById("发送按钮");
    const 上传按钮 = document.getElementById("上传按钮");

    if (输入框) {
      输入框.addEventListener("keydown", (事件) => {
        /* Enter 发送，Shift+Enter 换行 */
        if (事件.key === "Enter" && !事件.shiftKey) {
          事件.preventDefault();
          发送消息();
        }
      });
    }
    if (发送按钮) {
      发送按钮.addEventListener("click", 发送消息);
    }
    if (上传按钮) {
      /* 占位按钮：媒体采集模块接入后由它接管 */
      上传按钮.addEventListener("click", () => {
        window.对话界面.添加消息("系统", "上传功能由媒体采集模块提供，敬请期待");
      });
    }
  }

  /* ---------------- 防御性初始化并行模块 ---------------- */
  function 初始化并行模块() {
    if (typeof window.通话控制 !== "undefined" && window.通话控制.初始化) {
      window.通话控制.初始化();
    }
    if (window.音色面板 && window.音色面板.初始化) {
      window.音色面板.初始化();
    }
    if (window.媒体采集 && window.媒体采集.初始化) {
      window.媒体采集.初始化();
    }
  }

  function 初始化() {
    绑定标签切换();
    绑定合成表单();
    绑定输入事件();
    加载引导数据();
    初始化并行模块();
  }

  window.addEventListener("DOMContentLoaded", 初始化);
})();
