/* ============================================================
   EmoCompanion智能体 V2 —— 媒体采集（window.媒体采集）
   功能：
     - 动态注入工具条（录制视频 / 屏幕截图 / 上传图像 / 上传视频）
     - 上传图像 / 视频 / 音频并自动触发多模态识别
     - 5 秒视频录制（支持再次点击取消）
     - 屏幕截图
     - 长按上传（#输入框 长按 700ms）与拖拽上传（#对话区）
   依赖（防御式 typeof 调用，缺失时自动降级）：
     - window.请求.上传(url, 表单数据)     —— FormData 上传
     - window.对话界面.添加消息(类型, 内容, 元信息)
     - window.状态指示.设置状态(状态)
   暴露全局 API：
     window.媒体采集 = { 初始化, 上传文件, 开始录制, 截图 }
   ============================================================ */
(function () {
  "use strict";

  /* ================= 模块内部状态 ================= */
  const 状态 = {
    图片输入: null,   // 隐藏的图片文件选择 input
    视频输入: null,   // 隐藏的视频文件选择 input
    录制按钮: null,   // 工具条"录制视频"按钮
    录制中: false,    // 是否正在录制
    取消录制: false,  // 是否请求取消本次录制
    录制器: null,     // 当前 MediaRecorder 实例
    录制计时器: null, // 5 秒倒计时定时器 ID
    录制剩余: 5,      // 倒计时剩余秒数
  };

  /* 上传并发计数：多个文件同时上传时，全部完成才恢复按钮 */
  let 上传计数 = 0;
  /* 长按上传定时器与触发标志 */
  let 长按定时器 = null;
  let 长按已触发 = false;
  /* 初始化幂等标志 */
  let 已初始化 = false;

  /* 后端类型 → 对话消息类型映射 */
  const 消息类型表 = { 图像: "图像", 视频: "视频", 音频: "音频" };

  /* ================= 防御式全局 API 访问 ================= */

  /* 请求模块是否可用 */
  function 请求可用() {
    return !!(window.请求 && typeof window.请求.上传 === "function");
  }

  /* 添加消息（模块缺失则静默跳过） */
  function 添加消息(类型, 内容) {
    if (window.对话界面 && typeof window.对话界面.添加消息 === "function") {
      window.对话界面.添加消息(类型, 内容);
    }
  }

  /* ================= 内部工具函数 ================= */

  /* 时间戳（用于生成文件名，避免重名） */
  function 时间戳文件名() {
    const 现在 = new Date();
    const 补零 = (数字) => String(数字).padStart(2, "0");
    return (
      现在.getFullYear() + 补零(现在.getMonth() + 1) + 补零(现在.getDate()) +
      "_" + 补零(现在.getHours()) + 补零(现在.getMinutes()) + 补零(现在.getSeconds())
    );
  }

  /* 创建隐藏的原生文件选择 input 并挂到 body */
  function 创建隐藏文件输入(accept) {
    const 输入 = document.createElement("input");
    输入.type = "file";
    输入.accept = accept;
    输入.style.display = "none";
    document.body.appendChild(输入);
    return 输入;
  }

  /* 文件 input 变化时取第一个文件并上传；未选文件则忽略 */
  function 绑定输入变化(输入, 类型) {
    输入.addEventListener("change", () => {
      const 文件 = 输入.files && 输入.files[0];
      if (文件) {
        上传文件(文件, 类型);
      }
      输入.value = ""; // 清空以便再次选择同一文件
    });
  }

  /* 创建工具条按钮 */
  function 创建工具条按钮(容器, 文字) {
    const 按钮 = document.createElement("button");
    按钮.type = "button";
    按钮.className = "次要按钮";
    按钮.textContent = 文字;
    容器.appendChild(按钮);
    return 按钮;
  }

  /* 动态注入拖拽高亮样式（本模块只负责 JS，不修改 CSS 文件） */
  function 注入拖拽样式() {
    if (document.getElementById("媒体采集拖拽样式")) {
      return;
    }
    const 样式 = document.createElement("style");
    样式.id = "媒体采集拖拽样式";
    样式.textContent = [
      "#对话区.拖拽高亮 { outline: 2px dashed var(--主色, #4f8cff); outline-offset: -6px; background: rgba(79,140,255,.06); }",
      "#对话区.拖拽高亮 * { pointer-events: none; }",
    ].join("\n");
    document.head.appendChild(样式);
  }

  /* 切换"上传中"状态：禁用 / 恢复媒体相关按钮（并发计数） */
  function 设置上传中(是否) {
    上传计数 = Math.max(0, 上传计数 + (是否 ? 1 : -1));
    const 禁用 = 上传计数 > 0;
    const 工具条 = document.getElementById("媒体工具条");
    const 按钮们 = [];
    if (工具条) {
      按钮们.push(...工具条.querySelectorAll("button"));
    }
    const 上传按钮 = document.getElementById("上传按钮");
    if (上传按钮) {
      按钮们.push(上传按钮);
    }
    for (const 按钮 of 按钮们) {
      按钮.disabled = 禁用;
    }
  }

  /* 停止媒体流的所有轨道 */
  function 停止轨道(流) {
    if (!流 || typeof 流.getTracks !== "function") {
      return;
    }
    for (const 轨道 of 流.getTracks()) {
      try { 轨道.stop(); } catch (忽略) { /* 忽略停止异常 */ }
    }
  }

  /* ================= 识别（图像 / 视频） ================= */

  /* 上传成功后触发识别，结果以系统消息展示；失败不抛出 */
  async function 触发识别(文件, 类型) {
    try {
      if (!请求可用()) {
        return;
      }
      const 端点 = 类型 === "图像" ? "/api/识别/图像" : "/api/识别/视频";
      const 表单 = new FormData();
      表单.append("文件", 文件);
      const 数据 = await window.请求.上传(端点, 表单);
      const 结果 = 数据 && 数据.结果;
      if (结果) {
        添加消息("系统", "识别结果：" + 结果);
      } else {
        添加消息("系统", "识别服务未返回结果");
      }
    } catch (错误) {
      const 消息 = (错误 && 错误.message) ? String(错误.message) : String(错误);
      /* 502：账号模型未开通，给出友好提示 */
      if (消息.includes("502")) {
        添加消息("系统", "识别服务暂不可用（502）：当前账号模型可能未开通或服务繁忙，媒体文件已上传成功。");
      } else {
        添加消息("系统", "识别失败：" + 消息);
      }
    }
  }

  /* ================= 上传文件 ================= */

  /**
   * 上传文件（FormData：文件 + 类型）→ /api/上传。
   * 成功：按类型展示媒体消息（图像/视频/音频），图像与视频继续触发识别。
   * 失败：系统消息提示。返回 Promise（识别在内部自行处理，不抛出）。
   */
  async function 上传文件(文件, 类型) {
    if (!文件 || !类型) {
      添加消息("系统", "上传失败：无效的文件或类型");
      return;
    }
    if (!请求可用()) {
      添加消息("系统", "上传失败：请求模块未就绪");
      return;
    }
    设置上传中(true);
    try {
      const 表单 = new FormData();
      表单.append("文件", 文件);
      表单.append("类型", 类型);
      const 数据 = await window.请求.上传("/api/上传", 表单);
      const 路径 = 数据 && 数据.路径;
      const 消息类型 = 消息类型表[类型] || 类型;
      if (路径) {
        添加消息(消息类型, 路径);
      } else {
        添加消息("系统", "上传成功，但未返回文件路径");
      }
      /* 图像 / 视频触发多模态识别（音频不识别；不阻塞本 Promise） */
      if (类型 === "图像" || 类型 === "视频") {
        触发识别(文件, 类型);
      }
      return 数据;
    } catch (错误) {
      const 消息 = (错误 && 错误.message) ? String(错误.message) : String(错误);
      添加消息("系统", "上传失败：" + 消息);
    } finally {
      设置上传中(false);
    }
  }

  /* ================= 长按上传 ================= */

  /* 长按 700ms 触发图片文件选择 */
  function 长按开始(事件) {
    if (!状态.图片输入) {
      return;
    }
    长按已触发 = false;
    if (长按定时器) {
      clearTimeout(长按定时器);
    }
    长按定时器 = setTimeout(() => {
      长按定时器 = null;
      长按已触发 = true;
      if (状态.图片输入) {
        状态.图片输入.click();
      }
    }, 700);
  }

  /* 提前松开 / 滑动时取消长按定时器 */
  function 长按结束() {
    if (长按定时器) {
      clearTimeout(长按定时器);
      长按定时器 = null;
    }
  }

  /* ================= 拖拽上传 ================= */

  function 拖拽悬停(事件) {
    事件.preventDefault();
    事件.stopPropagation();
    const 对话区 = document.getElementById("对话区");
    if (对话区) {
      对话区.classList.add("拖拽高亮");
    }
  }

  function 拖拽离开(事件) {
    const 对话区 = document.getElementById("对话区");
    if (对话区) {
      对话区.classList.remove("拖拽高亮");
    }
  }

  function 拖拽放下(事件) {
    事件.preventDefault();
    事件.stopPropagation();
    const 对话区 = document.getElementById("对话区");
    if (对话区) {
      对话区.classList.remove("拖拽高亮");
    }
    const 文件们 = (事件.dataTransfer && 事件.dataTransfer.files) || [];
    for (const 文件 of 文件们) {
      if (文件.type.startsWith("image/")) {
        上传文件(文件, "图像");
      } else if (文件.type.startsWith("video/")) {
        上传文件(文件, "视频");
      } else if (文件.type.startsWith("audio/")) {
        上传文件(文件, "音频");
      } else {
        添加消息("系统", "已忽略不支持的文件类型：" + (文件.name || "未知文件"));
      }
    }
  }

  /* ================= 视频录制（5 秒） ================= */

  /* 选择浏览器支持的 MediaRecorder 视频类型（优先 mp4，其次 webm） */
  function 选择录制类型() {
    if (typeof MediaRecorder === "undefined" || !MediaRecorder.isTypeSupported) {
      return "video/webm";
    }
    const 候选 = ["video/mp4", "video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
    for (const 类型 of 候选) {
      if (MediaRecorder.isTypeSupported(类型)) {
        return 类型;
      }
    }
    return "";
  }

  /* 取消录制：停止倒计时并结束录制（onstop 中判定不再上传） */
  function 取消录制() {
    状态.取消录制 = true;
    if (状态.录制计时器) {
      clearInterval(状态.录制计时器);
      状态.录制计时器 = null;
    }
    const 录制器 = 状态.录制器;
    if (录制器 && 录制器.state !== "inactive") {
      try { 录制器.stop(); } catch (忽略) { /* 忽略停止异常 */ }
    }
  }

  /* 清理录制相关状态并恢复按钮文字 */
  function 清理录制状态() {
    状态.录制中 = false;
    状态.录制器 = null;
    if (状态.录制按钮) {
      状态.录制按钮.textContent = "录制视频";
    }
  }

  /**
   * 开始 5 秒视频录制（手动授权）：
   *  - getUserMedia 授权失败 → 系统消息提示并返回
   *  - 倒计时 5/4/3/2/1，到时自动停止
   *  - 录制中再次点击按钮 → 取消录制
   *  - 停止后把 blob 转 File 调用 上传文件
   */
  async function 开始录制() {
    /* 录制中再次点击 → 取消 */
    if (状态.录制中) {
      取消录制();
      return;
    }
    if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== "function") {
      添加消息("系统", "无法访问摄像头：当前环境不支持摄像头采集");
      return;
    }
    if (typeof MediaRecorder === "undefined") {
      添加消息("系统", "无法录制视频：当前浏览器不支持 MediaRecorder");
      return;
    }
    try {
      /* 手动授权：浏览器自动弹出授权框，用户允许后继续 */
      const 流 = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });

      状态.录制中 = true;
      状态.取消录制 = false;
      if (状态.录制按钮) {
        状态.录制按钮.disabled = false;
        状态.录制按钮.textContent = "录制中 5";
      }

      const 类型 = 选择录制类型();
      const 录制器 = new MediaRecorder(流, 类型 ? { mimeType: 类型 } : undefined);
      状态.录制器 = 录制器;

      const 分片 = [];
      录制器.ondataavailable = (事件) => {
        if (事件.data && 事件.data.size > 0) {
          分片.push(事件.data);
        }
      };
      录制器.onerror = () => {
        添加消息("系统", "录制过程中出现错误");
        if (录制器.state !== "inactive") {
          录制器.stop();
        }
      };
      录制器.onstop = () => {
        /* 无论取消还是到点，先停止轨道并清理状态 */
        停止轨道(流);
        清理录制状态();
        if (状态.取消录制) {
          添加消息("系统", "已取消视频录制");
          return;
        }
        /* 到点自动停止：组装 blob → File → 上传 */
        const blob = new Blob(分片, { type: 类型 || "video/webm" });
        const 扩展名 = String(blob.type).includes("mp4") ? "mp4" : "webm";
        const 文件 = new File([blob], "录制视频_" + 时间戳文件名() + "." + 扩展名, { type: blob.type });
        上传文件(文件, "视频");
      };

      /* 5 秒倒计时：显示 5/4/3/2/1，到时自动停止 */
      状态.录制剩余 = 5;
      状态.录制计时器 = setInterval(() => {
        if (状态.录制剩余 <= 1) {
          if (状态.录制计时器) {
            clearInterval(状态.录制计时器);
            状态.录制计时器 = null;
          }
          if (录制器.state !== "inactive") {
            录制器.stop();
          }
          return;
        }
        状态.录制剩余 -= 1;
        if (状态.录制按钮) {
          状态.录制按钮.textContent = "录制中 " + 状态.录制剩余;
        }
      }, 1000);

      录制器.start();
    } catch (错误) {
      /* 授权被拒绝 / 无摄像头设备 / 其他异常统一友好提示 */
      添加消息("系统", "无法访问摄像头：未授权或设备不存在");
    }
  }

  /* ================= 屏幕截图 ================= */

  /**
   * 屏幕捕获 → 隐藏 video → 等 loadeddata → canvas 绘制一帧 → PNG 上传。
   * 环境不支持（旧浏览器 / 非 localhost 限制）→ 系统消息提示。
   */
  async function 截图() {
    if (!navigator.mediaDevices || typeof navigator.mediaDevices.getDisplayMedia !== "function") {
      添加消息("系统", "当前环境不支持屏幕截图");
      return;
    }
    let 流 = null;
    let 视频 = null;
    try {
      流 = await navigator.mediaDevices.getDisplayMedia({ video: true });

      视频 = document.createElement("video");
      视频.muted = true;
      视频.playsInline = true;
      /* 屏幕外隐藏（display:none 在部分浏览器会阻止加载首帧） */
      视频.style.cssText =
        "position:fixed;left:-9999px;top:0;width:2px;height:2px;opacity:0;pointer-events:none;";
      document.body.appendChild(视频);
      视频.srcObject = 流;

      /* 等待第一帧可绘制（loadeddata） */
      await new Promise((解决, 拒绝) => {
        const 超时 = setTimeout(() => 拒绝(new Error("等待屏幕画面超时")), 10000);
        视频.addEventListener("loadeddata", () => {
          clearTimeout(超时);
          解决();
        }, { once: true });
        视频.addEventListener("error", () => {
          clearTimeout(超时);
          拒绝(new Error("屏幕画面加载失败"));
        }, { once: true });
        视频.play().catch(() => { /* 静音自动播放失败可忽略，loadeddata 仍会触发 */ });
      });

      const 宽 = 视频.videoWidth || 1280;
      const 高 = 视频.videoHeight || 720;
      const 画布 = document.createElement("canvas");
      画布.width = 宽;
      画布.height = 高;
      画布.getContext("2d").drawImage(视频, 0, 0, 宽, 高);

      视频.srcObject = null;

      const blob = await new Promise((解决) => 画布.toBlob(解决, "image/png"));
      if (!blob) {
        添加消息("系统", "截图失败：无法生成图像数据");
        return;
      }
      const 文件 = new File([blob], "屏幕截图_" + 时间戳文件名() + ".png", { type: "image/png" });
      await 上传文件(文件, "图像");
    } catch (错误) {
      /* 用户取消共享或环境不支持 */
      添加消息("系统", "当前环境不支持屏幕截图或已取消");
    } finally {
      if (视频) {
        视频.srcObject = null;
        if (视频.parentNode) {
          视频.parentNode.removeChild(视频);
        }
      }
      if (流) {
        停止轨道(流);
      }
    }
  }

  /* ================= 初始化 ================= */

  /**
   * 初始化媒体功能：
   *  - 工具条注入：录制视频 / 屏幕截图 / 上传图像 / 上传视频（含隐藏 file input ×2）
   *  - #上传按钮 点击 → 图片文件选择
   *  - #输入框 长按 700ms → 图片文件选择
   *  - #对话区 拖拽上传（图像 / 视频 / 音频）
   * 所有 DOM 查找均为防御式，元素不存在则跳过对应功能。
   */
  function 初始化() {
    if (已初始化) {
      return;
    }
    已初始化 = true;
    注入拖拽样式();

    const 工具条 = document.getElementById("媒体工具条");
    const 上传按钮 = document.getElementById("上传按钮");
    const 输入框 = document.getElementById("输入框");
    const 对话区 = document.getElementById("对话区");

    /* 工具条按钮 + 隐藏文件输入 */
    if (工具条) {
      状态.录制按钮 = 创建工具条按钮(工具条, "录制视频");
      状态.录制按钮.addEventListener("click", 开始录制);
      创建工具条按钮(工具条, "屏幕截图").addEventListener("click", 截图);

      状态.图片输入 = 创建隐藏文件输入("image/*");
      状态.视频输入 = 创建隐藏文件输入("video/*");
      绑定输入变化(状态.图片输入, "图像");
      绑定输入变化(状态.视频输入, "视频");

      创建工具条按钮(工具条, "上传图像").addEventListener("click", () => {
        if (状态.图片输入) {
          状态.图片输入.click();
        }
      });
      创建工具条按钮(工具条, "上传视频").addEventListener("click", () => {
        if (状态.视频输入) {
          状态.视频输入.click();
        }
      });
    }

    /* #上传按钮：点击触发图片文件选择 */
    if (上传按钮 && 状态.图片输入) {
      上传按钮.addEventListener("click", () => {
        状态.图片输入.click();
      });
    }

    /* 长按上传：#输入框 长按 700ms → 图片文件选择 */
    if (输入框 && 状态.图片输入) {
      输入框.addEventListener("touchstart", 长按开始, { passive: false });
      输入框.addEventListener("touchend", 长按结束);
      输入框.addEventListener("touchmove", 长按结束);
      输入框.addEventListener("touchcancel", 长按结束);
    }

    /* 拖拽上传：#对话区 */
    if (对话区) {
      对话区.addEventListener("dragover", 拖拽悬停);
      对话区.addEventListener("dragleave", 拖拽离开);
      对话区.addEventListener("drop", 拖拽放下);
    }
  }

  /* ================= 暴露全局 API ================= */
  window.媒体采集 = { 初始化, 上传文件, 开始录制, 截图 };
})();
