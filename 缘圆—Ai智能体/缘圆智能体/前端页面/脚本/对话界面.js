/* ============================================================
   缘圆智能体 V2 —— 对话界面
   暴露全局 API（其他并行模块依赖）：
     window.对话界面.添加消息(类型, 内容, 元信息) -> 消息 DOM 元素
     window.对话界面.流式消息(内容片段)
     window.对话界面.结束回复()
     window.状态指示.设置状态(状态)
   ============================================================ */
(function () {
  "use strict";

  /* 状态名 → 中文显示（含英文别名，兼容并行模块可能的传值） */
  const 状态中文表 = {
    空闲: "空闲", idle: "空闲",
    连接中: "连接中", connecting: "连接中",
    聆听: "聆听", listening: "聆听",
    回复: "回复", replying: "回复", speaking: "回复",
    打断: "打断", interrupted: "打断", interrupting: "打断",
    错误: "错误", error: "错误",
  };
  /* 需要开启动画（呼吸灯）的状态 */
  const 呼吸状态表 = ["连接中", "聆听", "回复", "打断"];

  /* 当前正在流式输出的 AI 气泡 */
  let 当前回复气泡 = null;

  /* 内部：时间戳（HH:MM:SS） */
  function 时间戳() {
    const 现在 = new Date();
    const 补零 = (数字) => String(数字).padStart(2, "0");
    return `${补零(现在.getHours())}:${补零(现在.getMinutes())}:${补零(现在.getSeconds())}`;
  }

  function 时间标签(时间) {
    const 标签 = document.createElement("span");
    标签.className = "消息时间";
    标签.textContent = 时间;
    return 标签;
  }

  function 文本节点(内容) {
    const 正文 = document.createElement("div");
    正文.className = "消息正文";
    正文.textContent = 内容 == null ? "" : String(内容);
    return 正文;
  }

  /* 内部：消息区滚动到底部 */
  function 消息区滚动() {
    const 容器 = document.getElementById("消息滚动区") || document.getElementById("消息列表");
    if (容器) {
      容器.scrollTop = 容器.scrollHeight;
    }
  }

  /**
   * 添加一条消息。
   * 类型：文本（用户气泡）/ 图像 / 音频 / 视频 / 系统（居中灰字）
   * 元信息：可选对象（如 { 时间: "HH:MM:SS" }），可含自定义字段，仅透传。
   * 返回：消息 DOM 元素。
   */
  function 添加消息(类型, 内容, 元信息) {
    const 列表 = document.getElementById("消息列表");
    if (!列表) return null;

    const 元 = window.请求 ? window.请求.合并事件(元信息) : (元信息 || {});
    const 项目 = document.createElement("li");
    const 有效类型 = 类型 || "系统";
    项目.className = "消息项";
    项目.dataset.类型 = 有效类型;
    项目.appendChild(时间标签(元.时间 || 时间戳()));

    if (有效类型 === "图像") {
      项目.classList.add("消息-图像");
      const 图片 = document.createElement("img");
      图片.src = 内容 || "";
      图片.alt = "图像消息";
      项目.appendChild(图片);
    } else if (有效类型 === "音频") {
      项目.classList.add("消息-音频");
      const 音频 = document.createElement("audio");
      音频.controls = true;
      音频.src = 内容 || "";
      项目.appendChild(音频);
    } else if (有效类型 === "视频") {
      项目.classList.add("消息-视频");
      const 视频 = document.createElement("video");
      视频.controls = true;
      视频.src = 内容 || "";
      项目.appendChild(视频);
    } else if (有效类型 === "文本") {
      项目.classList.add("消息-用户");
      项目.appendChild(文本节点(内容));
    } else if (有效类型 === "AI") {
      项目.classList.add("消息-AI");
      项目.appendChild(文本节点(内容));
    } else {
      项目.classList.add("消息-系统");
      项目.appendChild(文本节点(内容));
    }

    列表.appendChild(项目);
    消息区滚动();
    return 项目;
  }

  /**
   * 流式追加文本到"正在回复"的 AI 气泡；没有则新建一个。
   */
  function 流式消息(内容片段) {
    const 列表 = document.getElementById("消息列表");
    if (!列表) return null;

    if (!当前回复气泡 || !当前回复气泡.isConnected) {
      当前回复气泡 = document.createElement("li");
      当前回复气泡.className = "消息项 消息-AI 正在回复";
      当前回复气泡.appendChild(时间标签(时间戳()));
      当前回复气泡.appendChild(文本节点(""));
      列表.appendChild(当前回复气泡);
    }
    const 正文 = 当前回复气泡.querySelector(".消息正文");
    正文.textContent += 内容片段 == null ? "" : String(内容片段);
    消息区滚动();
    return 当前回复气泡;
  }

  /**
   * 结束当前流式回复（去掉正在回复标记）。
   */
  function 结束回复() {
    if (当前回复气泡) {
      当前回复气泡.classList.remove("正在回复");
      当前回复气泡 = null;
    }
  }

  /**
   * 设置连接状态：
   * - #连接状态灯 的 class 切换为 状态-xxx
   * - #全局状态 文字显示中文（空闲/连接中/聆听/回复/打断/错误）
   * - #呼吸灯图标 动画启停（回复/聆听/连接中/打断 开启，空闲/错误 暂停）
   */
  function 设置状态(状态) {
    const 中文 = 状态中文表[状态] || 状态 || "空闲";
    const 灯 = document.getElementById("连接状态灯");
    const 全局 = document.getElementById("全局状态");
    const 图标 = document.getElementById("呼吸灯图标");
    const 状态条 = document.getElementById("状态条");

    if (灯) 灯.className = `状态灯 状态-${中文}`;
    if (状态条) 状态条.className = `状态条 状态-${中文}`;
    if (全局) 全局.textContent = 中文;

    const 动画中 = 呼吸状态表.includes(中文);
    if (图标) 图标.classList.toggle("呼吸灯", 动画中);
  }

  window.对话界面 = { 添加消息, 流式消息, 结束回复 };
  window.状态指示 = { 设置状态 };
})();
