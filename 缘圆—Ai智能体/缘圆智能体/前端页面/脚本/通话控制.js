/* 通话控制：实时通话模块前端逻辑（仿微信语音通话页 UI）。
 *
 * 在 #通话区 容器（页面入口.html 提供）内动态构建 UI：
 *   拨号界面（中央呼吸球 + 顶部计时 + 底部三圆形按钮）与 模型/音色 弹层；
 *   若容器不存在则自动在 document.body 内创建并插入（防御性）。
 *
 * 采集：getUserMedia → AudioWorkletNode("音频采集处理器") → 16k PCM 字节 → WebSocket 二进制帧。
 * 播放：后端转发的 24k PCM 二进制 → 主线程按块累积 → createBuffer(1, 帧数, 24000) 排入播放队列。
 * 事件：{"类型":"状态"} 更新状态；收到"已连接"后开始通话时长计时；
 *       挂断/自动断开/错误 后自动切回「对话」标签页（文本模式）；
 *       通话中切换模型/音色走"静默挂断 → 重新开始"流程（不切标签页）。
 *
 * 依赖（防御式 typeof 调用）：window.对话界面.添加消息(类型, 内容, 元信息)、
 * window.状态指示.设置状态(状态)、window.实时模型列表（可选）、window.请求.获取（可选）。
 */
(function () {
  "use strict";

  // ---------------- 共享模块（防御式） ----------------
  const 对话界面 = (typeof window.对话界面 !== "undefined" && window.对话界面) || null;
  const 状态指示 = (typeof window.状态指示 !== "undefined" && window.状态指示) || null;
  const 实时模型列表 =
    (typeof window.实时模型列表 !== "undefined" &&
      Array.isArray(window.实时模型列表) && window.实时模型列表.length)
      ? window.实时模型列表
      : [
          { id: "qwen3.5-omni-plus-realtime", 名称: "千问3.5全模态", 默认音色: "Tina" },
          { id: "qwen3-omni-flash-realtime", 名称: "千问3.0全模态", 默认音色: "Cherry" },
        ];

  // ---------------- 运行时状态 ----------------
  let 网络 = null;           // WebSocket
  let 音频上下文 = null;     // AudioContext
  let 采集节点 = null;       // AudioWorkletNode
  let 麦克风流 = null;       // 采集用 MediaStream
  let 视频流 = null;         // 视频用 MediaStream
  let 视频元素 = null;       // <video> 预览
  let 画布 = null;           // 抽帧画布
  let 视频定时器 = null;     // setInterval 句柄
  let 播放队列 = [];         // ArrayBuffer 播放队列
  let 活跃源们 = [];         // 已调度未播完的 AudioBufferSourceNode
  let 计划播放时间 = 0;      // AudioContext 时间域，避免停顿
  let 正在播放 = false;
  let 当前回复文本 = "";
  let 正在回复 = false;
  let 视频开启 = false;
  let 挂断中 = false;
  let 静默挂断中 = false;    // "静默挂断 → 重新开始"流程中（挂断时切标签页）
  let 计时定时器 = null;     // 通话时长 setInterval 句柄
  let 通话开始时间 = 0;      // 通话开始时间戳（毫秒）
  let 麦克风设备ID = null;   // 当前选中的麦克风 deviceId（null = 系统默认）
  let 麦克风设备列表 = [];   // enumerateDevices 枚举到的 audioinput 设备
  let 复刻音色ID集合 = new Set();  // 仅合成可用的复刻音色（实时通话不支持）ID 集合
  let 用户手动选过音色 = false;   // 用户是否手动选择过音色（否则固定使用通话复刻音色）
  let 转写消息元素 = null;  // 当前"你说：xxx"实时转写系统消息 DOM
  let 转写缓冲 = "";        // 当前一句语音的转写文本累积
  let 通话记录 = [];        // 本次通话文字记录 [{时间, 角色, 内容}]
  let 麦克风活动定时器 = null;  // 麦克风活动指示器的闪烁定时器
  let 转写开关开启 = true;    // 识别转换（语音转文字实时显示）开关

  /* 当前时间 HH:MM:SS */
  function 当前时间() {
    const 现在 = new Date();
    const 补零 = (数字) => String(数字).padStart(2, "0");
    return 补零(现在.getHours()) + ":" + 补零(现在.getMinutes()) + ":" + 补零(现在.getSeconds());
  }

  /* 记录本次通话的文本条目（用户语音/文字、助手回复） */
  function 记录通话(角色, 内容) {
    if (!内容) { return; }
    通话记录.push({ 时间: 当前时间(), 角色, 内容 });
  }

  /* 挂断时把本次通话文本追加进本地历史（最多保留 30 次会话） */
  function 保存通话历史() {
    if (!通话记录.length) { return; }
    try {
      let 历史 = [];
      try { 历史 = JSON.parse(localStorage.getItem("通话历史记录") || "[]"); } catch (忽略) { /* 忽略 */ }
      if (!Array.isArray(历史)) { 历史 = []; }
      历史.unshift({
        时间: new Date().toLocaleString("zh-CN", { hour12: false }),
        模型: 模型ID,
        条目: 通话记录,
      });
      if (历史.length > 30) { 历史 = 历史.slice(0, 30); }
      localStorage.setItem("通话历史记录", JSON.stringify(历史));
    } catch (忽略) { /* 忽略存储异常 */ }
    通话记录 = [];
  }

  // 当前通话参数（与 弹层 / 开始() 共用，切换模型/音色时更新）
  let 模型ID = (实时模型列表[0] && 实时模型列表[0].id) || "qwen3.5-omni-plus-realtime";
  let 音色ID =
    (实时模型列表[0] && 实时模型列表[0].默认音色) || "Tina";
  const 轮次检测 = "auto";   // 交给后端按模型自动选择（semantic_vad/server_vad，遵循官方协议）

  const 输出采样率 = 24000;  // DashScope 实时模型输出采样率

  // ---------------- 内联 SVG 图标（全部内联，不依赖外部资源/emoji） ----------------
  const 电话SVG =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.127.96.361 1.903.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0 1 22 16.92z"/></svg>';
  const 挂断SVG =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><g transform="rotate(135 12 12)"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.127.96.361 1.903.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0 1 22 16.92z"/></g></svg>';
  const 手机SVG =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="5" y="2" width="14" height="20" rx="2"/><line x1="12" y1="18" x2="12.01" y2="18"/></svg>';
  const 声波SVG =
    '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="2" y="11" width="3" height="2" rx="1"/><rect x="6" y="9" width="3" height="6" rx="1"/><rect x="10" y="6" width="3" height="12" rx="1"/><rect x="14" y="9" width="3" height="6" rx="1"/><rect x="18" y="11" width="3" height="2" rx="1"/></svg>';
  const 模型SVG =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>';
  const 音色SVG =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>';

  // ---------------- 工具 ----------------
  function 显示消息(类型, 内容, 元信息) {
    if (对话界面 && typeof 对话界面.添加消息 === "function") {
      try {
        对话界面.添加消息(类型, 内容, 元信息);
        return;
      } catch (错误) { /* 落到控制台兜底 */ }
    }
    console.log("[" + 类型 + "]", 内容);
  }

  function 设置状态(状态) {
    window.通话控制.状态 = 状态;
    if (状态指示 && typeof 状态指示.设置状态 === "function") {
      try { 状态指示.设置状态(状态); } catch (错误) { /* 忽略 */ }
    }
    更新按钮();
  }

  function 解码Base64(base64) {
    const 二进制 = atob(base64);
    const 字节 = new Uint8Array(二进制.length);
    for (let i = 0; i < 二进制.length; i++) {
      字节[i] = 二进制.charCodeAt(i);
    }
    return 字节.buffer;
  }

  // ---------------- 标签页切换（挂断 → 文本模式） ----------------
  function 切到通话标签页() {
    const 按钮 = document.querySelector('.标签页按钮[data-标签="通话"]');
    if (按钮 && typeof 按钮.click === "function") { 按钮.click(); }
  }

  function 切回对话标签页() {
    const 按钮 = document.querySelector('.标签页按钮[data-标签="对话"]');
    if (按钮 && typeof 按钮.click === "function") { 按钮.click(); }
  }

  // ---------------- 通话时长计时 ----------------
  function 更新时长() {
    const 节点 = document.getElementById("通话-时长");
    if (!节点) { return; }
    const 秒 = 通话开始时间 ? Math.max(0, Math.floor((Date.now() - 通话开始时间) / 1000)) : 0;
    const 分 = Math.floor(秒 / 60);
    const 余秒 = 秒 % 60;
    节点.textContent =
      String(分).padStart(2, "0") + ":" + String(余秒).padStart(2, "0");
  }

  function 开始计时() {
    停止计时();
    通话开始时间 = Date.now();
    更新时长();
    计时定时器 = setInterval(更新时长, 1000);
  }

  function 停止计时() {
    if (计时定时器) {
      clearInterval(计时定时器);
      计时定时器 = null;
    }
    通话开始时间 = 0;
    const 节点 = document.getElementById("通话-时长");
    if (节点) { 节点.textContent = "00:00"; }
  }

  // ---------------- UI 构建（仿微信语音通话页） ----------------
  function 确保容器() {
    let 容器 = document.getElementById("通话区");
    if (!容器) {
      容器 = document.createElement("div");
      容器.id = "通话区";
      document.body.appendChild(容器);
    }
    // 已构建过则直接复用（防御性：避免重复注入）
    if (容器.querySelector(".通话拨号界面")) { return 容器; }

    const 模型选项 = 实时模型列表.map((模型) =>
      `<button class="通话弹层项" type="button" data-模型id="${模型.id}">${模型.名称}</button>`).join("");

    容器.innerHTML = `
      <div id="通话-拨号界面" class="通话拨号界面">
        <div class="通话顶栏">
          <span id="通话-时长" class="通话时长">00:00</span>
          <div class="通话顶栏控件">
            <label class="通话麦克风开关" title="选择麦克风设备（可随时切换）">
              <select id="通话-麦克风" class="通话麦克风选择"></select>
            </label>
            <button id="通话-历史" class="通话视频开关" type="button" title="查看通话历史记录">历史</button>
          </div>
        </div>
        <div class="通话球区">
          <div id="通话-球体" class="通话球体 呼吸-空闲">
            <div class="通话球体内核">
              <span id="通话-球图标" class="通话球图标">${手机SVG}</span>
              <span id="通话-球文字" class="通话球文字">待机</span>
            </div>
          </div>
          <div id="通话-状态文字" class="通话状态文字">通话模式 · 点击开始</div>
          <div id="通话-模型信息" class="通话模型信息"></div>
        </div>
        <!-- 麦克风采集指示 + 通话中的实时转写 / 助手回复显示区 -->
        <div class="通话-采集指示行">
          <span id="通话-麦克风活动" class="通话麦克风活动" title="麦克风采集指示：说话时圆点会点亮"></span>
        </div>
        <div id="通话-消息区" class="通话消息区"></div>
        <div class="通话按钮组">
          <button id="通话-切换模型" class="通话圆形按钮" type="button" title="切换模型">
            <span class="通话按钮图标">${模型SVG}</span>
            <span class="通话按钮文字">切换模型</span>
          </button>
          <button id="通话-主按钮" class="通话圆形按钮 通话挂断按钮" type="button" title="开始通话">
            <span class="通话按钮图标">${电话SVG}</span>
            <span class="通话按钮文字">开始通话</span>
          </button>
          <button id="通话-切换音色" class="通话圆形按钮" type="button" title="切换音色">
            <span class="通话按钮图标">${音色SVG}</span>
            <span class="通话按钮文字">切换音色</span>
          </button>
        </div>
        <div class="通话文本行">
          <input id="通话-文本输入" type="text" placeholder="语音或文字输入（回车发送）" />
          <button id="通话-文本发送" class="通话文本发送" type="button">发送</button>
        </div>
        <p class="通话提示">建议佩戴耳机播放，避免扬声器回声触发误打断。</p>
        <video id="通话-视频预览" autoplay muted playsinline
          style="display:none;max-width:180px;max-height:135px;position:absolute;right:12px;bottom:12px;border-radius:10px;border:1px solid var(--分隔线);"></video>
      </div>
      <div id="通话-模型弹层" class="通话弹层 隐藏">
        <div class="通话弹层标题">切换模型</div>
        <div class="通话弹层列表">${模型选项}</div>
      </div>
      <div id="通话-音色弹层" class="通话弹层 隐藏">
        <div class="通话弹层标题">切换音色</div>
        <div class="通话弹层列表" id="通话-音色列表"></div>
        <div class="通话弹层输入行">
          <input id="通话-音色输入" type="text" placeholder="输入音色ID" />
          <button id="通话-音色确定" class="次要按钮" type="button">确定</button>
        </div>
      </div>
      <div id="通话-历史弹层" class="通话弹层 隐藏">
        <div class="通话弹层标题">通话历史记录</div>
        <div class="通话弹层列表" id="通话-历史列表"></div>
      </div>`;
    return 容器;
  }

  // 状态 → 状态提示文字
  const 状态文字表 = {
    "空闲": "通话模式 · 点击开始",
    "连接中": "正在连接模型…",
    "聆听": "聆听中…",
    "回复": "回复中…",
    "打断": "已打断 · 聆听中…",
    "错误": "连接出错，请重试",
  };
  // 状态 → 球内短文字
  const 球文字表 = {
    "空闲": "待机",
    "连接中": "连接中",
    "聆听": "聆听中",
    "回复": "回复中",
    "打断": "聆听中",
    "错误": "出错",
  };

  function 通话状态中() {
    return ["连接中", "聆听", "回复", "打断"].indexOf(window.通话控制.状态) >= 0;
  }

  function 更新按钮() {
    const 状态 = window.通话控制.状态;
    const 可挂断 = 通话状态中() || 状态 === "错误";

    // 中间主按钮：空闲=绿色"开始通话"，通话中/错误=红色"挂断"（连接中禁用）
    const 主按钮 = document.getElementById("通话-主按钮");
    if (主按钮) {
      主按钮.classList.toggle("通话中", 可挂断);
      主按钮.disabled = 状态 === "连接中";
      主按钮.title = 可挂断 ? "挂断通话" : "开始通话";
      const 图标 = 主按钮.querySelector(".通话按钮图标");
      if (图标) { 图标.innerHTML = 可挂断 ? 挂断SVG : 电话SVG; }
      const 文字 = 主按钮.querySelector(".通话按钮文字");
      if (文字) { 文字.textContent = 可挂断 ? "挂断" : "开始通话"; }
    }

    // 呼吸球体：按状态切换呼吸节奏/颜色
    const 球体 = document.getElementById("通话-球体");
    if (球体) {
      球体.classList.remove("呼吸-空闲", "呼吸-连接中", "呼吸-通话", "呼吸-回复", "呼吸-错误");
      const 呼吸类 = {
        "空闲": "呼吸-空闲",
        "连接中": "呼吸-连接中",
        "聆听": "呼吸-通话",
        "回复": "呼吸-回复",
        "打断": "呼吸-通话",
        "错误": "呼吸-错误",
      }[状态] || "呼吸-空闲";
      球体.classList.add(呼吸类);
      const 球图标 = document.getElementById("通话-球图标");
      if (球图标) { 球图标.innerHTML = 可挂断 && 状态 !== "错误" ? 声波SVG : 手机SVG; }
      const 球文字 = document.getElementById("通话-球文字");
      if (球文字) { 球文字.textContent = 球文字表[状态] || "待机"; }
    }

    // 状态提示文字
    const 状态文字节点 = document.getElementById("通话-状态文字");
    if (状态文字节点) {
      状态文字节点.textContent = 状态文字表[状态] || "通话模式 · 点击开始";
    }
  }

  function 更新模型信息() {
    const 节点 = document.getElementById("通话-模型信息");
    if (!节点) { return; }
    const 模型 = 实时模型列表.find((m) => m.id === 模型ID);
    节点.textContent = (模型 ? 模型.名称 : 模型ID) + " · 音色 " + (音色ID || "未选择");
  }

  // ---------------- 对话控制条（模型/音色/识别转换） ----------------
  /* 填充对话控制条的音色下拉（系统音色 + 通话可用的复刻音色） */
  async function 填充对话音色下拉(下拉) {
    if (!下拉) { return; }
    下拉.innerHTML = "";
    const 候选们 = [];
    if (window.请求 && typeof window.请求.获取 === "function") {
      try {
        const 数据 = await window.请求.获取("/api/音色?model=" + encodeURIComponent(模型ID));
        const 音色们 = (数据 && Array.isArray(数据.voices)) ? 数据.voices : [];
        for (const 音色 of 音色们) {
          if (!音色 || !音色.id) { continue; }
          if (/clone|local_clone/.test(String(音色.kind || ""))) {
            if (音色.通话可用) { 候选们.push(音色); }   // 仅通话可用的复刻
          } else if (音色.kind === "system") {
            候选们.push(音色);
          }
        }
      } catch (错误) { /* 音色接口失败则只保留当前音色 */ }
    }
    const 已见 = new Set();
    for (const 音色 of 候选们.concat([{ id: 音色ID, name: 音色ID }])) {
      if (已见.has(音色.id)) { continue; }
      已见.add(音色.id);
      const 选项 = document.createElement("option");
      选项.value = 音色.id;
      选项.textContent = 音色.name || 音色.id;
      下拉.appendChild(选项);
    }
    下拉.value = 音色ID;
  }

  /* 构建对话页输入栏上方的控制条：模型下拉 + 音色下拉 + 识别转换开关 */
  async function 构建对话控制条() {
    const 条 = document.getElementById("对话控制条");
    if (!条 || 条.querySelector("select")) { return; }
    const 模型下拉 = document.createElement("select");
    模型下拉.className = "对话控制条下拉 对话控制条模型";
    模型下拉.title = "选择全模态模型";
    for (const 模型 of 实时模型列表) {
      const 选项 = document.createElement("option");
      选项.value = 模型.id;
      选项.textContent = 模型.名称;
      模型下拉.appendChild(选项);
    }
    模型下拉.value = 模型ID;
    模型下拉.addEventListener("change", () => 切换模型(模型下拉.value));

    const 音色下拉 = document.createElement("select");
    音色下拉.className = "对话控制条下拉 对话控制条音色";
    音色下拉.title = "选择 AI 回复音色";
    音色下拉.addEventListener("change", () => 选择音色(音色下拉.value));

    const 转写开关 = document.createElement("label");
    转写开关.className = "对话控制条开关";
    转写开关.title = "识别转换：实时显示语音识别文字";
    const 勾选 = document.createElement("input");
    勾选.type = "checkbox";
    勾选.checked = 转写开关开启;
    勾选.addEventListener("change", () => { 转写开关开启 = 勾选.checked; });
    转写开关.appendChild(勾选);
    转写开关.appendChild(document.createTextNode("识别转换"));

    条.appendChild(模型下拉);
    条.appendChild(音色下拉);
    条.appendChild(转写开关);
    await 填充对话音色下拉(音色下拉);
  }

  /* 模型切换后刷新对话控制条音色下拉 */
  async function 刷新对话音色下拉() {
    const 下拉 = document.querySelector("#对话控制条 select.对话控制条音色");
    if (下拉) { await 填充对话音色下拉(下拉); }
  }

  // ---------------- 弹层（模型/音色选择浮层） ----------------
  function 关闭弹层() {
    const 模型弹层 = document.getElementById("通话-模型弹层");
    const 音色弹层 = document.getElementById("通话-音色弹层");
    const 历史弹层 = document.getElementById("通话-历史弹层");
    if (模型弹层) { 模型弹层.classList.add("隐藏"); }
    if (音色弹层) { 音色弹层.classList.add("隐藏"); }
    if (历史弹层) { 历史弹层.classList.add("隐藏"); }
  }

  /* 渲染通话历史记录（localStorage 中的最近 30 次会话） */
  function 渲染通话历史() {
    const 列表 = document.getElementById("通话-历史列表");
    if (!列表) { return; }
    let 历史 = [];
    try { 历史 = JSON.parse(localStorage.getItem("通话历史记录") || "[]"); } catch (忽略) { /* 忽略 */ }
    if (!Array.isArray(历史)) { 历史 = []; }
    列表.innerHTML = "";
    if (!历史.length) {
      const 空 = document.createElement("div");
      空.className = "通话弹层空";
      空.textContent = "暂无通话历史记录";
      列表.appendChild(空);
      return;
    }
    for (const 会话 of 历史) {
      const 块 = document.createElement("div");
      块.className = "通话历史会话";
      const 头 = document.createElement("div");
      头.className = "通话历史头";
      头.textContent = (会话.时间 || "") + " · " + (会话.模型 || "");
      块.appendChild(头);
      const 条目们 = Array.isArray(会话.条目) ? 会话.条目 : [];
      if (!条目们.length) {
        const 行 = document.createElement("div");
        行.className = "通话历史行";
        行.textContent = "（无文字记录）";
        块.appendChild(行);
      }
      for (const 条 of 条目们) {
        const 行 = document.createElement("div");
        行.className = "通话历史行 通话历史行-" + (条.角色 === "用户" ? "用户" : "助手");
        行.textContent = "[" + (条.时间 || "") + "] " + (条.角色 === "用户" ? "你说" : "助手") + "：" + 条.内容;
        块.appendChild(行);
      }
      列表.appendChild(块);
    }
  }

  // 内置常用音色（后端 /api/音色 不可用时的兜底，含当前模型默认音色）
  const 内置常用音色 = ["Tina", "Cherry", "Serena", "Ethan", "Chelsie"];

  // 拉取音色列表：优先 GET /api/音色?model=当前实时模型ID（可能 502/404，try/catch 兜底内置列表）
  async function 加载音色列表() {
    const 列表 = document.getElementById("通话-音色列表");
    if (!列表) { return; }
    let 系统音色们 = [];
    let 通话复刻音色们 = [];   // Omni 复刻（绑定 realtime 模型，通话/对话可用）
    let 合成复刻音色们 = [];   // TTS 复刻（仅语音合成可用）
    复刻音色ID集合 = new Set();
    if (window.请求 && typeof window.请求.获取 === "function") {
      try {
        const 数据 = await window.请求.获取("/api/音色?model=" + encodeURIComponent(模型ID));
        const 音色们 = (数据 && Array.isArray(数据.voices)) ? 数据.voices : [];
        for (const 音色 of 音色们) {
          if (!音色 || !音色.id) { continue; }
          const kind = String(音色.kind || "");
          // 本地复刻音色：kind 含 clone / local_clone
          if (/clone|local_clone/.test(kind)) {
            if (音色.通话可用) {
              通话复刻音色们.push(音色);
            } else {
              合成复刻音色们.push(音色);
              复刻音色ID集合.add(音色.id);
            }
          } else if (kind === "system") {
            系统音色们.push(音色);
          }
        }
      } catch (错误) {
        // /api/音色 可能 502/404：静默兜底用内置列表
        系统音色们 = [];
        通话复刻音色们 = [];
        合成复刻音色们 = [];
      }
    }
    // 保证内置常用音色 + 当前模型默认音色一定可选（去重）
    const 已见 = new Set(系统音色们.map((音色) => 音色.id));
    for (const 名 of 内置常用音色) {
      if (!已见.has(名)) {
        系统音色们.push({ id: 名, name: 名, kind: "system" });
      }
    }
    const 当前模型 = 实时模型列表.find((m) => m.id === 模型ID);
    if (当前模型 && 当前模型.默认音色 && !已见.has(当前模型.默认音色)) {
      系统音色们.push({ id: 当前模型.默认音色, name: 当前模型.默认音色, kind: "system" });
    }
    // 若用户未手动选过音色，且存在通话可用的复刻音色：固定使用定制音色作为回复音色
    if (!用户手动选过音色 && 通话复刻音色们.length) {
      音色ID = 通话复刻音色们[0].id;
      更新模型信息();
    }
    渲染音色列表(列表, 系统音色们, 通话复刻音色们, 合成复刻音色们);
  }

  function 渲染音色列表(列表, 系统音色们, 通话复刻音色们, 合成复刻音色们) {
    列表.innerHTML = "";
    const 添加分组 = (标题, 条目们, 通话禁用) => {
      if (!条目们.length) { return; }
      const 分组名 = document.createElement("div");
      分组名.className = "通话弹层分组名";
      分组名.textContent = 标题;
      列表.appendChild(分组名);
      for (const 音色 of 条目们) {
        const 项 = document.createElement("button");
        项.type = "button";
        项.className = "通话弹层项" + (音色.id === 音色ID ? " 当前" : "");
        项.dataset.音色id = 音色.id;
        // 仅合成可用的复刻音色：禁用并标注（disabled 按钮不触发点击）
        if (通话禁用) {
          项.disabled = true;
          项.classList.add("通话弹层项-禁用");
          项.title = "该复刻音色仅支持语音合成，实时通话暂不支持";
          项.textContent = (音色.name || 音色.id) + "（合成专用）";
        } else {
          项.textContent = 音色.name || 音色.id;
        }
        列表.appendChild(项);
      }
    };
    添加分组("常用音色", 系统音色们);
    添加分组("复刻音色（通话可用）", 通话复刻音色们);
    添加分组("复刻音色（仅合成）", 合成复刻音色们, true);
    if (!系统音色们.length && !通话复刻音色们.length && !合成复刻音色们.length) {
      const 空 = document.createElement("div");
      空.className = "通话弹层空";
      空.textContent = "暂无音色，可在下方输入音色ID";
      列表.appendChild(空);
    }
  }

  // 音色弹层内当前项高亮（音色切换后同步）
  function 更新音色弹层高亮() {
    const 列表 = document.getElementById("通话-音色列表");
    if (!列表) { return; }
    const 当前ID = 音色ID;
    列表.querySelectorAll("[data-音色id]").forEach((项) => {
      项.classList.toggle("当前", 项.dataset.音色id === 当前ID);
    });
  }

  function 绑定UI() {
    // 中间主按钮：空闲=开始通话，通话中=挂断
    const 主按钮 = document.getElementById("通话-主按钮");
    if (主按钮) {
      主按钮.addEventListener("click", () => {
        if (可挂断状态()) { 挂断(); } else { 开始(); }
      });
    }

    // 切换模型：弹出/收起模型弹层
    const 切换模型按钮 = document.getElementById("通话-切换模型");
    if (切换模型按钮) {
      切换模型按钮.addEventListener("click", (事件) => {
        事件.stopPropagation();
        const 弹层 = document.getElementById("通话-模型弹层");
        if (!弹层) { return; }
        const 隐藏 = 弹层.classList.contains("隐藏");
        关闭弹层();
        if (隐藏) { 弹层.classList.remove("隐藏"); }
      });
    }

    // 切换音色：弹出/收起音色弹层（打开时刷新列表）
    const 切换音色按钮 = document.getElementById("通话-切换音色");
    if (切换音色按钮) {
      切换音色按钮.addEventListener("click", (事件) => {
        事件.stopPropagation();
        const 弹层 = document.getElementById("通话-音色弹层");
        if (!弹层) { return; }
        const 隐藏 = 弹层.classList.contains("隐藏");
        关闭弹层();
        if (隐藏) {
          加载音色列表();
          弹层.classList.remove("隐藏");
        }
      });
    }

    // 模型弹层：点击选项切换模型（事件委托）
    const 模型弹层 = document.getElementById("通话-模型弹层");
    if (模型弹层) {
      模型弹层.addEventListener("click", (事件) => {
        const 项 = 事件.target && 事件.target.closest ? 事件.target.closest("[data-模型id]") : null;
        if (项) { 切换模型(项.dataset.模型id); }
      });
    }

    // 音色弹层：点击列表项 / 输入音色ID 后确定（事件委托）
    const 音色弹层 = document.getElementById("通话-音色弹层");
    if (音色弹层) {
      音色弹层.addEventListener("click", (事件) => {
        const 项 = 事件.target && 事件.target.closest ? 事件.target.closest("[data-音色id]") : null;
        if (项) {
          选择音色(项.dataset.音色id);
          return;
        }
        const 确定按钮 = 事件.target && 事件.target.closest ? 事件.target.closest("#通话-音色确定") : null;
        if (确定按钮) {
          const 输入 = document.getElementById("通话-音色输入");
          const 值 = 输入 ? String(输入.value || "").trim() : "";
          if (值) { 选择音色(值); }
        }
      });
    }

    const 视频开关 = document.getElementById("通话-视频");
    if (视频开关) { 视频开关.addEventListener("change", 切换视频); }

    // 麦克风下拉：切换设备；通话中切换走"静默挂断 → 重新开始"
    const 麦克风下拉 = document.getElementById("通话-麦克风");
    if (麦克风下拉) {
      麦克风下拉.addEventListener("change", () => {
        const 设备ID = 麦克风下拉.value || null;
        if (设备ID === 麦克风设备ID) { return; }
        麦克风设备ID = 设备ID;
        try {
          localStorage.setItem("通话麦克风设备ID", 设备ID || "");
        } catch (忽略) { /* 忽略存储异常 */ }
        if (通话状态中()) {
          显示消息("系统", "已切换麦克风，正在重新连接…");
          静默重连();
        } else {
          显示消息("系统", 设备ID ? "已切换麦克风设备" : "已恢复使用系统默认麦克风");
        }
      });
    }

    // 历史记录按钮：弹出/收起历史弹层（打开时刷新）
    const 历史按钮 = document.getElementById("通话-历史");
    if (历史按钮) {
      历史按钮.addEventListener("click", (事件) => {
        事件.stopPropagation();
        const 弹层 = document.getElementById("通话-历史弹层");
        if (!弹层) { return; }
        const 隐藏 = 弹层.classList.contains("隐藏");
        关闭弹层();
        if (隐藏) {
          渲染通话历史();
          弹层.classList.remove("隐藏");
        }
      });
    }

    // 通话文字输入：回车/按钮发送，作为用户消息发给模型
    const 文本输入 = document.getElementById("通话-文本输入");
    const 文本发送 = document.getElementById("通话-文本发送");
    const 发送通话文本 = () => {
      if (!文本输入) { return; }
      const 内容 = String(文本输入.value || "").trim();
      if (!内容) { return; }
      文本输入.value = "";
      if (!网络 || 网络.readyState !== WebSocket.OPEN) {
        显示消息("系统", "通话未连接，无法发送文字");
        return;
      }
      网络.send(JSON.stringify({ 类型: "文本", 内容: 内容 }));
      显示消息("文本", 内容);
      记录通话("用户", 内容);
      const 消息区 = document.getElementById("通话-消息区");
      if (消息区) {
        消息区.textContent = "你说：" + 内容;
        消息区.classList.add("通话消息-最终");
      }
    };
    if (文本发送) { 文本发送.addEventListener("click", 发送通话文本); }
    if (文本输入) {
      文本输入.addEventListener("keydown", (事件) => {
        if (事件.key === "Enter" && !事件.shiftKey) {
          事件.preventDefault();
          发送通话文本();
        }
      });
    }
  }

  // ---------------- 采集 ----------------
  // 触发一次麦克风授权（成功即停止轨道，仅用于换取权限/设备信息）
  async function 请求麦克风授权() {
    try {
      const 流 = await navigator.mediaDevices.getUserMedia({ audio: true });
      for (const 轨道 of 流.getTracks()) {
        try { 轨道.stop(); } catch (忽略) { /* 忽略 */ }
      }
      return true;
    } catch (错误) {
      return false;
    }
  }

  /* 枚举音频输入设备并刷新下拉框；未授权时先请求授权再重新枚举 */
  async function 刷新麦克风列表() {
    const 下拉 = document.getElementById("通话-麦克风");
    if (!下拉) { return; }
    if (!navigator.mediaDevices || typeof navigator.mediaDevices.enumerateDevices !== "function") {
      下拉.innerHTML = "";
      const 选项 = document.createElement("option");
      选项.value = "";
      选项.textContent = "当前环境不支持设备枚举";
      下拉.appendChild(选项);
      下拉.disabled = true;
      return;
    }
    let 设备们 = await navigator.mediaDevices.enumerateDevices();
    // 未授权时设备 label 为空 → 先请求一次授权，再重新枚举拿设备名
    if (!设备们.some((设备) => 设备.kind === "audioinput" && 设备.label)) {
      await 请求麦克风授权();
      设备们 = await navigator.mediaDevices.enumerateDevices();
    }
    麦克风设备列表 = 设备们.filter((设备) => 设备.kind === "audioinput");
    const 之前选中 = 麦克风设备ID;
    下拉.innerHTML = "";
    下拉.disabled = false;
    const 默认项 = document.createElement("option");
    默认项.value = "";
    默认项.textContent = 麦克风设备列表.length ? "系统默认麦克风" : "未检测到麦克风设备";
    下拉.appendChild(默认项);
    麦克风设备列表.forEach((设备, 索引) => {
      const 选项 = document.createElement("option");
      选项.value = 设备.deviceId;
      选项.textContent = 设备.label || ("麦克风 " + (索引 + 1));
      下拉.appendChild(选项);
    });
    if (之前选中 && 麦克风设备列表.some((设备) => 设备.deviceId === 之前选中)) {
      下拉.value = 之前选中;
    } else {
      麦克风设备ID = null;
      下拉.value = "";
    }
  }

  // 把浏览器错误翻译成可操作的中文提示（诊断用，会再次触发一次 getUserMedia）
  async function 诊断麦克风错误(原始错误) {
    // 具体错误先落控制台，便于排查（后台/前端均无日志时可用 F12 查看）
    try { console.error("[通话] 麦克风采集失败：", 原始错误); } catch (忽略) { /* 忽略 */ }
    try {
      await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (错误) {
      const 提示表 = {
        NotAllowedError: "麦克风权限被拒绝：请点击页面地址栏旁的麦克风图标允许访问；若仍失败，请检查 Windows「设置 → 隐私 → 麦克风」是否允许本应用/浏览器访问",
        PermissionDeniedError: "麦克风权限被拒绝：请在浏览器中允许访问麦克风",
        NotFoundError: "未检测到麦克风设备：请确认已连接麦克风（内置/耳机/USB），并在系统声音设置中选择正确的输入设备",
        NotReadableError: "麦克风被占用或不可用：请关闭正在使用麦克风的其他程序（会议软件、录音工具等）后重试",
        OverconstrainedError: "所选麦克风无法满足采集要求：请尝试切换为「系统默认麦克风」或重新选择设备",
        SecurityError: "当前页面不安全，浏览器禁止使用麦克风：请通过 http://127.0.0.1 或 https 打开本应用（内置浏览器窗口不受此限制）",
        AbortError: "麦克风初始化被中断（通常为设备热插拔或重复请求）：请重试，必要时刷新页面",
        TypeError: "当前环境不支持麦克风接口：请使用新版 Chrome/Edge，或直接使用本应用的内置浏览器窗口",
      };
      const 名称 = 错误 && 错误.name;
      throw new Error(
        提示表[名称] || ("无法发起麦克风：" + ((错误 && 错误.message) || 名称 || "未知错误")));
    }
    throw new Error("无法发起麦克风（浏览器未返回明确原因，请检查系统声音设置中的输入设备后重试）");
  }

  /* 格式化错误：name: message，避免只显示含糊的 message */
  function 格式化错误(错误, 兜底) {
    if (!错误) { return 兜底 || "未知错误"; }
    const 名称 = 错误.name ? 错误.name + "：" : "";
    return 名称 + (错误.message || 兜底 || String(错误));
  }

  /* 恢复音频上下文到 running：若保持 suspended，AudioWorklet 采集不运行、播放也无声。
     失败不抛出（不阻断通话），但记录告警便于排查。 */
  async function 尝试恢复音频上下文(来源) {
    if (!音频上下文) { return; }
    try {
      if (音频上下文.state !== "running") {
        await 音频上下文.resume();
      }
    } catch (错误) {
      try { console.warn("[通话] AudioContext.resume 失败（" + 来源 + "，可能导致无声）：", 错误); } catch (忽略) { /* 忽略 */ }
    }
  }

  /* 麦克风活动指示：每次收到采集 PCM 数据时点亮圆点，确认麦克风采集在工作 */
  function 麦克风活动指示() {
    const 指示 = document.getElementById("通话-麦克风活动");
    if (!指示) { return; }
    指示.classList.add("活动");
    if (麦克风活动定时器) { clearTimeout(麦克风活动定时器); }
    麦克风活动定时器 = setTimeout(() => {
      指示.classList.remove("活动");
    }, 150);
  }

  async function 初始化采集() {
    const 音频上下文类 = window.AudioContext || window.webkitAudioContext;
    if (!音频上下文类) {
      throw new Error("当前浏览器不支持 AudioContext，请使用新版 Chrome/Edge（或本应用内置浏览器窗口）");
    }
    // 注意：不能用「音频上下文类.prototype.audioWorklet」做存在性判断——
    // 该属性在 Chromium 中是 getter，在 prototype（而非实例）上访问会抛
    // "TypeError: Illegal invocation"。用 in 操作符只查属性存在、不触发 getter。
    if (!("audioWorklet" in 音频上下文类.prototype)) {
      throw new Error("当前浏览器不支持 AudioWorklet，请使用新版 Chrome/Edge（或本应用内置浏览器窗口）");
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("当前环境不允许使用麦克风：本功能需要安全上下文（通过 http://127.0.0.1 或 https 访问），浏览器才会开放麦克风接口");
    }

    // ---- 第一步：创建并唤醒 AudioContext（贴近用户手势） ----
    try {
      音频上下文 = new 音频上下文类();
    } catch (错误) {
      throw new Error("创建音频上下文失败：" + 格式化错误(错误));
    }
    await 尝试恢复音频上下文("初始化");

    // ---- 第二步：获取麦克风流（最依赖权限/用户手势） ----
    麦克风流 = null;
    /* 所选设备（deviceId）作为第一优先级；选中设备不可用时自动回退默认设备 */
    const 设备约束 = 麦克风设备ID ? { deviceId: { exact: 麦克风设备ID } } : {};
    const 约束们 = 麦克风设备ID
      ? [
          { audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, ...设备约束 } },
          { audio: 设备约束 },
          { audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } },
          { audio: true },
        ]
      : [
          { audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } },
          { audio: true },
        ];
    let 最后错误 = null;
    for (const 约束 of 约束们) {
      try {
        麦克风流 = await navigator.mediaDevices.getUserMedia(约束);
        break;
      } catch (错误) { 最后错误 = 错误; }
    }
    if (!麦克风流) {
      throw await 诊断麦克风错误(最后错误);
    }
    // 关键：getUserMedia 授权成功后再次恢复 AudioContext —— 媒体授权可解锁浏览器的
    // 自动播放限制，此时 resume 成功率最高（WebView2 常见无声根因在此）。
    await 尝试恢复音频上下文("麦克风授权后");

    // ---- 第三步：加载 AudioWorklet 处理器并挂接采集节点 ----
    function 停止预留流() {
      if (麦克风流) {
        麦克风流.getTracks().forEach((轨道) => { try { 轨道.stop(); } catch (忽略) { /* 忽略 */ } });
        麦克风流 = null;
      }
    }
    /* 加载工作集模块：
       1) 直接 addModule，失败重试 3 次（抗瞬时中断/服务重启竞态）；
       2) 仍失败则把模块源码装进 Blob 再用 addModule（Blob URL 恒为 JS MIME，
          绕开中文路径 / 服务器 MIME 推断差异等导致的
          AbortError: Unable to load a worklet's module）。 */
    async function 加载音频处理器模块() {
      const 模块地址 = "/静态/脚本/音频处理器.js";
      for (let 次 = 0; 次 < 3; 次++) {
        try {
          await 音频上下文.audioWorklet.addModule(模块地址);
          return;
        } catch (错误) {
          if (次 >= 2) { throw 错误; }
          await new Promise((解决) => setTimeout(解决, 300 * (次 + 1)));
        }
      }
    }
    try {
      try {
        await 加载音频处理器模块();
      } catch (直接加载错误) {
        const 响应 = await fetch("/静态/脚本/音频处理器.js");
        if (!响应.ok) {
          throw new Error("HTTP " + 响应.status + "（无法获取音频处理器脚本）");
        }
        const 源码 = await 响应.text();
        const 对象地址 = URL.createObjectURL(
          new Blob([源码], { type: "application/javascript" }));
        try {
          await 音频上下文.audioWorklet.addModule(对象地址);
        } finally {
          URL.revokeObjectURL(对象地址);
        }
      }
    } catch (错误) {
      停止预留流();
      try { 音频上下文.close(); } catch (忽略) { /* 忽略 */ }
      throw new Error("音频处理器加载失败，请刷新页面或更换浏览器后重试：" + 格式化错误(错误));
    }
    try {
      采集节点 = new AudioWorkletNode(音频上下文, "音频采集处理器", {
        numberOfInputs: 1,
        numberOfOutputs: 0,
      });
    } catch (错误) {
      停止预留流();
      try { 音频上下文.close(); } catch (忽略) { /* 忽略 */ }
      throw new Error("创建采集节点失败：" + 格式化错误(错误));
    }
    try {
      const 源 = 音频上下文.createMediaStreamSource(麦克风流);
      源.connect(采集节点);
    } catch (错误) {
      停止预留流();
      try { 采集节点.disconnect(); } catch (忽略) { /* 忽略 */ }
      try { 音频上下文.close(); } catch (忽略) { /* 忽略 */ }
      throw new Error("连接音频节点失败：" + 格式化错误(错误));
    }
    采集节点.port.onmessage = (事件) => {
      if (网络 && 网络.readyState === WebSocket.OPEN && 事件.data instanceof ArrayBuffer) {
        网络.send(事件.data);   // 16k PCM 字节直接作为二进制帧发送
        麦克风活动指示();
      }
    };
  }

  function 停止采集() {
    if (采集节点) {
      try { 采集节点.port.postMessage("暂停"); } catch (错误) { /* 忽略 */ }
      try { 采集节点.disconnect(); } catch (错误) { /* 忽略 */ }
      采集节点 = null;
    }
    if (麦克风流) {
      麦克风流.getTracks().forEach((轨道) => 轨道.stop());
      麦克风流 = null;
    }
    if (音频上下文) {
      try { 音频上下文.close(); } catch (错误) { /* 忽略 */ }
      音频上下文 = null;
    }
  }

  // ---------------- 播放 ----------------
  function 播放音频(字节) {
    if (!字节 || !字节.byteLength) { return; }
    if (!音频上下文) { return; }
    播放队列.push(字节);
    正在回复 = true;
    if (!正在播放) {
      播放下一块();
    }
  }

  /* 串行播放队列：每块解码为 AudioBuffer 后调度，确保音频上下文 running（suspended
     时 start() 会静默无声——这是"有回复文本但没声音"的常见原因）。 */
  async function 播放下一块() {
    if (正在播放) { return; }
    正在播放 = true;
    try {
      while (播放队列.length) {
        const 字节 = 播放队列.shift();
        if (!音频上下文) { break; }
        // 确保上下文 running：重试恢复音频引擎（自动播放限制下可能 suspended）。
        // 注意不跳过块：即使恢复稍慢，start() 已调度，context 恢复后仍会出声。
        if (音频上下文.state !== "running") {
          for (let 次 = 0; 次 < 3 && 音频上下文.state !== "running"; 次++) {
            try { await 音频上下文.resume(); } catch (错误) { /* 忽略 */ }
            if (音频上下文.state !== "running") { await new Promise((解决) => setTimeout(解决, 120)); }
          }
        }
        const 帧数 = 字节.byteLength / 2;
        if (!帧数) { continue; }
        const 缓冲 = 音频上下文.createBuffer(1, 帧数, 输出采样率);
        const 数据 = 缓冲.getChannelData(0);
        const 视图 = new DataView(字节);
        for (let i = 0; i < 帧数; i++) {
          const 值 = 视图.getInt16(i * 2, true);
          数据[i] = 值 < 0 ? 值 / 0x8000 : 值 / 0x7fff;
        }
        const 源 = 音频上下文.createBufferSource();
        源.buffer = 缓冲;
        源.connect(音频上下文.destination);
        const 开始时间 = Math.max(音频上下文.currentTime, 计划播放时间);
        源.start(开始时间);
        计划播放时间 = 开始时间 + 帧数 / 输出采样率;
        活跃源们.push(源);
        // 等待播放完成（带超时保护，避免 context 关闭时 onended 不触发导致卡死）
        await Promise.race([
          new Promise((解决) => { 源.onended = 解决; }),
          new Promise((解决) => setTimeout(解决, 30000)),
        ]);
        const 索引 = 活跃源们.indexOf(源);
        if (索引 >= 0) { 活跃源们.splice(索引, 1); }
      }
    } finally {
      正在播放 = false;
    }
  }

  function 停止播放() {
    播放队列 = [];
    计划播放时间 = 0;
    活跃源们.forEach((源) => {
      try { 源.stop(); } catch (错误) { /* 忽略 */ }
    });
    活跃源们 = [];
    正在播放 = false;
  }

  // ---------------- 网络 ----------------
  function 建立连接(模型ID, 音色, 轮次检测) {
    return new Promise((解决, 拒绝) => {
      const 协议 = location.protocol === "https:" ? "wss://" : "ws://";
      const 地址 =
        协议 + location.host +
        "/api/通话?model=" + encodeURIComponent(模型ID) +
        "&voice=" + encodeURIComponent(音色) +
        "&turn_detection=" + encodeURIComponent(轮次检测);
      网络 = new WebSocket(地址);
      网络.binaryType = "arraybuffer";
      网络.onopen = () => 解决();
      网络.onerror = () => 拒绝(new Error("无法连接服务器（" + 地址 + "）"));
      网络.onclose = () => {
        if (!挂断中) {
          挂断中 = true;
          清理();
          停止计时();                    // 自动断开：清零计时
          设置状态("空闲");
          显示消息("系统", "通话连接已断开");
          if (!静默挂断中) { 切回对话标签页(); }   // 自动断开 → 文本模式
          挂断中 = false;
        }
      };
      网络.onmessage = (事件) => 处理消息(事件.data);
    });
  }

  function 处理消息(数据) {
    if (typeof 数据 === "string") {
      let 事件 = null;
      try { 事件 = JSON.parse(数据); } catch (错误) { return; }
      if (!事件 || typeof 事件 !== "object") { return; }
      if (事件.类型 === "状态") {
        处理状态(事件);
      } else if (事件.类型 === "关闭") {
        显示消息("系统", "服务端连接已关闭：" + (事件.内容 || ""));
        挂断();
      } else if (事件.类型 === "二进制") {
        // 后端透传的服务端二进制帧（如音频）
        播放音频(事件.数据);
      } else {
        处理DashScope事件(事件);
      }
    } else if (数据 instanceof ArrayBuffer) {
      播放音频(数据);
    }
  }

  function 处理状态(事件) {
    const 状态 = 事件.状态;
    if (状态 === "连接中") {
      设置状态("连接中");
      显示消息("系统", "正在连接模型…");
    } else if (状态 === "已连接") {
      设置状态("聆听");
      开始计时();   // 连接成功 → 完全进入通话状态并开始计时
      显示消息("系统", "已连接，请说话（建议佩戴耳机）");
    } else if (状态 === "错误") {
      设置状态("错误");
      显示消息("系统", 事件.内容 || "连接出错");
      挂断();
    } else if (状态 === "已关闭") {
      显示消息("系统", 事件.内容 || "通话已结束");
      挂断();
    }
  }

  /* 实时转写展示：把用户语音转写以系统消息"你说：xxx"实时更新到对话区与通话消息区 */
  function 显示实时转写(文本, 最终) {
    if (!转写开关开启) { return; }   // 识别转换开关关闭时不显示转写
    const 消息区 = document.getElementById("通话-消息区");
    if (消息区) {
      消息区.textContent = "你说：" + 文本;
      消息区.classList.toggle("通话消息-最终", !!最终);
      if (最终) {
        // 定型后短暂停留再清空，避免遮挡下一句
        setTimeout(() => { if (消息区 && 消息区.textContent === "你说：" + 文本) { 消息区.textContent = ""; } }, 2500);
      }
    }
    if (对话界面 && typeof 对话界面.添加消息 === "function") {
      if (!转写消息元素 || !转写消息元素.isConnected) {
        转写消息元素 = 对话界面.添加消息("系统", "你说：" + 文本);
      } else {
        const 正文 = 转写消息元素.querySelector(".消息正文");
        if (正文) { 正文.textContent = "你说：" + 文本; }
      }
    }
    if (最终) {
      记录通话("用户", 文本);   // 语音转写 → 通话历史
      转写消息元素 = null;      // 最终结果定型后，下句重新创建
    }
  }

  /* 在通话消息区显示助手回复文本 */
  function 显示助手文本(文本) {
    const 消息区 = document.getElementById("通话-消息区");
    if (消息区) {
      消息区.textContent = "助手：" + 文本;
      消息区.classList.remove("通话消息-最终");
    }
  }

  /* 清空通话消息区（回复结束/挂断时） */
  function 清空通话消息区() {
    const 消息区 = document.getElementById("通话-消息区");
    if (消息区) { 消息区.textContent = ""; }
  }

  function 处理DashScope事件(事件) {
    const 类型 = 事件.type;
    switch (类型) {
      case "input_audio_buffer.speech_started":
        if (正在回复 || 正在播放) {
          设置状态("打断");
          停止播放();
          当前回复文本 = "";
        }
        // 新一句语音开始：重置转写累积
        转写缓冲 = "";
        转写消息元素 = null;
        设置状态("聆听");
        break;
      case "input_audio_buffer.speech_stopped":
        设置状态("聆听");
        break;
      case "conversation.item.input_audio_transcription.delta":
      case "conversation.item.input_audio_transcription.text":
        // 流式转写（官方为 .delta + text 字段；兼容 .text / transcript / stash）
        转写缓冲 += (事件.text || 事件.transcript || 事件.stash || "");
        显示实时转写(转写缓冲, false);
        break;
      case "conversation.item.input_audio_transcription.completed":
        // 最终转写结果
        转写缓冲 = 事件.transcript || 转写缓冲;
        显示实时转写(转写缓冲, true);
        break;
      case "response.audio_transcript.delta":
      case "response.text.delta":
        当前回复文本 += (事件.delta || "");
        设置状态("回复");
        显示助手文本(当前回复文本);
        // 流式打字机效果：AI 回复显示为左侧气泡（对话面板）
        if (对话界面 && typeof 对话界面.流式消息 === "function") {
          try { 对话界面.流式消息(事件.delta || ""); } catch (忽略) { /* 忽略 */ }
        }
        break;
      case "response.audio.delta":
        设置状态("回复");
        // 官方协议音频字段为 delta（实测 15 个分片均在 delta）；兼容回退 audio
        const 音频数据 = 事件.delta || 事件.audio;
        if (音频数据) {
          播放音频(解码Base64(音频数据));
        }
        break;
      case "response.done":
        结束回复();
        break;
      case "error":
        设置状态("错误");
        显示消息("系统", "模型错误：" + (事件.message || 事件.error?.message || JSON.stringify(事件)));
        break;
      default:
        break;
    }
  }

  function 结束回复() {
    if (当前回复文本) {
      记录通话("助手", 当前回复文本);   // 助手回复 → 通话历史
      当前回复文本 = "";
    }
    if (对话界面 && typeof 对话界面.结束回复 === "function") {
      try { 对话界面.结束回复(); } catch (忽略) { /* 忽略 */ }
    }
    正在回复 = false;
    清空通话消息区();
    设置状态("聆听");
  }

  // ---------------- 视频 ----------------
  async function 开启视频() {
    if (视频流) { return; }
    try {
      视频流 = await navigator.mediaDevices.getUserMedia({ video: true });
    } catch (错误) {
      视频开启 = false;
      const 开关 = document.getElementById("通话-视频");
      if (开关) { 开关.checked = false; }
      显示消息("系统", "未检测到摄像头或已拒绝授权（" + (错误.name || 错误.message) + "）");
      return;
    }
    视频元素 = document.getElementById("通话-视频预览");
    if (视频元素) {
      视频元素.srcObject = 视频流;
      视频元素.style.display = "block";
      视频元素.play().catch(() => {});
    }
    画布 = document.createElement("canvas");
    const 轨道 = 视频流.getVideoTracks()[0];
    const 设置 = 轨道 && 轨道.getSettings ? 轨道.getSettings() : null;
    画布.width = (设置 && 设置.width) || 640;
    画布.height = (设置 && 设置.height) || 480;
    视频定时器 = setInterval(发送视频帧, 1000);
    显示消息("系统", "视频模式已开启（每 1 秒发送一帧）");
  }

  function 发送视频帧() {
    if (!画布 || !网络 || 网络.readyState !== WebSocket.OPEN) { return; }
    const 上下文 = 画布.getContext("2d");
    if (!上下文 || !视频元素 || !视频元素.videoWidth) { return; }
    try {
      上下文.drawImage(视频元素, 0, 0, 画布.width, 画布.height);
      const 数据 = 画布.toDataURL("image/jpeg", 0.7);
      网络.send(JSON.stringify({ 类型: "图像", 数据: 数据 }));
    } catch (错误) { /* 抽帧失败则跳过本帧 */ }
  }

  function 关闭视频() {
    if (视频定时器) {
      clearInterval(视频定时器);
      视频定时器 = null;
    }
    if (视频流) {
      视频流.getTracks().forEach((轨道) => 轨道.stop());
      视频流 = null;
    }
    if (视频元素) {
      视频元素.srcObject = null;
      视频元素.style.display = "none";
    }
    画布 = null;
  }

  // ---------------- 对外 API ----------------
  async function 开始() {
    if (挂断中) { 挂断中 = false; }
    if (网络 && (网络.readyState === WebSocket.OPEN || 网络.readyState === WebSocket.CONNECTING)) {
      显示消息("系统", "通话已在进行中");
      return;
    }
    设置状态("连接中");
    try {
      await 初始化采集();   // 先取麦克风授权（浏览器复用之前的授权，不重复询问）
    } catch (错误) {
      设置状态("错误");
      try { console.error("[通话] 麦克风初始化失败：", 错误); } catch (忽略) { /* 忽略 */ }
      显示消息("系统", "无法访问麦克风：" + 格式化错误(错误));
      return;
    }
    try {
      await 建立连接(模型ID, 音色ID, 轮次检测);
    } catch (错误) {
      设置状态("错误");
      显示消息("系统", "连接失败：" + 错误.message);
      停止采集();
      return;
    }
    if (视频开启) {
      await 开启视频();
    }
  }

  // 挂断：清理完成后自动切回「对话」标签页（文本模式）；
  // 若处于"静默挂断 → 重新开始"流程（静默挂断中）则保持通话标签页。
  function 挂断() {
    if (挂断中) { return; }
    挂断中 = true;
    关闭视频();
    停止播放();
    停止采集();
    if (网络) {
      // 置空 onclose 防止手动挂断后旧连接的 onclose 重复处理/误切标签页
      try { 网络.onclose = null; } catch (错误) { /* 忽略 */ }
      try { 网络.close(); } catch (错误) { /* 忽略 */ }
      网络 = null;
    }
    当前回复文本 = "";
    正在回复 = false;
    停止计时();
    设置状态("空闲");
    if (!静默挂断中) {
      保存通话历史();          // 本次通话文字记录写入历史
      清空通话消息区();
      转写缓冲 = "";
      转写消息元素 = null;
      显示消息("系统", "通话已挂断");
      切回对话标签页();   // 挂断 → 文本模式
    }
    挂断中 = false;
  }

  function 清理() {
    // onclose 触发时的兜底清理（不重复弹消息）
    关闭视频();
    停止播放();
    停止采集();
    网络 = null;
    保存通话历史();   // 自动断开也要落历史
    清空通话消息区();
    转写缓冲 = "";
    转写消息元素 = null;
  }

  function 切换视频() {
    const 开关 = document.getElementById("通话-视频");
    视频开启 = !!(开关 && 开关.checked);
    if (视频开启) {
      if (!网络 || 网络.readyState !== WebSocket.OPEN) {
        显示消息("系统", "请先开始通话，再开启视频");
        视频开启 = false;
        if (开关) { 开关.checked = false; }
        return;
      }
      开启视频();
    } else {
      关闭视频();
    }
  }

  // 切换模型：更新模型信息、自动设置该模型默认音色；通话中走"静默挂断 → 重新开始"
  async function 切换模型(新模型ID) {
    const 模型 = 实时模型列表.find((m) => m.id === 新模型ID);
    if (!模型) { return; }
    if (新模型ID === 模型ID) { 关闭弹层(); return; }   // 相同模型不做处理
    模型ID = 新模型ID;
    if (模型.默认音色) {
      音色ID = 模型.默认音色;   // 3.5→Tina、3.0→Cherry
    }
    更新模型信息();
    await 加载音色列表();   // 刷新音色弹层（跟随新模型的默认音色高亮）
    刷新对话音色下拉();     // 同步对话控制条的音色下拉
    关闭弹层();
    if (通话状态中()) {
      显示消息("系统", "已切换模型，正在重新连接…");
      静默重连();
    } else {
      显示消息("系统", "已切换模型：" + 模型.名称);
    }
  }

  // 切换音色：更新当前音色；通话中后端会话音色在连接时固定，走"静默挂断 → 重新开始"
  async function 选择音色(新音色ID) {
    新音色ID = String(新音色ID || "").trim();
    if (!新音色ID) { return; }
    // 仅合成可用的复刻音色（绑定 TTS 模型）：实时通话模型不支持，禁止选择/输入
    if (复刻音色ID集合.has(新音色ID)) {
      关闭弹层();
      显示消息("系统", "该复刻音色仅支持语音合成，实时通话暂不支持，已保留原音色");
      return;
    }
    用户手动选过音色 = true;
    const 已变 = 新音色ID !== 音色ID;
    音色ID = 新音色ID;
    更新模型信息();
    更新音色弹层高亮();
    关闭弹层();
    if (已变 && 通话状态中()) {
      显示消息("系统", "已切换音色，正在重新连接…");
      静默重连();
    } else if (已变) {
      显示消息("系统", "已切换音色：" + 音色ID);
    }
  }

  // 静默挂断 → 用当前（新）参数重新开始，全程不切标签页
  async function 静默重连() {
    静默挂断中 = true;
    挂断();
    静默挂断中 = false;
    await 开始();
  }

  // 打开：切到通话标签页；空闲时自动开始通话
  function 打开() {
    切到通话标签页();
    if (window.通话控制.状态 === "空闲") {
      开始();
    }
  }

  function 可挂断状态() {
    return 通话状态中() || window.通话控制.状态 === "错误";
  }

  // ---------------- 全模态文字对话（无需麦克风） ----------------
  /* 对话页发送文字：若未连接则静默建立全模态会话（不切标签页、不采集麦克风），
     发文字后模型以当前音色（默认固定为定制复刻音色）返回文本+语音。 */
  async function 对话发送(文本) {
    文本 = String(文本 || "").trim();
    if (!文本) { return; }
    显示消息("文本", 文本);
    记录通话("用户", 文本);
    if (挂断中) { 挂断中 = false; }

    // 确保有音频上下文用于播放回复语音（文字对话不采集麦克风）
    if (!音频上下文) {
      const 音频上下文类 = window.AudioContext || window.webkitAudioContext;
      if (音频上下文类) {
        try {
          音频上下文 = new 音频上下文类();
          await 音频上下文.resume();
        } catch (错误) {
          try { console.warn("[通话] 音频上下文初始化失败：", 错误); } catch (忽略) { /* 忽略 */ }
        }
      }
    }

    const 发送 = () => {
      if (网络 && 网络.readyState === WebSocket.OPEN) {
        网络.send(JSON.stringify({ 类型: "文本", 内容: 文本 }));
        return true;
      }
      return false;
    };
    if (发送()) { return; }

    // 连接建立中：等待就绪后发送
    if (网络 && 网络.readyState === WebSocket.CONNECTING) {
      await new Promise((解决) => {
        const 定时 = setInterval(() => {
          if (!网络 || 网络.readyState === WebSocket.OPEN ||
              网络.readyState === WebSocket.CLOSED) {
            clearInterval(定时);
            解决();
          }
        }, 100);
        setTimeout(() => { clearInterval(定时); 解决(); }, 8000);
      });
      if (发送()) { return; }
    }

    // 静默建立全模态会话
    设置状态("连接中");
    try {
      await 建立连接(模型ID, 音色ID, 轮次检测);
    } catch (错误) {
      设置状态("错误");
      显示消息("系统", "对话连接失败：" + 格式化错误(错误));
      return;
    }
    if (发送()) { return; }
    显示消息("系统", "会话未连接，发送失败");
  }

  // ---------------- 初始化 ----------------
  window.通话控制 = {
    开始: 开始,
    挂断: 挂断,
    切换视频: 切换视频,
    切换模型: 切换模型,
    切换音色: 选择音色,
    打开: 打开,
    对话发送: 对话发送,
    状态: "空闲",
    get 模型ID() { return 模型ID; },
    get 音色ID() { return 音色ID; },
  };

  function 初始化() {
    确保容器();
    绑定UI();
    更新模型信息();
    /* 先加载音色并固定默认音色（优先通话可用的定制复刻音色），再构建对话控制条 */
    Promise.resolve().then(async () => {
      try { await 加载音色列表(); } catch (忽略) { /* 音色加载失败不阻塞 */ }
      构建对话控制条().catch(() => { /* 控制条构建失败不阻塞 */ });
    });
    /* 恢复上次选中的麦克风设备，并预授权 + 枚举填充设备下拉（不阻塞界面） */
    try {
      麦克风设备ID = localStorage.getItem("通话麦克风设备ID") || null;
    } catch (忽略) {
      麦克风设备ID = null;
    }
    刷新麦克风列表().catch(() => { /* 枚举失败静默，开始通话时仍可采集默认设备 */ });
    // 聊天框旁"开启通话"按钮：切到通话标签页并直接开始通话
    const 通话按钮 = document.getElementById("通话按钮");
    if (通话按钮) {
      通话按钮.addEventListener("click", 打开);
    }
    // 弹层：点击外部关闭（点在弹层或触发按钮内不关闭）
    document.addEventListener("click", (事件) => {
      const 目标 = 事件.target;
      if (!目标 || !目标.closest) { return; }
      if (
        目标.closest("#通话-模型弹层") || 目标.closest("#通话-音色弹层") ||
        目标.closest("#通话-历史弹层") || 目标.closest("#通话-切换模型") ||
        目标.closest("#通话-切换音色") || 目标.closest("#通话-历史")
      ) { return; }
      关闭弹层();
    });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", 初始化);
  } else {
    初始化();
  }
})();
