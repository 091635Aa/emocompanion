/* 音频处理器：AudioWorklet 音频采集处理器。
 *
 * 职责：浏览器麦克风（多声道，常见 48kHz/44.1kHz）→ 线性插值下采样到 16kHz
 *       → 单声道 16-bit PCM 字节，通过 port.postMessage(字节, [字节.buffer]) 发到主线程。
 * 主线程 port.onmessage 用于控制：发送 "暂停" / "恢复"。
 *
 * 处理器注册名："音频采集处理器"（AudioWorkletNode 以该名称创建）。
 */
class 音频采集处理器 extends AudioWorkletProcessor {
  constructor() {
    super();
    this.输入采样率 = sampleRate || 48000;   // AudioContext 采样率（麦克风实际采样率）
    this.目标采样率 = 16000;                 // DashScope 实时模型要求的输入采样率
    this.输入位置 = 0;                       // 已处理的输入帧计数（相对游标）
    this.输出累积 = [];                      // 16kHz 采样点累积
    this.发送块大小 = 512;                   // 32ms @16kHz，一个发送块
    this.暂停 = false;
    this.port.onmessage = (事件) => {
      if (事件.data === "暂停") {
        this.暂停 = true;
        this.输出累积 = [];
      } else if (事件.data === "恢复") {
        this.暂停 = false;
      }
    };
  }

  /* 输入 帧们：Float32Array（-1..1）；对每一输入块做线性插值下采样。 */
  process(输入们) {
    if (this.暂停) {
      return true;
    }
    const 输入 = 输入们 && 输入们[0];
    if (!输入 || !输入[0] || 输入[0].length === 0) {
      return true;
    }
    const 帧们 = 输入[0];   // 取第一声道（麦克风声道数 1/2 都兼容）
    const 比例 = this.目标采样率 / this.输入采样率;
    // 本块覆盖的输出采样序号区间 [起始, 结束)（输出域浮点）
    const 起始 = this.输入位置 * 比例;
    const 结束 = (this.输入位置 + 帧们.length) * 比例;
    for (let 输出序号 = Math.ceil(起始); 输出序号 < 结束; 输出序号++) {
      // 输出采样在输入域中的位置（相对本块起始）
      const 输入位置 = 输出序号 / 比例 - this.输入位置;
      const 下索引 = Math.floor(输入位置);
      const 上索引 = Math.min(下索引 + 1, 帧们.length - 1);
      const 权重 = 输入位置 - 下索引;
      this.输出累积.push(帧们[下索引] * (1 - 权重) + 帧们[上索引] * 权重);
    }
    this.输入位置 += 帧们.length;
    if (this.输出累积.length >= this.发送块大小) {
      this.发送();
    }
    return true;
  }

  /* 把累积的 16kHz 采样转成 16-bit 小端 PCM 字节并发送（转移 buffer 所有权）。 */
  发送() {
    const 块 = this.输出累积.splice(0, this.发送块大小);
    if (!块.length) {
      return;
    }
    const 字节 = new ArrayBuffer(块.length * 2);
    const 视图 = new DataView(字节);
    for (let i = 0; i < 块.length; i++) {
      const 采样 = Math.max(-1, Math.min(1, 块[i]));
      视图.setInt16(i * 2, 采样 < 0 ? 采样 * 0x8000 : 采样 * 0x7fff, true);
    }
    this.port.postMessage(字节, [字节]);
  }
}

registerProcessor("音频采集处理器", 音频采集处理器);
