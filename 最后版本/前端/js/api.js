/* ============================================================
   一体化全流程AI应用 前端 API 封装层（Task 11）
   ------------------------------------------------------------
   统一封装所有 /api/* fetch 调用：
   - 自动处理 JSON 序列化 / 解析
   - 网络错误、HTTP 错误统一转为 { 错误: "后端接口未就绪：…" }，
     交互型调用自动弹出 toast（静默型用于轮询，不弹）
   - 前端不崩溃：所有接口返回对象，调用方只需检查 结果.错误
   ============================================================ */
"use strict";

const api = {
    /**
     * 核心请求函数
     * @param {string} 路径  - 接口路径，如 "/api/硬件"
     * @param {object} 选项  - { 方法, 数据(JSON对象), 表单(FormData), 静默(bool) }
     */
    async 请求(路径, 选项 = {}) {
        const 方法 = 选项.方法 || "GET";
        const 配置 = { 方法 };
        if (选项.表单) {
            // multipart/form-data：浏览器自动设置 Content-Type 与 boundary
            配置.body = 选项.表单;
        } else if (选项.数据 !== undefined && 选项.数据 !== null) {
            配置.headers = { "Content-Type": "application/json" };
            配置.body = JSON.stringify(选项.数据);
        }

        let 响应;
        try {
            响应 = await fetch(路径, 配置);
        } catch (错误) {
            const 消息 = `后端接口未就绪：${路径}（网络错误）`;
            if (!选项.静默 && window.toast) window.toast(消息, "错误");
            return { 错误: 消息 };
        }

        // 读取响应体（可能是 JSON，也可能是纯文本，如 markdown 报告）
        const 文本 = await 响应.text();
        let 结果;
        try {
            结果 = JSON.parse(文本);
        } catch (错误) {
            结果 = { 原始文本: 文本 };
        }

        if (!响应.ok) {
            let 细节 = "";
            if (结果 && 结果.detail !== undefined) {
                细节 = "：" + (typeof 结果.detail === "string" ? 结果.detail : JSON.stringify(结果.detail));
            }
            const 消息 = `后端接口未就绪：${路径}（HTTP ${响应.status}）${细节}`;
            if (!选项.静默 && window.toast) window.toast(消息, "错误");
            return { 错误: 消息, 状态: 响应.status };
        }
        return 结果;
    },

    // ---------------- 健康 ----------------
    健康() {
        return this.请求("/api/健康");
    },

    // ---------------- 硬件 ----------------
    硬件() {
        return this.请求("/api/硬件");
    },
    硬件状态() {
        return this.请求("/api/硬件/状态", { 静默: true });
    },
    显存预估(参数量亿, 量化) {
        const 参数 = new URLSearchParams({ 参数量亿: String(参数量亿), 量化: String(量化 || "fp16") });
        return this.请求(`/api/硬件/显存预估?${参数}`);
    },

    // ---------------- 配置 ----------------
    获取配置() {
        return this.请求("/api/配置");
    },
    保存配置(配置内容) {
        return this.请求("/api/配置", { 方法: "PUT", 数据: { 配置内容 } });
    },

    // ---------------- 模型管理 ----------------
    模型列表() {
        return this.请求("/api/模型");
    },
    模型扫描() {
        return this.请求("/api/模型/扫描", { 方法: "POST" });
    },
    模型下载(模型ID, 镜像源, 量化) {
        return this.请求("/api/模型/下载", { 方法: "POST", 数据: { 模型ID, 镜像源, 量化 } });
    },
    模型下载进度() {
        return this.请求("/api/模型/下载/进度", { 静默: true });
    },
    模型评估(模型路径) {
        return this.请求("/api/模型/评估", { 方法: "POST", 数据: { 模型路径 } });
    },

    // ---------------- 数据预处理 ----------------
    预处理上传(表单) {
        return this.请求("/api/预处理/上传", { 方法: "POST", 表单 });
    },
    预处理转写(任务ID) {
        return this.请求("/api/预处理/转写", { 方法: "POST", 数据: { 任务ID } });
    },
    预处理转写进度(任务ID) {
        const 参数 = new URLSearchParams({ 任务ID: String(任务ID || "") });
        return this.请求(`/api/预处理/转写/进度?${参数}`, { 静默: true });
    },
    预处理话题分割(任务ID) {
        return this.请求("/api/预处理/话题分割", { 方法: "POST", 数据: { 任务ID } });
    },
    预处理预览(任务ID) {
        const 参数 = new URLSearchParams({ 任务ID: String(任务ID || "") });
        return this.请求(`/api/预处理/预览?${参数}`, { 静默: true });
    },
    预处理调整边界(片段ID, 开始秒, 结束秒) {
        return this.请求("/api/预处理/调整边界", { 方法: "POST", 数据: { 片段ID, 开始秒, 结束秒 } });
    },

    // ---------------- 打标 ----------------
    打标自动(任务ID) {
        return this.请求("/api/打标/自动", { 方法: "POST", 数据: { 任务ID } });
    },
    打标批量(任务ID) {
        return this.请求("/api/打标/批量", { 方法: "POST", 数据: { 任务ID } });
    },
    打标进度(任务ID) {
        const 参数 = new URLSearchParams({ 任务ID: String(任务ID || "") });
        return this.请求(`/api/打标/进度?${参数}`, { 静默: true });
    },
    打标复核(片段ID, 标签) {
        return this.请求("/api/打标/复核", { 方法: "POST", 数据: { 片段ID, 标签 } });
    },
    打标结果(任务ID) {
        const 参数 = new URLSearchParams({ 任务ID: String(任务ID || "") });
        return this.请求(`/api/打标/结果?${参数}`, { 静默: true });
    },
    打标导出(任务ID, 格式, 数据包类型) {
        return this.请求("/api/打标/导出", { 方法: "POST", 数据: { 任务ID, 格式, 数据包类型 } });
    },

    // ---------------- 日记生成 ----------------
    日记规划(人设) {
        return this.请求("/api/日记/规划", { 方法: "POST", 数据: { 人设 } });
    },
    日记生成(人设, 数据目录) {
        return this.请求("/api/日记/生成", { 方法: "POST", 数据: { 人设, 数据目录 } });
    },
    日记进度() {
        return this.请求("/api/日记/进度", { 静默: true });
    },
    日记列表(角色名) {
        const 参数 = new URLSearchParams({ 角色名: String(角色名 || "") });
        return this.请求(`/api/日记/列表?${参数}`);
    },
    日记单篇(角色名, 年龄) {
        const 参数 = new URLSearchParams({ 角色名: String(角色名 || ""), 年龄: String(年龄 ?? "") });
        return this.请求(`/api/日记/单篇?${参数}`);
    },
    日记审阅(角色名, 年龄, 修改后正文) {
        return this.请求("/api/日记/审阅", { 方法: "POST", 数据: { 角色名, 年龄, 修改后正文 } });
    },
    日记导出(角色名) {
        return this.请求("/api/日记/导出", { 方法: "POST", 数据: { 角色名 } });
    },
    日记对话(消息列表) {
        return this.请求("/api/日记/对话", { 方法: "POST", 数据: { 消息列表 } });
    },

    // ---------------- 微调 ----------------
    微调检测(模型路径) {
        return this.请求("/api/微调/检测", { 方法: "POST", 数据: { 模型路径 } });
    },
    微调数据预览(训练配置) {
        return this.请求("/api/微调/数据预览", { 方法: "POST", 数据: 训练配置 });
    },
    微调开始(训练配置) {
        return this.请求("/api/微调/开始", { 方法: "POST", 数据: 训练配置 });
    },
    微调进度() {
        return this.请求("/api/微调/进度", { 静默: true });
    },

    // ---------------- 推理 ----------------
    推理初始化(架构类型, 模型路径, 参数) {
        return this.请求("/api/推理/初始化", { 方法: "POST", 数据: { 架构类型, 模型路径, 参数 } });
    },
    推理生成(提示词, 角色名) {
        return this.请求("/api/推理/生成", { 方法: "POST", 数据: { 提示词, 角色名 } });
    },
    推理释放() {
        return this.请求("/api/推理/释放", { 方法: "POST" });
    },
    参数推荐(模型路径) {
        const 参数 = new URLSearchParams({ 模型路径: String(模型路径 || "") });
        return this.请求(`/api/推理/参数推荐?${参数}`, { 静默: true });
    },
    记忆添加(角色名, 内容, 标签) {
        return this.请求("/api/记忆/添加", { 方法: "POST", 数据: { 角色名, 内容, 标签 } });
    },
    记忆检索(查询, 角色名) {
        const 参数 = new URLSearchParams({ 查询: String(查询 || ""), 角色名: String(角色名 || "") });
        return this.请求(`/api/记忆/检索?${参数}`);
    },

    // ---------------- 达标评估 ----------------
    达标评估(模型路径) {
        return this.请求("/api/达标/评估", { 方法: "POST", 数据: { 模型路径 } });
    },
    达标进度() {
        return this.请求("/api/达标/进度", { 静默: true });
    },
    达标报告(路径) {
        const 参数 = new URLSearchParams({ 路径: String(路径 || "") });
        return this.请求(`/api/达标/报告?${参数}`);
    },
    达标历史() {
        return this.请求("/api/达标/历史", { 静默: true });
    },
};

window.api = api;
