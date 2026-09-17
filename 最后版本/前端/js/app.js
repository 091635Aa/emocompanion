/* ============================================================
   一体化全流程AI应用 前端入口脚本（Task 11）
   ------------------------------------------------------------
   - SPA 页面切换：window.切换页面(名称)
   - 各模块初始化函数：初始化_仪表盘 / 初始化_数据预处理 / …
   - 统一轮询管理：切换页面时停止全部轮询，避免后台浪费请求
   - 动态按钮通过 document 事件委托（[data-动作]）处理
   ============================================================ */
"use strict";

/* ===================== 全局工具 ===================== */

function 转义HTML(文本) {
    return String(文本 ?? "").replace(/[&<>"']/g, (字符) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
    })[字符]);
}

window.toast = function (消息, 类型 = "信息") {
    const 容器 = document.getElementById("toast容器");
    if (!容器) return;
    const 元素 = document.createElement("div");
    元素.className = "toast " + 类型;
    元素.textContent = 消息;
    容器.appendChild(元素);
    setTimeout(() => {
        元素.classList.add("消失");
        setTimeout(() => 元素.remove(), 320);
    }, 3500);
};

function 取任务ID(数据) {
    if (!数据 || typeof 数据 !== "object") return "";
    return 数据.任务ID || 数据.任务编号 || 数据.task_id || 数据.taskId || 数据.id || "";
}

function 取进度(数据) {
    if (!数据 || typeof 数据 !== "object") return null;
    let 值 = 数据.进度 ?? 数据.百分比 ?? 数据.完成度 ?? 数据.progress ?? 数据.percent;
    if (值 === undefined || 值 === null) return null;
    值 = Number(值);
    if (isNaN(值)) return null;
    if (值 > 1) 值 = 值 / 100;
    return Math.max(0, Math.min(1, 值));
}

function 取消息(数据, 默认 = "") {
    return (数据 && (数据.消息 || 数据.message || 数据.提示 || 数据.阶段 || "")) || 默认;
}

function 取字段(对象, 键列表, 默认 = "") {
    if (对象 && typeof 对象 === "object") {
        for (const 键 of 键列表) {
            if (对象[键] !== undefined && 对象[键] !== null) return 对象[键];
        }
    }
    return 默认;
}

function 取数组(数据, 键列表) {
    if (Array.isArray(数据)) return 数据;
    if (数据 && typeof 数据 === "object") {
        for (const 键 of 键列表) {
            if (Array.isArray(数据[键])) return 数据[键];
        }
    }
    return [];
}

function 格式化MB(值) {
    const n = Number(值);
    if (isNaN(n)) return 值;
    if (n >= 1024) return (n / 1024).toFixed(1) + " GB";
    return n.toFixed(0) + " MB";
}

function 格式化秒(秒) {
    const n = Number(秒);
    if (isNaN(n)) return String(秒 ?? "");
    const 分 = Math.floor(n / 60);
    const 秒余 = Math.round(n % 60);
    return 分 > 0 ? 分 + "分" + String(秒余).padStart(2, "0") + "秒" : 秒余 + "秒";
}

function 渲染键值(条目列表) {
    const 行 = 条目列表
        .filter(([, 值]) => 值 !== undefined && 值 !== null && 值 !== "")
        .map(([键, 值]) => '<span class="键">' + 转义HTML(键) + '</span><span class="值">' + 转义HTML(值) + "</span>")
        .join("");
    return 行 || '<p class="占位提示">暂无数据</p>';
}

function 显示进度(填充元素, 文本元素, 进度, 消息) {
    if (填充元素) 填充元素.style.width = Math.round(进度 * 100) + "%";
    if (文本元素) 文本元素.textContent = (消息 ? 消息 + " " : "") + Math.round(进度 * 100) + "%";
}

function 添加消息(容器, 角色, 内容) {
    if (!容器) return;
    const 元素 = document.createElement("div");
    元素.className = "消息 " + (角色 === "用户" ? "用户" : "助手");
    元素.textContent = 内容;
    容器.appendChild(元素);
    容器.scrollTop = 容器.scrollHeight;
}

/* ===================== 轮询管理 ===================== */
const 轮询表 = {};

function 启动轮询(键, 间隔, 回调) {
    停止轮询(键);
    轮询表[键] = setInterval(async () => {
        try {
            await 回调();
        } catch (错误) {
            /* 单次轮询失败不影响整体 */
        }
    }, 间隔);
}

function 停止轮询(键) {
    if (轮询表[键]) {
        clearInterval(轮询表[键]);
        delete 轮询表[键];
    }
}

function 停止全部轮询() {
    Object.keys(轮询表).forEach(停止轮询);
}

/* ===================== 页面切换 ===================== */
const 页面清单 = ["仪表盘", "数据预处理", "打标", "日记生成", "微调", "模型管理", "推理", "达标评估", "配置"];

window.切换页面 = function (名称) {
    if (!页面清单.includes(名称)) return;
    停止全部轮询();
    document.querySelectorAll(".页面").forEach((页) => 页.classList.toggle("当前", 页.id === "页面_" + 名称));
    document.querySelectorAll(".导航项").forEach((项) => 项.classList.toggle("当前", 项.dataset.模块 === 名称));
    const 标题 = document.getElementById("当前页面标题");
    if (标题) 标题.textContent = 名称;
    // 移动端收起侧栏
    const 侧栏 = document.getElementById("侧边栏");
    const 遮罩 = document.getElementById("遮罩");
    if (侧栏) 侧栏.classList.remove("展开");
    if (遮罩) 遮罩.classList.remove("显示");
    // 调用对应初始化函数（顶层 function 声明，挂载于 window）
    const 初始化 = window["初始化_" + 名称];
    if (typeof 初始化 === "function") 初始化();
};

/* ===================== 服务状态（全局轮询） ===================== */
async function 刷新服务状态() {
    const 灯 = document.getElementById("服务状态灯");
    const 文字 = document.getElementById("服务状态文字");
    const 结果 = await api.健康();
    if (结果.错误) {
        if (灯) 灯.className = "服务状态灯 离线";
        if (文字) 文字.textContent = "服务离线";
        return;
    }
    const 版本 = 结果.版本 || "";
    if (灯) 灯.className = "服务状态灯 在线";
    if (文字) 文字.textContent = (结果.状态 === "正常" ? "服务在线" : "服务异常") + (版本 ? " v" + 版本 : "");
    if (版本) {
        const 版本元素1 = document.getElementById("版本徽标");
        const 版本元素2 = document.getElementById("顶栏版本");
        if (版本元素1) 版本元素1.textContent = "v" + 版本;
        if (版本元素2) 版本元素2.textContent = "v" + 版本;
    }
}

/* ===================== 模型下拉（微调/推理共用） ===================== */
async function 加载模型下拉(选择器id) {
    const 选择 = document.getElementById(选择器id);
    if (!选择) return;
    选择.innerHTML = '<option value="">（加载中…）</option>';
    const 结果 = await api.模型列表();
    const 列表 = 结果.错误 ? [] : 取数组(结果, ["模型列表", "模型", "models", "result"]);
    if (!列表.length) {
        选择.innerHTML = '<option value="">（无本地模型，可手动填写路径）</option>';
        return;
    }
    选择.innerHTML = '<option value="">（选择模型）</option>' +
        列表.map((模型) => {
            const 路径 = 模型.本地路径 || 模型.路径 || 模型.path || "";
            const 名称 = 模型.模型ID || 模型.模型名 || 模型.name || 路径;
            return '<option value="' + 转义HTML(路径) + '">' + 转义HTML(名称) + "</option>";
        }).join("");
}

/* ============================================================
   页面 1：仪表盘
   ============================================================ */
async function 初始化_仪表盘() {
    const 硬件信息 = document.getElementById("硬件信息");
    const 服务信息 = document.getElementById("服务信息");
    硬件信息.innerHTML = '<p class="占位提示">正在检测硬件…</p>';
    服务信息.innerHTML = '<p class="占位提示">正在检测服务…</p>';
    const [硬件结果, 健康结果] = await Promise.all([api.硬件(), api.健康()]);
    if (硬件结果.错误) {
        硬件信息.innerHTML = '<p class="占位提示">' + 转义HTML(硬件结果.错误) + "</p>";
    } else {
        硬件信息.innerHTML = 渲染键值([
            ["CUDA 可用", 硬件结果.CUDA可用],
            ["CUDA 版本", 硬件结果.CUDA版本],
            ["GPU 型号", 硬件结果.GPU型号],
            ["驱动版本", 硬件结果.驱动版本],
            ["显存总量", 硬件结果.显存总量MB !== undefined ? 格式化MB(硬件结果.显存总量MB) : ""],
            ["显存可用", 硬件结果.显存可用MB !== undefined ? 格式化MB(硬件结果.显存可用MB) : ""],
            ["内存总量", 硬件结果.内存总量MB !== undefined ? 格式化MB(硬件结果.内存总量MB) : ""],
            ["内存可用", 硬件结果.内存可用MB !== undefined ? 格式化MB(硬件结果.内存可用MB) : ""],
            ["支持运行", 硬件结果.支持运行],
            ["提示", 硬件结果.提示],
        ]);
    }
    if (健康结果.错误) {
        服务信息.innerHTML = '<p class="占位提示">' + 转义HTML(健康结果.错误) + "</p>";
    } else {
        服务信息.innerHTML = 渲染键值([
            ["项目名称", 健康结果.项目名称],
            ["版本", 健康结果.版本],
            ["状态", 健康结果.状态],
        ]);
    }
}

async function 显存预估() {
    const 参数量亿 = document.getElementById("显存参数量亿").value;
    if (!参数量亿) return toast("请先填写模型参数量", "警告");
    const 量化 = document.getElementById("显存量化").value;
    const 结果 = await api.显存预估(参数量亿, 量化);
    const 容器 = document.getElementById("显存预估结果");
    if (结果.错误) {
        容器.innerHTML = '<p class="占位提示">' + 转义HTML(结果.错误) + "</p>";
        return;
    }
    容器.innerHTML = '<div class="键值列表" style="margin-top:12px">' +
        渲染键值([
            ["参数量（亿）", 结果.参数量亿],
            ["量化", 结果.量化],
            ["推理显存", 结果.推理显存MB !== undefined ? 格式化MB(结果.推理显存MB) : ""],
            ["微调显存", 结果.微调显存MB !== undefined ? 格式化MB(结果.微调显存MB) : ""],
            ["可用显存", 结果.可用显存MB !== undefined ? 格式化MB(结果.可用显存MB) : ""],
            ["可推理", 结果.可推理],
            ["可微调", 结果.可微调],
            ["建议", 结果.建议],
        ]) + "</div>";
}

/* ============================================================
   页面 2：数据预处理
   ============================================================ */
let 当前任务ID = "";
let 转写进行中 = false;

function 更新视频解析行() {
    const 行 = document.getElementById("视频解析行");
    const 类型 = (document.querySelector('input[name="上传类型"]:checked') || {}).value;
    行.classList.toggle("隐藏", 类型 !== "视频");
}

async function 上传文件() {
    const 文件输入 = document.getElementById("上传文件");
    const 文件 = 文件输入.files && 文件输入.files[0];
    if (!文件) return toast("请先选择要上传的文件", "警告");
    const 类型 = (document.querySelector('input[name="上传类型"]:checked') || {}).value || "文本";
    const 表单 = new FormData();
    表单.append("文件", 文件);
    表单.append("类型", 类型);
    if (类型 === "视频") {
        表单.append("视频解析", document.getElementById("视频解析").checked ? "true" : "false");
    }
    const 结果 = await api.预处理上传(表单);
    if (结果.错误) return;
    const 任务ID = 取任务ID(结果);
    if (任务ID) {
        当前任务ID = 任务ID;
        document.getElementById("任务ID显示").textContent = 任务ID;
        document.getElementById("打标任务ID").value = 任务ID;
        toast("上传成功：" + (结果.状态 || "已完成"), "成功");
    } else {
        toast("上传成功，但未返回任务ID，请到打标页手动填写", "成功");
    }
}

async function 开始转写() {
    if (!当前任务ID) return toast("请先上传数据获得任务ID", "警告");
    const 结果 = await api.预处理转写(当前任务ID);
    if (结果.错误) return;
    转写进行中 = true;
    启动转写轮询();
    toast("转写任务已提交", "成功");
}

function 启动转写轮询() {
    停止轮询("转写");
    if (!转写进行中 || !当前任务ID) return;
    启动轮询("转写", 1500, async () => {
        const 结果 = await api.预处理转写进度(当前任务ID);
        if (结果.错误) {
            转写进行中 = false;
            停止轮询("转写");
            document.getElementById("转写进度文本").textContent = "转写进度接口未就绪";
            return;
        }
        const 进度 = 取进度(结果);
        const 消息 = 取消息(结果);
        if (进度 !== null) {
            显示进度(document.getElementById("转写进度填充"), document.getElementById("转写进度文本"), 进度, 消息);
        } else {
            document.getElementById("转写进度文本").textContent = 消息 || "转写进行中…";
        }
        if (进度 !== null && 进度 >= 1) {
            转写进行中 = false;
            停止轮询("转写");
            document.getElementById("转写进度文本").textContent = "转写完成 ✓";
            toast("转写完成", "成功");
        }
    });
}

async function 查看转写文本() {
    if (!当前任务ID) return toast("请先上传数据获得任务ID", "警告");
    const 结果 = await api.预处理预览(当前任务ID);
    const 元素 = document.getElementById("转写文本");
    元素.classList.remove("隐藏");
    if (结果.错误) {
        元素.textContent = 结果.错误;
        return;
    }
    元素.textContent = 结果.转写文本 || 结果.文本 || 结果.内容 || 结果.原始文本 || JSON.stringify(结果, null, 2);
}

function 渲染片段卡片(片段, 序号) {
    const 片段ID = 片段.片段ID || 片段.id || ("seg_" + String(序号 + 1).padStart(4, "0"));
    const 开始 = 片段.开始秒 ?? 片段.start ?? 0;
    const 结束 = 片段.结束秒 ?? 片段.end ?? 0;
    const 摘要 = 片段.话题摘要 || 片段.摘要 || 片段.title || 片段.标题 || ("片段 " + (序号 + 1));
    const 文本 = 片段.文本 || 片段.content || 片段.正文 || "";
    return (
        '<div class="片段卡片" data-片段ID="' + 转义HTML(片段ID) + '">' +
            '<div class="片段头部">' +
                '<span class="徽章">#' + (序号 + 1) + '</span>' +
                '<span class="徽章 成功">' + 转义HTML(片段ID) + '</span>' +
                '<span class="片段时间">⏱ ' + 格式化秒(开始) + " ~ " + 格式化秒(结束) + '</span>' +
                '<span class="片段摘要">' + 转义HTML(摘要) + '</span>' +
            "</div>" +
            '<div class="片段文本">' + 转义HTML(文本) + "</div>" +
            '<div class="边界编辑行">' +
                "边界调整 开始秒 <input type=\"number\" class=\"边界开始\" value=\"" + 开始 + '" step="0.1">' +
                "结束秒 <input type=\"number\" class=\"边界结束\" value=\"" + 结束 + '" step="0.1">' +
                '<button class="按钮 次要" data-动作="调整边界" data-片段ID="' + 转义HTML(片段ID) + '">提交边界</button>' +
            "</div>" +
        "</div>"
    );
}

function 显示分割结果(数据) {
    const 容器 = document.getElementById("分割列表");
    const 片段列表 = 取数组(数据, ["片段列表", "分割结果", "片段", "segments", "result"]);
    if (!片段列表.length) {
        容器.innerHTML = '<p class="占位提示">分割结果为空（返回：' + 转义HTML(JSON.stringify(数据)) + "）</p>";
        return;
    }
    容器.innerHTML = 片段列表.map(渲染片段卡片).join("");
    window.最近片段列表 = 片段列表;
}

async function 执行话题分割() {
    if (!当前任务ID) return toast("请先上传数据获得任务ID", "警告");
    const 结果 = await api.预处理话题分割(当前任务ID);
    if (结果.错误) return;
    显示分割结果(结果);
    toast("话题分割完成", "成功");
}

async function 预览分割() {
    if (!当前任务ID) return toast("请先上传数据获得任务ID", "警告");
    const 结果 = await api.预处理预览(当前任务ID);
    if (结果.错误) {
        toast(结果.错误, "错误");
        return;
    }
    显示分割结果(结果);
    toast("已加载分割预览", "成功");
}

async function 提交边界(片段ID, 按钮) {
    const 卡片 = 按钮.closest(".片段卡片");
    if (!卡片) return;
    const 开始 = parseFloat(卡片.querySelector(".边界开始").value) || 0;
    const 结束 = parseFloat(卡片.querySelector(".边界结束").value) || 0;
    const 结果 = await api.预处理调整边界(片段ID, 开始, 结束);
    if (结果.错误) return;
    toast("片段 " + 片段ID + " 边界已调整", "成功");
}

function 初始化_数据预处理() {
    更新视频解析行();
    if (转写进行中) 启动转写轮询();
}

/* ============================================================
   页面 3：打标
   ============================================================ */
let 打标进行中 = false;

async function 加载打标片段() {
    const 任务ID = document.getElementById("打标任务ID").value.trim();
    if (!任务ID) return toast("请填写任务 ID", "警告");
    当前任务ID = 任务ID;
    document.getElementById("任务ID显示").textContent = 任务ID;
    const 结果 = await api.预处理预览(任务ID);
    const 容器 = document.getElementById("打标片段列表");
    if (结果.错误) {
        容器.innerHTML = '<p class="占位提示">' + 转义HTML(结果.错误) + "</p>";
        return;
    }
    const 片段列表 = 取数组(结果, ["片段列表", "分割结果", "片段", "segments"]);
    if (!片段列表.length) {
        const 文本 = 结果.转写文本 || 结果.文本 || 结果.原始文本 || "";
        容器.innerHTML = '<p class="占位提示">' +
            (文本 ? "接口仅返回了转写文本，请先在数据预处理页执行话题分割" : "接口未返回片段列表（空结果）") +
            "</p>";
        return;
    }
    容器.innerHTML = 片段列表.map((片段, i) => {
        const 片段ID = 片段.片段ID || 片段.id || ("seg_" + String(i + 1).padStart(4, "0"));
        return (
            '<div class="片段卡片" data-片段ID="' + 转义HTML(片段ID) + '">' +
                '<div class="片段头部">' +
                    '<span class="徽章 成功">' + 转义HTML(片段ID) + '</span>' +
                    '<span class="片段时间">⏱ ' + 格式化秒(片段.开始秒 ?? 片段.start) + " ~ " + 格式化秒(片段.结束秒 ?? 片段.end) + '</span>' +
                    '<span class="片段摘要">' + 转义HTML(片段.话题摘要 || 片段.摘要 || 片段.title || "片段" + (i + 1)) + '</span>' +
                "</div>" +
                '<div class="片段文本">' + 转义HTML(片段.文本 || 片段.content || "") + "</div>" +
            "</div>"
        );
    }).join("");
    toast("已加载 " + 片段列表.length + " 个片段", "成功");
}

async function 自动打标() {
    const 任务ID = document.getElementById("打标任务ID").value.trim();
    if (!任务ID) return toast("请填写任务 ID", "警告");
    const 结果 = await api.打标批量(任务ID);
    if (结果.错误) return;
    打标进行中 = true;
    启动打标轮询(任务ID);
    toast("自动打标任务已提交", "成功");
}

function 启动打标轮询(任务ID) {
    停止轮询("打标");
    if (!打标进行中 || !任务ID) return;
    启动轮询("打标", 1500, async () => {
        const 结果 = await api.打标进度(任务ID);
        if (结果.错误) {
            打标进行中 = false;
            停止轮询("打标");
            document.getElementById("打标进度文本").textContent = "打标进度接口未就绪";
            return;
        }
        const 进度 = 取进度(结果);
        const 消息 = 取消息(结果);
        if (进度 !== null) {
            显示进度(document.getElementById("打标进度填充"), document.getElementById("打标进度文本"), 进度, 消息);
        } else {
            document.getElementById("打标进度文本").textContent = 消息 || "打标进行中…";
        }
        if (进度 !== null && 进度 >= 1) {
            打标进行中 = false;
            停止轮询("打标");
            document.getElementById("打标进度文本").textContent = "打标完成 ✓";
            toast("打标完成", "成功");
            加载打标结果(任务ID);
        }
    });
}

async function 加载打标结果(任务ID) {
    const 结果 = await api.打标结果(任务ID);
    const 容器 = document.getElementById("打标结果列表");
    if (结果.错误) {
        容器.innerHTML = '<p class="占位提示">' + 转义HTML(结果.错误) + "</p>";
        return;
    }
    const 结果列表 = 取数组(结果, ["打标结果", "结果", "片段列表", "items", "result"]);
    if (!结果列表.length) {
        容器.innerHTML = '<p class="占位提示">打标结果为空</p>';
        return;
    }
    容器.innerHTML = 结果列表.map((项, i) => {
        const 片段ID = 项.片段ID || 项.片段id || 项.id || ("seg_" + String(i + 1).padStart(4, "0"));
        const 情感维度 = 项.情感维度 || "";
        const 情感标签 = 项.情感标签 || 情感维度 || "";
        const 内容标签 = Array.isArray(项.内容标签) ? 项.内容标签.join(",") : (项.内容标签 || "");
        const 风格标签 = Array.isArray(项.风格标签) ? 项.风格标签.join(",") : (项.风格标签 || "");
        const 文本 = 项.文本 || 项.content || 项.片段文本 || "";
        const 摘要 = 文本 ? 文本.slice(0, 60) : (项.话题摘要 || "");
        return (
            '<div class="片段卡片" data-片段ID="' + 转义HTML(片段ID) + '">' +
                '<div class="片段头部">' +
                    '<span class="徽章 成功">' + 转义HTML(片段ID) + '</span>' +
                    (项.置信度 !== undefined ? '<span class="徽章 警告">置信度 ' + 转义HTML(项.置信度) + "</span>" : "") +
                    '<span class="片段摘要">' + 转义HTML(摘要) + '</span>' +
                "</div>" +
                '<div class="打标表单">' +
                    '<label>情感维度<input class="标签情感维度" value="' + 转义HTML(情感维度) + '" placeholder="积极/消极/中性"></label>' +
                    '<label>情感标签<input class="标签情感" value="' + 转义HTML(情感标签) + '"></label>' +
                    '<label>内容标签<input class="标签内容" value="' + 转义HTML(内容标签) + '" placeholder="逗号分隔"></label>' +
                    '<label>风格标签<input class="标签风格" value="' + 转义HTML(风格标签) + '" placeholder="逗号分隔"></label>' +
                "</div>" +
                '<button class="按钮 主要" data-动作="打标复核" data-片段ID="' + 转义HTML(片段ID) + '">✅ 复核提交</button>' +
            "</div>"
        );
    }).join("");
}

async function 提交打标复核(片段ID, 按钮) {
    const 卡片 = 按钮.closest(".片段卡片");
    if (!卡片) return;
    const 标签 = {
        情感维度: 卡片.querySelector(".标签情感维度").value.trim(),
        情感标签: 卡片.querySelector(".标签情感").value.trim(),
        内容标签: 卡片.querySelector(".标签内容").value.split(/[,，]/).map((s) => s.trim()).filter(Boolean),
        风格标签: 卡片.querySelector(".标签风格").value.split(/[,，]/).map((s) => s.trim()).filter(Boolean),
    };
    const 结果 = await api.打标复核(片段ID, 标签);
    if (结果.错误) return;
    toast("片段 " + 片段ID + " 复核已提交", "成功");
}

async function 导出打标数据包() {
    const 任务ID = document.getElementById("打标任务ID").value.trim();
    if (!任务ID) return toast("请填写任务 ID", "警告");
    const 格式 = document.getElementById("导出格式").value;
    const 数据包类型 = document.getElementById("导出类型").value;
    const 结果 = await api.打标导出(任务ID, 格式, 数据包类型);
    const 容器 = document.getElementById("导出结果");
    if (结果.错误) {
        容器.innerHTML = '<p class="占位提示">' + 转义HTML(结果.错误) + "</p>";
        return;
    }
    容器.innerHTML = '<div class="键值列表" style="margin-top:12px">' +
        渲染键值([
            ["格式", 结果.格式],
            ["文件路径", 结果.文件路径],
            ["条目数", 结果.条目数],
            ["状态", 结果.状态],
        ]) + "</div>";
    toast("数据包导出成功", "成功");
}

function 初始化_打标() {
    if (打标进行中) {
        const 任务ID = document.getElementById("打标任务ID").value.trim();
        启动打标轮询(任务ID);
    }
}

/* ============================================================
   页面 4：日记生成
   ============================================================ */
let 日记生成进行中 = false;
const 日记对话历史 = [];

function 收集人设() {
    return {
        姓名: document.getElementById("人设姓名").value.trim(),
        性别: document.getElementById("人设性别").value,
        出生年份: document.getElementById("人设出生年份").value.trim(),
        人设描述: document.getElementById("人设描述").value.trim(),
        关键经历: document.getElementById("人设经历").value.trim(),
    };
}

async function 规划时间线() {
    const 人设 = 收集人设();
    if (!人设.姓名) return toast("请填写姓名", "警告");
    const 结果 = await api.日记规划(人设);
    const 容器 = document.getElementById("时间线结果");
    if (结果.错误) {
        容器.innerHTML = '<p class="占位提示">' + 转义HTML(结果.错误) + "</p>";
        return;
    }
    const 时间线 = 取数组(结果, ["时间线", "规划", "主题", "timeline", "规划结果"]);
    if (!时间线.length) {
        容器.innerHTML = '<p class="占位提示">规划完成：' + 转义HTML(JSON.stringify(结果)) + "</p>";
        return;
    }
    容器.innerHTML = '<h4 class="配置分组标题" style="margin-top:14px">📐 各年龄主题</h4>' +
        时间线.map((项) => {
            const 年龄 = 项.年龄 ?? 项.岁 ?? 项.year ?? "";
            const 主题 = 项.主题 || 项.摘要 || 项.topic || 项.标题 || "";
            return '<div class="片段卡片"><div class="片段头部">' +
                '<span class="徽章 成功">' + 转义HTML(年龄 ? 年龄 + "岁" : "?") + '</span>' +
                '<span class="片段摘要">' + 转义HTML(主题) + "</span></div></div>";
        }).join("");
    toast("时间线规划完成", "成功");
}

async function 开始生成日记() {
    const 人设 = 收集人设();
    if (!人设.姓名) return toast("请填写姓名", "警告");
    const 数据目录 = document.getElementById("日记数据目录").value.trim();
    const 结果 = await api.日记生成(人设, 数据目录);
    if (结果.错误) return;
    日记生成进行中 = true;
    启动日记轮询();
    toast("日记生成任务已提交", "成功");
}

function 启动日记轮询() {
    停止轮询("日记");
    if (!日记生成进行中) return;
    启动轮询("日记", 2000, async () => {
        const 结果 = await api.日记进度();
        if (结果.错误) {
            日记生成进行中 = false;
            停止轮询("日记");
            document.getElementById("日记进度文本").textContent = "日记进度接口未就绪";
            return;
        }
        const 进度 = 取进度(结果);
        const 消息 = 取消息(结果);
        if (进度 !== null) {
            显示进度(document.getElementById("日记进度填充"), document.getElementById("日记进度文本"), 进度, 消息);
        } else {
            document.getElementById("日记进度文本").textContent = 消息 || "日记生成中…";
        }
        if (进度 !== null && 进度 >= 1) {
            日记生成进行中 = false;
            停止轮询("日记");
            document.getElementById("日记进度文本").textContent = "日记生成完成 ✓";
            toast("日记生成完成", "成功");
            加载日记列表();
        }
    });
}

async function 加载日记列表() {
    let 角色名 = document.getElementById("日记角色名").value.trim();
    if (!角色名) {
        角色名 = document.getElementById("人设姓名").value.trim();
        if (角色名) document.getElementById("日记角色名").value = 角色名;
        else return toast("请填写角色名", "警告");
    }
    const 结果 = await api.日记列表(角色名);
    const 容器 = document.getElementById("日记列表");
    if (结果.错误) {
        容器.innerHTML = '<p class="占位提示">' + 转义HTML(结果.错误) + "</p>";
        return;
    }
    const 列表 = 取数组(结果, ["日记列表", "列表", "日记", "items", "result"]);
    if (!列表.length) {
        容器.innerHTML = '<p class="占位提示">暂无日记，请先生成。</p>';
        return;
    }
    容器.innerHTML = 列表.map((篇, i) => {
        const 年龄 = 篇.年龄 ?? 篇.岁 ?? "";
        const 日期 = 篇.日期 || 篇.date || "";
        const 标题 = 篇.标题 || 篇.title || ("日记 " + (i + 1));
        const 状态 = 篇.状态 || "";
        return (
            '<div class="片段卡片">' +
                '<div class="片段头部">' +
                    (年龄 ? '<span class="徽章 成功">' + 转义HTML(年龄) + "岁</span>" : "") +
                    (日期 ? '<span class="徽章">' + 转义HTML(日期) + "</span>" : "") +
                    '<span class="片段摘要">' + 转义HTML(标题) + "</span>" +
                    (状态 ? '<span class="徽章 警告">' + 转义HTML(状态) + "</span>" : "") +
                    '<button class="按钮 次要" data-动作="日记打开" data-角色名="' + 转义HTML(角色名) + '" data-年龄="' + 转义HTML(年龄) + '">查看/审阅</button>' +
                "</div>" +
            "</div>"
        );
    }).join("");
}

async function 打开日记单篇(角色名, 年龄) {
    const 结果 = await api.日记单篇(角色名, 年龄);
    const 容器 = document.getElementById("日记详情");
    if (结果.错误) {
        容器.innerHTML = '<p class="占位提示">' + 转义HTML(结果.错误) + "</p>";
        return;
    }
    const 篇 = 结果.篇 || 结果.日记 || 结果;
    const 正文 = 篇.正文 || 篇.内容 || 篇.content || 篇.原始文本 || "";
    const 标题 = 篇.标题 || 篇.title || (年龄 ? 年龄 + "岁日记" : "日记");
    容器.innerHTML =
        '<div class="片段卡片">' +
            '<div class="片段头部">' +
                '<span class="徽章 成功">' + 转义HTML(年龄) + "岁</span>" +
                '<span class="片段摘要">' + 转义HTML(标题) + "</span>" +
            "</div>" +
            '<div class="表单组" style="margin:10px 0">' +
                '<label for="日记审阅正文">正文（可直接编辑后审阅提交）</label>' +
                '<textarea id="日记审阅正文" rows="10">' + 转义HTML(正文) + "</textarea>" +
            "</div>" +
            '<button class="按钮 主要" data-动作="日记审阅" data-角色名="' + 转义HTML(角色名) + '" data-年龄="' + 转义HTML(年龄) + '">✅ 审阅提交</button>' +
        "</div>";
}

async function 提交日记审阅(角色名, 年龄) {
    const 正文 = document.getElementById("日记审阅正文").value;
    if (!正文.trim()) return toast("正文不能为空", "警告");
    const 结果 = await api.日记审阅(角色名, 年龄, 正文);
    if (结果.错误) return;
    toast("审阅已提交", "成功");
    加载日记列表();
}

async function 导出日记文本() {
    const 角色名 = document.getElementById("日记角色名").value.trim();
    if (!角色名) return toast("请填写角色名", "警告");
    const 结果 = await api.日记导出(角色名);
    const 容器 = document.getElementById("日记导出结果");
    if (结果.错误) {
        容器.innerHTML = '<p class="占位提示">' + 转义HTML(结果.错误) + "</p>";
        return;
    }
    容器.innerHTML = '<div class="键值列表" style="margin-top:12px">' +
        渲染键值([
            ["角色名", 角色名],
            ["文件路径", 结果.文件路径 || 结果.路径],
            ["篇数", 结果.篇数 || 结果.条目数],
            ["状态", 结果.状态],
        ]) + "</div>";
    toast("日记导出成功", "成功");
}

async function 发送日记消息() {
    const 输入 = document.getElementById("日记聊天输入");
    const 内容 = 输入.value.trim();
    if (!内容) return;
    输入.value = "";
    日记对话历史.push({ 角色: "用户", 内容 });
    添加消息(document.getElementById("日记聊天记录"), "用户", 内容);
    const 结果 = await api.日记对话(日记对话历史);
    if (结果.错误) return;
    const 回复 = 结果.回复 || 结果.内容 || 结果.回答 || 结果.原始文本 || JSON.stringify(结果);
    日记对话历史.push({ 角色: "助手", 内容: 回复 });
    添加消息(document.getElementById("日记聊天记录"), "助手", 回复);
}

function 初始化_日记生成() {
    if (日记生成进行中) 启动日记轮询();
}

/* ============================================================
   页面 5：微调
   ============================================================ */
let 微调进行中 = false;

function 更新维度组合提示() {
    const 提示 = document.getElementById("维度组合提示");
    const 勾选 = [];
    if (document.getElementById("勾选情感").checked) 勾选.push("情感");
    if (document.getElementById("勾选记忆").checked) 勾选.push("记忆");
    if (document.getElementById("勾选身份").checked) 勾选.push("身份");
    提示.textContent = 勾选.length ? "本次微调组合：[" + 勾选.join(" + ") + "]" : "未勾选任何维度！";
}

function 取微调模型路径() {
    const 手填 = document.getElementById("微调模型路径").value.trim();
    if (手填) return 手填;
    return document.getElementById("微调模型选择").value;
}

async function 检测模型能力() {
    const 模型路径 = 取微调模型路径();
    if (!模型路径) return toast("请先选择或填写模型路径", "警告");
    const 结果 = await api.微调检测(模型路径);
    const 容器 = document.getElementById("微调检测结果");
    if (结果.错误) {
        容器.innerHTML = '<p class="占位提示">' + 转义HTML(结果.错误) + "</p>";
        return;
    }
    容器.innerHTML = '<div class="键值列表" style="margin-top:12px">' +
        渲染键值([
            ["模型路径", 结果.模型路径 || 模型路径],
            ["架构", 结果.架构],
            ["支持思考模式", 结果.支持思考模式],
            ["推荐量化", 结果.推荐量化],
            ["chat模板", 结果.chat模板],
            ["target_modules", Array.isArray(结果.target_modules) ? 结果.target_modules.join(", ") : 结果.target_modules],
            ["建议策略", 结果.建议策略],
        ]) + "</div>";
    toast("模型能力检测完成", "成功");
}

function 收集训练配置() {
    const 基座模型路径 = 取微调模型路径();
    const 角色名 = document.getElementById("微调角色名").value.trim();
    if (!基座模型路径) {
        toast("请先选择或填写模型路径", "警告");
        return null;
    }
    if (!角色名) {
        toast("请填写角色名", "警告");
        return null;
    }
    const 组织方式 = (document.querySelector('input[name="组织方式"]:checked') || {}).value || "一角色一模型";
    return {
        基座模型路径,
        启用情感微调: document.getElementById("勾选情感").checked,
        启用记忆微调: document.getElementById("勾选记忆").checked,
        启用身份微调: document.getElementById("勾选身份").checked,
        轮数: parseInt(document.getElementById("微调轮数").value) || 3,
        学习率: parseFloat(document.getElementById("微调学习率").value) || 0.0002,
        批量: parseInt(document.getElementById("微调批量").value) || 4,
        量化: document.getElementById("微调量化").value,
        角色名,
        模型组织方式: 组织方式,
    };
}

async function 数据预览() {
    const 配置 = 收集训练配置();
    if (!配置) return;
    const 结果 = await api.微调数据预览(配置);
    const 容器 = document.getElementById("微调数据预览结果");
    if (结果.错误) {
        容器.innerHTML = '<p class="占位提示">' + 转义HTML(结果.错误) + "</p>";
        return;
    }
    const 条数 = 结果.各维度条数 || 结果.数据条数 || 结果.counts || 结果;
    容器.innerHTML = '<div class="键值列表" style="margin-top:12px">' +
        渲染键值([
            ["情感维度", 取字段(条数, ["情感", "情感维度", "情感微调"])],
            ["记忆维度", 取字段(条数, ["记忆", "记忆维度", "记忆微调"])],
            ["身份维度", 取字段(条数, ["身份", "身份维度", "身份微调"])],
            ["总计", 取字段(结果, ["总计", "总条数", "total"])],
            ["状态", 结果.状态],
        ]) + "</div>";
    toast("数据预览完成", "成功");
}

async function 开始微调() {
    const 配置 = 收集训练配置();
    if (!配置) return;
    // 实验性警告：三维合一单模型 + 勾选记忆/身份（多记忆承载）
    if (配置.模型组织方式 === "三维合一单模型" && (配置.启用记忆微调 || 配置.启用身份微调)) {
        const 继续 = confirm("⚠️ 单模型承载多记忆属实验性，不推荐。确定继续吗？");
        if (!继续) return;
    }
    const 结果 = await api.微调开始(配置);
    if (结果.错误) return;
    微调进行中 = true;
    启动微调轮询();
    toast("微调任务已提交", "成功");
}

function 启动微调轮询() {
    停止轮询("微调");
    if (!微调进行中) return;
    启动轮询("微调", 2500, async () => {
        const 结果 = await api.微调进度();
        if (结果.错误) {
            微调进行中 = false;
            停止轮询("微调");
            document.getElementById("微调进度文本").textContent = "微调进度接口未就绪";
            return;
        }
        const 进度 = 取进度(结果);
        const 消息 = 取消息(结果);
        const 阶段 = 结果.阶段 || 结果.stage || "";
        if (阶段) document.getElementById("微调阶段").textContent = "阶段：" + 阶段;
        if (进度 !== null) {
            显示进度(document.getElementById("微调进度填充"), document.getElementById("微调进度文本"), 进度, 消息);
        } else {
            document.getElementById("微调进度文本").textContent = 消息 || "训练中…";
        }
        const 日志 = 结果.日志 || 结果.log || "";
        if (日志) {
            const 日志元素 = document.getElementById("微调日志");
            日志元素.textContent = typeof 日志 === "string" ? 日志 : JSON.stringify(日志, null, 2);
            日志元素.scrollTop = 日志元素.scrollHeight;
        }
        if (进度 !== null && 进度 >= 1) {
            微调进行中 = false;
            停止轮询("微调");
            document.getElementById("微调进度文本").textContent = "微调完成 ✓";
            toast("微调完成", "成功");
        }
    });
}

function 初始化_微调() {
    更新维度组合提示();
    加载模型下拉("微调模型选择");
    if (微调进行中) 启动微调轮询();
}

/* ============================================================
   页面 6：模型管理
   ============================================================ */
let 下载进行中 = false;

async function 加载模型列表() {
    const 结果 = await api.模型列表();
    const tbody = document.querySelector("#模型列表 tbody");
    if (!tbody) return;
    if (结果.错误) {
        tbody.innerHTML = '<tr><td colspan="6" class="占位提示">' + 转义HTML(结果.错误) + "</td></tr>";
        return;
    }
    const 列表 = 取数组(结果, ["模型列表", "模型", "models", "result"]);
    if (!列表.length) {
        tbody.innerHTML = '<tr><td colspan="6" class="占位提示">暂无本地模型，可扫描或下载。</td></tr>';
        return;
    }
    tbody.innerHTML = 列表.map((模型) => {
        const 路径 = 模型.本地路径 || 模型.路径 || 模型.path || "";
        const 名称 = 模型.模型ID || 模型.模型名 || 模型.name || 路径;
        return (
            "<tr>" +
                "<td>" + 转义HTML(名称) + "</td>" +
                "<td>" + 转义HTML(模型.参数量亿 ?? "") + (模型.参数量亿 ? "B" : "") + "</td>" +
                "<td>" + 转义HTML(模型.架构 || "") + "</td>" +
                "<td>" + 转义HTML(模型.量化档位 || 模型.量化 || "") + "</td>" +
                '<td class="模型路径">' + 转义HTML(路径) + "</td>" +
                '<td><button class="按钮 次要" data-动作="模型评估" data-路径="' + 转义HTML(路径) + '">性能评估</button></td>' +
            "</tr>"
        );
    }).join("");
}

async function 模型扫描() {
    const 提示 = document.getElementById("模型扫描结果");
    提示.textContent = "扫描中…";
    const 结果 = await api.模型扫描();
    if (结果.错误) {
        提示.textContent = 结果.错误;
        return;
    }
    提示.textContent = "扫描完成：" + (结果.消息 || 结果.状态 || "已更新模型库");
    toast("模型库扫描完成", "成功");
    加载模型列表();
}

async function 下载模型() {
    const 模型ID = document.getElementById("下载模型ID").value.trim();
    if (!模型ID) return toast("请填写模型 ID", "警告");
    const 镜像源 = document.getElementById("下载镜像源").value;
    const 量化 = document.getElementById("下载量化").value;
    const 结果 = await api.模型下载(模型ID, 镜像源, 量化);
    if (结果.错误) return;
    下载进行中 = true;
    启动下载轮询();
    toast("下载任务已提交", "成功");
}

function 启动下载轮询() {
    停止轮询("下载");
    if (!下载进行中) return;
    启动轮询("下载", 2000, async () => {
        const 结果 = await api.模型下载进度();
        if (结果.错误) {
            下载进行中 = false;
            停止轮询("下载");
            document.getElementById("下载进度文本").textContent = "下载进度接口未就绪";
            return;
        }
        const 进度 = 取进度(结果);
        const 消息 = 取消息(结果);
        const 已下载 = 结果.已下载MB ?? 结果.已下载;
        const 总计 = 结果.总计MB ?? 结果.总计;
        if (进度 !== null) {
            显示进度(document.getElementById("下载进度填充"), document.getElementById("下载进度文本"), 进度, 消息);
        } else if (已下载 !== undefined) {
            document.getElementById("下载进度文本").textContent = 消息 || (已下载 + " MB / " + (总计 ?? "?") + " MB");
        } else {
            document.getElementById("下载进度文本").textContent = 消息 || "下载中…";
        }
        if (进度 !== null && 进度 >= 1) {
            下载进行中 = false;
            停止轮询("下载");
            document.getElementById("下载进度文本").textContent = "下载完成 ✓";
            toast("模型下载完成", "成功");
            加载模型列表();
        }
    });
}

async function 模型性能评估(路径) {
    if (!路径) return toast("模型路径为空", "警告");
    const 结果 = await api.模型评估(路径);
    const 容器 = document.getElementById("模型评估结果");
    if (结果.错误) {
        容器.innerHTML = '<p class="占位提示">' + 转义HTML(结果.错误) + "</p>";
        return;
    }
    容器.innerHTML = '<h4 class="配置分组标题" style="margin-top:14px">📋 评估报告</h4><div class="键值列表">' +
        渲染键值([
            ["模型路径", 结果.模型路径 || 路径],
            ["显存占用", 结果.显存占用MB !== undefined ? 格式化MB(结果.显存占用MB) : ""],
            ["生成速度", 结果.生成速度Token每秒 !== undefined ? 结果.生成速度Token每秒 + " tok/s" : ""],
            ["可推理", 结果.可推理],
            ["可微调", 结果.可微调],
            ["评估时间", 结果.评估时间秒 !== undefined ? 结果.评估时间秒 + " 秒" : ""],
            ["备注", 结果.备注],
        ]) + "</div>";
    toast("性能评估完成", "成功");
}

function 初始化_模型管理() {
    加载模型列表();
    if (下载进行中) 启动下载轮询();
}

/* ============================================================
   页面 7：推理
   ============================================================ */
let 推理已初始化 = false;

async function 初始化推理() {
    const 架构类型 = (document.querySelector('input[name="架构类型"]:checked') || {}).value || "V通用架构";
    const 模型路径 = document.getElementById("推理模型选择").value;
    if (!模型路径) return toast("请选择模型", "警告");
    const 参数 = {
        架构: 架构类型,
        λ: parseFloat(document.getElementById("推理λ").value),
        γ: parseFloat(document.getElementById("推理γ").value),
        τ: parseFloat(document.getElementById("推理τ").value),
        max_new_tokens: parseInt(document.getElementById("推理max").value) || 256,
    };
    const 结果 = await api.推理初始化(架构类型, 模型路径, 参数);
    const 状态 = document.getElementById("推理状态");
    if (结果.错误) {
        状态.textContent = "初始化失败";
        return;
    }
    推理已初始化 = true;
    状态.textContent = 结果.状态 || "就绪";
    toast("推理引擎初始化完成", "成功");
}

async function 参数自动推荐() {
    const 模型路径 = document.getElementById("推理模型选择").value;
    if (!模型路径) return toast("请先选择模型", "警告");
    const 结果 = await api.参数推荐(模型路径);
    if (结果.错误) return;
    const λ = 结果.λ ?? 结果.默认λ ?? 结果.lambda;
    const γ = 结果.γ ?? 结果.默认γ ?? 结果.gamma;
    const τ = 结果.τ ?? 结果.默认τ ?? 结果.tau ?? 结果.温度;
    const max = 结果.max_new_tokens ?? 结果.maxTokens ?? 结果.最大新Token;
    if (λ !== undefined) document.getElementById("推理λ").value = λ;
    if (γ !== undefined) document.getElementById("推理γ").value = γ;
    if (τ !== undefined) document.getElementById("推理τ").value = τ;
    if (max !== undefined) document.getElementById("推理max").value = max;
    toast("参数推荐已应用", "成功");
}

async function 发送推理消息() {
    const 输入 = document.getElementById("推理输入");
    const 内容 = 输入.value.trim();
    if (!内容) return;
    输入.value = "";
    const 容器 = document.getElementById("推理聊天记录");
    添加消息(容器, "用户", 内容);
    const 角色名 = document.getElementById("推理角色名").value.trim();
    const 结果 = await api.推理生成(内容, 角色名);
    if (结果.错误) return;
    const 回复 = 结果.回复 || 结果.回答 || 结果.内容 || 结果.原始文本 || JSON.stringify(结果);
    添加消息(容器, "助手", 回复);
    const 指标行 = document.getElementById("推理指标");
    const 指标 = [];
    if (结果.语义熵 !== undefined) 指标.push(["语义熵", 结果.语义熵]);
    if (结果.重复率 !== undefined) 指标.push(["重复率", 结果.重复率]);
    if (结果.情感命中率 !== undefined) 指标.push(["情感命中率", 结果.情感命中率]);
    if (结果.耗时秒 !== undefined) 指标.push(["耗时", 结果.耗时秒 + "s"]);
    if (结果.显存占用MB !== undefined) 指标.push(["显存", 格式化MB(结果.显存占用MB)]);
    if (结果.回响命中 !== undefined) 指标.push(["回响命中", 结果.回响命中]);
    指标行.innerHTML = 指标.map(([键, 值]) => '<span class="指标">' + 转义HTML(键) + " <b>" + 转义HTML(值) + "</b></span>").join("");
}

async function 添加记忆() {
    const 内容 = document.getElementById("记忆内容").value.trim();
    if (!内容) return toast("请填写记忆内容", "警告");
    const 角色名 = document.getElementById("推理角色名").value.trim();
    const 标签 = document.getElementById("记忆标签").value.trim();
    const 结果 = await api.记忆添加(角色名, 内容, 标签);
    if (结果.错误) return;
    document.getElementById("记忆内容").value = "";
    document.getElementById("记忆标签").value = "";
    toast("记忆已添加", "成功");
}

async function 检索记忆() {
    const 查询 = document.getElementById("记忆查询").value.trim();
    if (!查询) return toast("请填写检索查询", "警告");
    const 角色名 = document.getElementById("推理角色名").value.trim();
    const 结果 = await api.记忆检索(查询, 角色名);
    const 容器 = document.getElementById("记忆检索结果");
    if (结果.错误) {
        容器.innerHTML = '<p class="占位提示">' + 转义HTML(结果.错误) + "</p>";
        return;
    }
    const 条目 = 取数组(结果, ["结果", "记忆", "条目", "items", "memories", "检索结果"]);
    容器.innerHTML = 条目.length
        ? 条目.map((记忆) => '<div class="片段卡片"><div class="片段文本">' +
            转义HTML(记忆.内容 || 记忆.content || 记忆.文本 || JSON.stringify(记忆)) + "</div></div>").join("")
        : '<p class="占位提示">未检索到相关记忆</p>';
}

function 初始化_推理() {
    加载模型下拉("推理模型选择");
}

/* ============================================================
   页面 8：达标评估
   ============================================================ */
let 达标进行中 = false;

async function 开始达标评估() {
    const 路径 = document.getElementById("达标模型路径").value.trim();
    if (!路径) return toast("请填写模型路径", "警告");
    const 结果 = await api.达标评估(路径);
    if (结果.错误) return;
    达标进行中 = true;
    启动达标轮询();
    toast("评估任务已提交", "成功");
    // 若后端同步返回了评估结果
    if (结果.各项得分 || 结果.综合均分) 显示达标结果(结果);
}

function 启动达标轮询() {
    停止轮询("达标");
    if (!达标进行中) return;
    启动轮询("达标", 2000, async () => {
        const 结果 = await api.达标进度();
        if (结果.错误) {
            达标进行中 = false;
            停止轮询("达标");
            document.getElementById("达标进度文本").textContent = "达标进度接口未就绪";
            return;
        }
        const 进度 = 取进度(结果);
        const 消息 = 取消息(结果);
        if (进度 !== null) {
            显示进度(document.getElementById("达标进度填充"), document.getElementById("达标进度文本"), 进度, 消息);
        } else {
            document.getElementById("达标进度文本").textContent = 消息 || "评估中…";
        }
        if (进度 !== null && 进度 >= 1) {
            达标进行中 = false;
            停止轮询("达标");
            document.getElementById("达标进度文本").textContent = "评估完成 ✓";
            toast("评估完成", "成功");
            if (结果.各项得分 || 结果.综合均分) 显示达标结果(结果);
            if (结果.报告路径) 加载达标报告(结果.报告路径);
            加载达标历史();
        }
    });
}

function 显示达标结果(结果) {
    const 容器 = document.getElementById("达标评估结果");
    const 各项 = 结果.各项得分 || 结果.得分 || {};
    const 门槛 = Number(结果.门槛 ?? 0.5);
    const 行 = Object.entries(各项).map(([键, 值]) =>
        '<span class="徽章 ' + (Number(值) >= 门槛 ? "成功" : "警告") + '">' + 转义HTML(键) + "：" + 转义HTML(值) + "</span>"
    ).join("");
    容器.innerHTML = '<div style="margin-top:12px">' + 行 +
        '<div class="键值列表" style="margin-top:10px">' +
        渲染键值([
            ["综合均分", 结果.综合均分],
            ["达标", 结果.达标],
            ["门槛", 结果.门槛],
            ["报告路径", 结果.报告路径],
        ]) + "</div></div>";
}

function 渲染Markdown(md) {
    if (!md) return '<p class="占位提示">（空报告）</p>';
    let html = 转义HTML(md);
    // 代码块
    html = html.replace(/```([\s\S]*?)```/g, (m, code) => "<pre><code>" + code.trim() + "</code></pre>");
    // 标题
    html = html.replace(/^#### (.*)$/gm, "<h4>$1</h4>");
    html = html.replace(/^### (.*)$/gm, "<h3>$1</h3>");
    html = html.replace(/^## (.*)$/gm, "<h2>$1</h2>");
    html = html.replace(/^# (.*)$/gm, "<h1>$1</h1>");
    // 列表
    html = html.replace(/^\s*[-*] (.*)$/gm, "<li>$1</li>");
    html = html.replace(/((?:<li>.*?<\/li>\s*)+)/gs, "<ul>$1</ul>");
    // 引用（转义后 > 为 &gt;）
    html = html.replace(/^&gt; (.*)$/gm, "<blockquote>$1</blockquote>");
    // 粗体
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    // 行内代码
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    // 换行
    html = html.replace(/\n/g, "<br>");
    return html;
}

async function 加载达标报告(路径) {
    const 结果 = await api.达标报告(路径);
    const 容器 = document.getElementById("达标报告");
    if (结果.错误) {
        容器.innerHTML = '<p class="占位提示">' + 转义HTML(结果.错误) + "</p>";
        return;
    }
    let markdown = 结果.报告 || 结果.内容 || 结果.markdown || 结果.原始文本 || "";
    if (!markdown) markdown = JSON.stringify(结果, null, 2);
    容器.innerHTML = '<div class="报告区">' + 渲染Markdown(String(markdown)) + "</div>";
}

async function 加载达标历史() {
    const 结果 = await api.达标历史();
    const 容器 = document.getElementById("达标历史列表");
    if (结果.错误) {
        容器.innerHTML = '<p class="占位提示">' + 转义HTML(结果.错误) + "</p>";
        return;
    }
    const 历史 = 取数组(结果, ["历史", "历史列表", "items", "result"]);
    if (!历史.length) {
        容器.innerHTML = '<p class="占位提示">暂无历史记录</p>';
        return;
    }
    容器.innerHTML = 历史.map((项) => {
        const 路径 = 项.模型路径 || 项.路径 || "";
        const 均分 = 项.综合均分 ?? 项.均分;
        const 达标 = 项.达标;
        const 时间 = 项.评估时间 || 项.时间 || "";
        return '<div class="片段卡片"><div class="片段头部">' +
            '<span class="徽章 ' + (达标 ? "成功" : "危险") + '">' + (达标 ? "达标" : "未达标") + "</span>" +
            '<span class="片段摘要">' + 转义HTML(路径) + "</span>" +
            (均分 !== undefined ? '<span class="徽章">均分 ' + 均分 + "</span>" : "") +
            (时间 ? '<span class="徽章">' + 转义HTML(时间) + "</span>" : "") +
            "</div></div>";
    }).join("");
}

function 初始化_达标评估() {
    加载达标历史();
    if (达标进行中) 启动达标轮询();
}

/* ============================================================
   页面 9：配置
   ============================================================ */
let 当前配置 = null;

async function 加载配置() {
    const 容器 = document.getElementById("配置表格");
    const 结果 = await api.获取配置();
    if (结果.错误) {
        容器.innerHTML = '<p class="占位提示">' + 转义HTML(结果.错误) + "</p>";
        return;
    }
    当前配置 = 结果;
    容器.innerHTML = 渲染配置分组(结果);
}

function 渲染配置分组(配置) {
    let html = "";
    for (const [分组名, 分组内容] of Object.entries(配置 || {})) {
        if (typeof 分组内容 !== "object" || 分组内容 === null || Array.isArray(分组内容)) continue;
        html += '<div class="配置分组"><div class="配置分组标题">' + 转义HTML(分组名) + "</div>";
        for (const [键, 值] of Object.entries(分组内容)) {
            html += 渲染配置行(分组名, 键, 值);
        }
        html += "</div>";
    }
    return html || '<p class="占位提示">无配置数据</p>';
}

function 渲染配置行(分组名, 键, 值) {
    const 名称 = 分组名 + "." + 键;
    if (Array.isArray(值)) {
        return '<div class="配置行"><span class="配置键">' + 转义HTML(名称) + "</span>" +
            '<input type="text" data-配置路径="' + 转义HTML(名称) + '" value="' + 转义HTML(值.join(",")) + '" title="数组，用逗号分隔">' +
            "</div>";
    }
    if (typeof 值 === "boolean") {
        return '<div class="配置行"><span class="配置键">' + 转义HTML(名称) + "</span>" +
            '<select data-配置路径="' + 转义HTML(名称) + '">' +
            '<option value="true"' + (值 ? " selected" : "") + ">true</option>" +
            '<option value="false"' + (!值 ? " selected" : "") + ">false</option>" +
            "</select></div>";
    }
    if (typeof 值 === "number") {
        return '<div class="配置行"><span class="配置键">' + 转义HTML(名称) + "</span>" +
            '<input type="number" data-配置路径="' + 转义HTML(名称) + '" value="' + 转义HTML(值) + '">' +
            "</div>";
    }
    return '<div class="配置行"><span class="配置键">' + 转义HTML(名称) + "</span>" +
        '<input type="text" data-配置路径="' + 转义HTML(名称) + '" value="' + 转义HTML(值 ?? "") + '">' +
        "</div>";
}

function 收集配置值() {
    const 结果 = {};
    document.querySelectorAll("#配置表格 [data-配置路径]").forEach((输入) => {
        const [分组, 键] = 输入.dataset.配置路径.split(".");
        let 值;
        if (输入.tagName === "SELECT") {
            值 = 输入.value === "true";
        } else if (输入.type === "number") {
            值 = Number(输入.value);
        } else {
            const 原始 = 输入.value.trim();
            值 = 原始.includes(",") ? 原始.split(/[,，]/).map((s) => s.trim()).filter(Boolean) : 原始;
        }
        if (!结果[分组]) 结果[分组] = {};
        结果[分组][键] = 值;
    });
    return 结果;
}

async function 保存配置() {
    const 新配置 = 收集配置值();
    const 结果 = await api.保存配置(新配置);
    if (结果.错误) return;
    当前配置 = 结果.配置 || 新配置;
    toast("配置已保存", "成功");
}

function 恢复默认() {
    if (confirm("确定恢复默认配置吗？当前修改将丢失。\n（若后端无重置接口，请重启服务使设置生效）")) {
        toast("已提示恢复默认：请在后端配置系统执行重置，或重启服务使默认值生效。", "警告");
    }
}

function 初始化_配置() {
    加载配置();
}

/* ============================================================
   事件绑定
   ============================================================ */
function 绑定事件() {
    // 侧边导航
    document.querySelectorAll(".导航项").forEach((项) => {
        项.addEventListener("click", () => 切换页面(项.dataset.模块));
    });

    // 移动端菜单按钮 / 遮罩
    const 菜单按钮 = document.getElementById("菜单按钮");
    const 侧栏 = document.getElementById("侧边栏");
    const 遮罩 = document.getElementById("遮罩");
    if (菜单按钮) 菜单按钮.addEventListener("click", () => {
        侧栏.classList.add("展开");
        遮罩.classList.add("显示");
    });
    if (遮罩) 遮罩.addEventListener("click", () => {
        侧栏.classList.remove("展开");
        遮罩.classList.remove("显示");
    });

    // ---- 仪表盘 ----
    document.getElementById("显存预估按钮").addEventListener("click", 显存预估);

    // ---- 数据预处理 ----
    document.getElementById("上传按钮").addEventListener("click", 上传文件);
    document.getElementById("转写按钮").addEventListener("click", 开始转写);
    document.getElementById("查看转写按钮").addEventListener("click", 查看转写文本);
    document.getElementById("分割按钮").addEventListener("click", 执行话题分割);
    document.getElementById("预览分割按钮").addEventListener("click", 预览分割);
    document.querySelectorAll('input[name="上传类型"]').forEach((radio) => {
        radio.addEventListener("change", 更新视频解析行);
    });

    // ---- 打标 ----
    document.getElementById("加载片段按钮").addEventListener("click", 加载打标片段);
    document.getElementById("自动打标按钮").addEventListener("click", 自动打标);
    document.getElementById("打标导出按钮").addEventListener("click", 导出打标数据包);

    // ---- 日记生成 ----
    document.getElementById("规划按钮").addEventListener("click", 规划时间线);
    document.getElementById("日记生成按钮").addEventListener("click", 开始生成日记);
    document.getElementById("日记刷新按钮").addEventListener("click", 加载日记列表);
    document.getElementById("日记导出按钮").addEventListener("click", 导出日记文本);
    document.getElementById("日记聊天发送").addEventListener("click", 发送日记消息);
    document.getElementById("日记聊天输入").addEventListener("keydown", (事件) => {
        if (事件.key === "Enter") 发送日记消息();
    });

    // ---- 微调 ----
    document.getElementById("勾选情感").addEventListener("change", 更新维度组合提示);
    document.getElementById("勾选记忆").addEventListener("change", 更新维度组合提示);
    document.getElementById("勾选身份").addEventListener("change", 更新维度组合提示);
    document.getElementById("微调模型选择").addEventListener("change", (事件) => {
        document.getElementById("微调模型路径").value = 事件.target.value;
    });
    document.getElementById("微调检测按钮").addEventListener("click", 检测模型能力);
    document.getElementById("微调数据预览按钮").addEventListener("click", 数据预览);
    document.getElementById("微调开始按钮").addEventListener("click", 开始微调);

    // ---- 模型管理 ----
    document.getElementById("模型扫描按钮").addEventListener("click", 模型扫描);
    document.getElementById("下载按钮").addEventListener("click", 下载模型);

    // ---- 推理 ----
    document.getElementById("参数推荐按钮").addEventListener("click", 参数自动推荐);
    document.getElementById("推理初始化按钮").addEventListener("click", 初始化推理);
    document.getElementById("推理发送").addEventListener("click", 发送推理消息);
    document.getElementById("推理输入").addEventListener("keydown", (事件) => {
        if (事件.key === "Enter") 发送推理消息();
    });
    document.getElementById("记忆添加按钮").addEventListener("click", 添加记忆);
    document.getElementById("记忆检索按钮").addEventListener("click", 检索记忆);

    // ---- 达标评估 ----
    document.getElementById("达标开始按钮").addEventListener("click", 开始达标评估);

    // ---- 配置 ----
    document.getElementById("保存配置按钮").addEventListener("click", 保存配置);
    document.getElementById("恢复默认按钮").addEventListener("click", 恢复默认);

    // ---- 全局事件委托：动态生成的按钮 ----
    document.addEventListener("click", async (事件) => {
        const 按钮 = 事件.target.closest("[data-动作]");
        if (!按钮) return;
        const 动作 = 按钮.dataset.动作;
        if (动作 === "快捷入口") {
            切换页面(按钮.dataset.目标);
        } else if (动作 === "调整边界") {
            提交边界(按钮.dataset.片段ID, 按钮);
        } else if (动作 === "打标复核") {
            提交打标复核(按钮.dataset.片段ID, 按钮);
        } else if (动作 === "日记打开") {
            打开日记单篇(按钮.dataset.角色名, 按钮.dataset.年龄);
        } else if (动作 === "日记审阅") {
            提交日记审阅(按钮.dataset.角色名, 按钮.dataset.年龄);
        } else if (动作 === "模型评估") {
            模型性能评估(按钮.dataset.路径);
        }
    });
}

/* ===================== 启动 ===================== */
document.addEventListener("DOMContentLoaded", function () {
    绑定事件();
    切换页面("仪表盘");
    刷新服务状态();
    setInterval(刷新服务状态, 15000);
});
