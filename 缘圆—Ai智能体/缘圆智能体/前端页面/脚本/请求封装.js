/* ============================================================
   缘圆智能体 V2 —— 请求封装（window.请求）
   全部方法返回解析后的 JSON；非 2xx 抛出含后端错误消息的 Error
   （优先解析 FastAPI 的 detail 字段）。
   ============================================================ */
(function () {
  "use strict";

  /* 从响应体提取后端错误消息：detail 可能是字符串 / {msg} / 校验错误数组 */
  function 提取错误消息(数据, 响应) {
    if (数据 && typeof 数据.detail === "string") {
      return 数据.detail;
    }
    if (Array.isArray(数据 && 数据.detail)) {
      return 数据.detail
        .map(条目 => (条目 && 条目.msg) || JSON.stringify(条目))
        .join("；");
    }
    if (数据 && 数据.detail && typeof 数据.detail === "object") {
      if (typeof 数据.detail.msg === "string") {
        return 数据.detail.msg;
      }
      try {
        return JSON.stringify(数据.detail);
      } catch (忽略) { /* 序列化失败则走兜底 */ }
    }
    return `请求失败（HTTP ${响应.status} ${响应.statusText || ""}）`;
  }

  async function 处理响应(响应) {
    let 数据 = null;
    try {
      数据 = await 响应.json();
    } catch (忽略) {
      /* 响应体不是 JSON（如空响应），忽略 */
    }
    if (!响应.ok) {
      throw new Error(提取错误消息(数据, 响应));
    }
    return 数据;
  }

  /* GET：返回解析后的 JSON */
  async function 获取(url) {
    const 响应 = await fetch(url, { method: "GET" });
    return 处理响应(响应);
  }

  /* POST JSON：自动设置 Content-Type */
  async function 提交(url, 数据对象) {
    const 响应 = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(数据对象 || {}),
    });
    return 处理响应(响应);
  }

  /* POST FormData：不手动设置 Content-Type（浏览器自动带 multipart 边界） */
  async function 上传(url, 表单数据) {
    const 响应 = await fetch(url, {
      method: "POST",
      body: 表单数据,
    });
    return 处理响应(响应);
  }

  /* DELETE：参数对象拼接为查询串（也接受字符串原样拼接） */
  async function 删除(url, 参数) {
    let 完整地址 = url;
    if (参数) {
      const 查询串 = typeof 参数 === "string"
        ? 参数
        : new URLSearchParams(参数).toString();
      if (查询串) {
        完整地址 = url.includes("?") ? `${url}&${查询串}` : `${url}?${查询串}`;
      }
    }
    const 响应 = await fetch(完整地址, { method: "DELETE" });
    return 处理响应(响应);
  }

  /* 合并事件载荷辅助：多个对象合并为一个，无参数时返回空对象 */
  function 合并事件(...对象们) {
    const 结果 = {};
    for (const 对象 of 对象们) {
      if (对象 && typeof 对象 === "object") {
        Object.assign(结果, 对象);
      }
    }
    return 结果;
  }

  window.请求 = { 获取, 提交, 上传, 删除, 合并事件 };
})();
