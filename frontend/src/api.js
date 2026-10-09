// 前端 → 后端(FastAPI) 的 API 封装。
// ★ 全部走同源 /api（vite 代理到 127.0.0.1:8000），无需处理 CORS。
// ★ operator_id 参数化：当前无鉴权，用一个固定 web 用户；接登录后换成真实用户 id 即可。

export const OPERATOR_ID = "web";

async function req(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* 忽略解析失败 */
    }
    throw new Error(detail);
  }
  return res.json();
}

const q = (params) => {
  const s = new URLSearchParams(params).toString();
  return s ? `?${s}` : "";
};

export const api = {
  health: () => req("/api/health"),

  listSessions: (operatorId = OPERATOR_ID) =>
    req(`/api/sessions${q({ operator_id: operatorId })}`),

  newSession: (operatorId = OPERATOR_ID) =>
    req("/api/sessions", {
      method: "POST",
      body: JSON.stringify({ operator_id: operatorId }),
    }),

  getSession: (sessionId, operatorId = OPERATOR_ID) =>
    req(`/api/sessions/${encodeURIComponent(sessionId)}${q({ operator_id: operatorId })}`),

  postMessage: (sessionId, text, operatorId = OPERATOR_ID) =>
    req(`/api/sessions/${encodeURIComponent(sessionId)}/messages`, {
      method: "POST",
      body: JSON.stringify({ operator_id: operatorId, text }),
    }),

  getMemory: (operatorId = OPERATOR_ID) =>
    req(`/api/memory${q({ operator_id: operatorId })}`),

  remember: (content, operatorId = OPERATOR_ID) =>
    req("/api/memory/remember", {
      method: "POST",
      body: JSON.stringify({ operator_id: operatorId, content }),
    }),
};
