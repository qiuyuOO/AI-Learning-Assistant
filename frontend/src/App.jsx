import { useCallback, useEffect, useRef, useState } from "react";
import { api, OPERATOR_ID } from "./api.js";
import Sidebar from "./components/Sidebar.jsx";
import ChatWindow from "./components/ChatWindow.jsx";
import MemoryPanel from "./components/MemoryPanel.jsx";

export default function App() {
  const [sessions, setSessions] = useState([]);
  const [currentId, setCurrentId] = useState(null);
  const [turns, setTurns] = useState([]);
  const [memory, setMemory] = useState(null);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [booting, setBooting] = useState(true);
  const [online, setOnline] = useState(null); // null=未知 true/false
  const [toast, setToast] = useState(null);
  const [mobilePanel, setMobilePanel] = useState(null); // 'sessions' | 'memory' | null

  const toastTimer = useRef(null);
  const bootedRef = useRef(false);
  const flash = useCallback((msg, kind = "info") => {
    setToast({ msg, kind });
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 3200);
  }, []);

  const refreshMemory = useCallback(async () => {
    try {
      setMemory(await api.getMemory());
    } catch (e) {
      /* 记忆读失败不致命 */
    }
  }, []);

  const refreshSessions = useCallback(async () => {
    try {
      const data = await api.listSessions();
      setSessions(data.sessions || []);
      return data.sessions || [];
    } catch (e) {
      return [];
    }
  }, []);

  // 启动：探活 + 拉会话/记忆；有历史就打开最近一个，否则新建
  useEffect(() => {
    if (bootedRef.current) return;   // ★ StrictMode 开发期会双调本 effect，守住只初始化一次
    bootedRef.current = true;
    (async () => {
      try {
        await api.health();
        setOnline(true);
      } catch {
        setOnline(false);
        setBooting(false);
        flash("无法连接后端，请确认已启动 uvicorn server:app", "error");
        return;
      }
      const [list] = await Promise.all([refreshSessions(), refreshMemory()]);
      if (list.length > 0) {
        await openSession(list[0].session_id);
      } else {
        await handleNew(true);
      }
      setBooting(false);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const openSession = useCallback(
    async (sessionId) => {
      try {
        const data = await api.getSession(sessionId);
        setCurrentId(sessionId);
        setTurns(data.turns || []);
        setMobilePanel(null);
      } catch (e) {
        flash(`打开会话失败：${e.message}`, "error");
      }
    },
    [flash]
  );

  const handleNew = useCallback(
    async (silent = false) => {
      try {
        const meta = await api.newSession();
        setCurrentId(meta.session_id);
        setTurns([]);
        setMobilePanel(null);
        // 新会话尚未落盘，先只在本地置顶一个占位，待首条消息后刷新真实列表
        setSessions((prev) =>
          prev.some((s) => s.session_id === meta.session_id) ? prev : [meta, ...prev]
        );
        if (!silent) flash("已开启新会话", "info");
      } catch (e) {
        flash(`新建会话失败：${e.message}`, "error");
      }
    },
    [flash]
  );

  const handleSend = useCallback(
    async (text) => {
      const body = text.trim();
      if (!body || sending) return;
      let sid = currentId;
      setInput("");
      setSending(true);

      // 乐观更新：先把用户这条显示出来
      const optimistic = [...turns, { role: "user", text: body, ts: "", pending: true }];
      setTurns(optimistic);

      try {
        if (!sid) {
          const meta = await api.newSession();
          sid = meta.session_id;
          setCurrentId(sid);
        }
        const data = await api.postMessage(sid, body);
        setTurns(data.turns || []); // 以服务端为准（含干净的 user/assistant）
        await Promise.all([refreshSessions(), refreshMemory()]);
      } catch (e) {
        setTurns(optimistic.map((t) => ({ ...t, pending: false })));
        flash(`发送失败：${e.message}`, "error");
      } finally {
        setSending(false);
      }
    },
    [currentId, turns, sending, refreshSessions, refreshMemory, flash]
  );

  const handleRemember = useCallback(
    async (content) => {
      try {
        const r = await api.remember(content);
        if (r.ok) {
          flash(r.message || "已记住", "success");
          await refreshMemory();
        } else {
          flash(r.message || "记忆更新未成功，稍后重试", "error");
        }
      } catch (e) {
        flash(`记住失败：${e.message}`, "error");
      }
    },
    [refreshMemory, flash]
  );

  return (
    <div className={`app${booting ? " is-booting" : ""}`}>
      <div className="grain" aria-hidden="true" />

      <header className="topbar">
        <div className="brand">
          <span className="brand__mark">J</span>
          <span className="brand__text">
            <span className="brand__name">Jev</span>
            <span className="brand__sub">AI 研究观测台</span>
          </span>
        </div>

        <div className="topbar__meta">
          <span className={`dot dot--${online === false ? "off" : online ? "on" : "wait"}`} />
          <span className="chip">
            operator · <b>{OPERATOR_ID}</b>
          </span>
          <button
            className="iconbtn only-narrow"
            onClick={() => setMobilePanel(mobilePanel === "sessions" ? null : "sessions")}
            title="会话"
          >
            ☰
          </button>
          <button
            className="iconbtn only-narrow"
            onClick={() => setMobilePanel(mobilePanel === "memory" ? null : "memory")}
            title="记忆"
          >
            ◈
          </button>
        </div>
      </header>

      <main className="layout">
        <Sidebar
          sessions={sessions}
          currentId={currentId}
          onOpen={openSession}
          onNew={handleNew}
          className={mobilePanel === "sessions" ? "is-open" : ""}
        />

        <ChatWindow
          turns={turns}
          input={input}
          setInput={setInput}
          onSend={handleSend}
          sending={sending}
          booting={booting}
          online={online}
          currentId={currentId}
        />

        <MemoryPanel
          memory={memory}
          onRemember={handleRemember}
          className={mobilePanel === "memory" ? "is-open" : ""}
        />
      </main>

      {toast && <div className={`toast toast--${toast.kind}`}>{toast.msg}</div>}
    </div>
  );
}
