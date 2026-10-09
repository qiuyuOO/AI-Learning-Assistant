import React, { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

function Bubble({ turn }) {
  const isUser = turn.role === "user";
  return (
    <div className={`msg msg--${isUser ? "user" : "agent"}${turn.pending ? " is-pending" : ""}`}>
      <div className="msg__avatar" aria-hidden="true">
        {isUser ? "你" : "J"}
      </div>
      <div className="msg__body">
        <div className="msg__who">{isUser ? "你" : "Jev"}</div>
        <div className="msg__content">
          {isUser ? (
            <p>{turn.text}</p>
          ) : (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{turn.text || ""}</ReactMarkdown>
          )}
        </div>
      </div>
    </div>
  );
}

function TypingRow() {
  return (
    <div className="msg msg--agent msg--typing">
      <div className="msg__avatar" aria-hidden="true">J</div>
      <div className="msg__body">
        <div className="msg__who">Jev</div>
        <div className="typing">
          <span /><span /><span />
        </div>
      </div>
    </div>
  );
}

export default function ChatWindow({
  turns, input, setInput, onSend, sending, booting, online, currentId,
}) {
  const scroller = useRef(null);
  const ta = useRef(null);

  useEffect(() => {
    const el = scroller.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [turns, sending]);

  // 自适应高度的输入框
  useEffect(() => {
    const el = ta.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  }, [input]);

  const submit = () => {
    if (!input.trim() || sending) return;
    onSend(input);
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <section className="chat">
      <div className="chat__head">
        <h2 className="panel-title">
          <span className="panel-title__idx">02</span> 对话
        </h2>
        <span className="chat__sid" title="当前会话 id">
          {currentId ? currentId : "—"}
        </span>
      </div>

      <div className="chat__scroll" ref={scroller}>
        {booting ? (
          <div className="boot">
            <div className="boot__ring" />
            <p>正在接入观测台…</p>
          </div>
        ) : online === false ? (
          <div className="offline">
            <h3>后端未连接</h3>
            <p>
              请在 <code>Ai/</code> 目录启动：
              <br />
              <code>uvicorn server:app --port 8000</code>
            </p>
          </div>
        ) : turns.length === 0 && !sending ? (
          <div className="chat__empty">
            <div className="chat__empty-mark">✦</div>
            <h3>开始一段新的观测</h3>
            <p>
              我能帮你追踪 arXiv 论文、Hugging Face / OpenAI 官方动态，
              并按你的长期关注方向（例如 RAG）筛选排序。
            </p>
            <div className="suggestions">
              {["拉一份最近 3 天的 RAG / 检索相关论文",
                "今天 AI 领域有什么值得看的动态",
                "记住我：只关注检索增强与 Agent 方向"].map((s) => (
                <button key={s} className="suggestion" onClick={() => onSend(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {turns.map((t, i) => (
              <Bubble key={i} turn={t} />
            ))}
            {sending && <TypingRow />}
          </>
        )}
      </div>

      <div className="composer">
        <textarea
          ref={ta}
          className="composer__input"
          rows={1}
          placeholder={online === false ? "后端未连接…" : "输入消息 · Enter 发送 / Shift+Enter 换行"}
          value={input}
          disabled={online === false || booting}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
        />
        <button
          className="composer__send"
          onClick={submit}
          disabled={!input.trim() || sending || online === false}
          title="发送"
        >
          {sending ? <span className="spin" /> : <span aria-hidden="true">↑</span>}
        </button>
      </div>
    </section>
  );
}
