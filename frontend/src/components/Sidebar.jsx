import React from "react";

function relTime(iso) {
  if (!iso) return "";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const diff = Date.now() - then;
  const m = Math.round(diff / 60000);
  if (m < 1) return "刚刚";
  if (m < 60) return `${m} 分钟前`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h} 小时前`;
  const d = Math.round(h / 24);
  if (d < 30) return `${d} 天前`;
  return new Date(iso).toLocaleDateString("zh-CN");
}

export default function Sidebar({ sessions, currentId, onOpen, onNew, className = "" }) {
  return (
    <aside className={`sidebar ${className}`}>
      <div className="sidebar__head">
        <h2 className="panel-title">
          <span className="panel-title__idx">01</span> 会话
        </h2>
        <button className="btn btn--new" onClick={() => onNew()} title="新建会话">
          <span className="btn__plus">＋</span> 新建
        </button>
      </div>

      <nav className="session-list">
        {sessions.length === 0 && (
          <p className="empty-hint">还没有会话，点「新建」开始。</p>
        )}
        {sessions.map((s, i) => {
          const active = s.session_id === currentId;
          return (
            <button
              key={s.session_id}
              className={`session-item${active ? " is-active" : ""}`}
              onClick={() => onOpen(s.session_id)}
              style={{ animationDelay: `${Math.min(i, 12) * 28}ms` }}
            >
              <span className="session-item__rail" aria-hidden="true" />
              <span className="session-item__title">
                {s.title || "（新会话 · 未命名）"}
              </span>
              <span className="session-item__meta">
                <span>{relTime(s.updated_at)}</span>
                <span className="session-item__turns">{s.turn_count ?? 0} 轮</span>
              </span>
            </button>
          );
        })}
      </nav>
    </aside>
  );
}
