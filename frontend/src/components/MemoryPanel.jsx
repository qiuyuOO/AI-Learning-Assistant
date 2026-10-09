import React, { useState } from "react";

function ConfidenceBar({ value }) {
  const pct = Math.max(0, Math.min(1, value || 0)) * 100;
  return (
    <span className="conf" title={`置信度 ${(value || 0).toFixed(2)}`}>
      <span className="conf__fill" style={{ width: `${pct}%` }} />
    </span>
  );
}

function Group({ label, children, count }) {
  if (!count) return null;
  return (
    <div className="mem-group">
      <div className="mem-group__head">
        <span>{label}</span>
        <span className="mem-group__count">{count}</span>
      </div>
      <div className="mem-group__body">{children}</div>
    </div>
  );
}

export default function MemoryPanel({ memory, onRemember, className = "" }) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const m = memory?.memory || null;

  const submit = async () => {
    const c = text.trim();
    if (!c || busy) return;
    setBusy(true);
    await onRemember(c);
    setBusy(false);
    setText("");
  };

  const interests = (m?.interests || []).filter((x) => x.status !== "deprecated");
  const dislikes = (m?.dislikes || []).filter((x) => x.status !== "deprecated");
  const prefs = m?.preferences || [];
  const criteria = m?.decision_criteria || [];
  const beliefs = m?.beliefs || [];
  const habits = m?.habits || [];

  return (
    <aside className={`memory ${className}`}>
      <div className="memory__head">
        <h2 className="panel-title">
          <span className="panel-title__idx">03</span> 长期记忆
        </h2>
        {m?.updated_at && (
          <span className="memory__updated">
            更新于 {new Date(m.updated_at).toLocaleString("zh-CN")}
          </span>
        )}
      </div>

      {/* 记住偏好入口 */}
      <div className="remember">
        <label className="remember__label">让 Jev 长期记住</label>
        <div className="remember__row">
          <input
            className="remember__input"
            value={text}
            placeholder="例如：我只关注 RAG 与 Agent 方向"
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && submit()}
          />
          <button className="btn btn--remember" onClick={submit} disabled={!text.trim() || busy}>
            {busy ? <span className="spin" /> : "记住"}
          </button>
        </div>
      </div>

      <div className="memory__scroll">
        {!m ? (
          <p className="empty-hint">记忆加载中…</p>
        ) : (
          <>
            {memory.brief && (
              <div className="brief">
                <div className="brief__tag">当前简报</div>
                <pre className="brief__text">{memory.brief}</pre>
              </div>
            )}

            <Group label="长期关注" count={interests.length}>
              {interests.map((it, i) => (
                <div className="mem-item" key={it.id || i}>
                  <span className="mem-item__value">{String(it.value)}</span>
                  <span className="mem-item__side">
                    <ConfidenceBar value={it.confidence} />
                    {it.source === "user_explicit" && (
                      <span className="tag tag--explicit">明说</span>
                    )}
                  </span>
                </div>
              ))}
            </Group>

            <Group label="不喜欢" count={dislikes.length}>
              {dislikes.map((it, i) => (
                <div className="mem-item mem-item--neg" key={it.id || i}>
                  <span className="mem-item__value">{String(it.value)}</span>
                  <ConfidenceBar value={it.confidence} />
                </div>
              ))}
            </Group>

            <Group label="偏好" count={prefs.length}>
              {prefs.map((p, i) => (
                <div className="mem-item" key={i}>
                  <span className="mem-item__value">
                    {p.subject}
                    <em className={p.preference >= 0 ? "pos" : "neg"}>
                      {p.preference >= 0 ? " 喜欢" : " 回避"}
                    </em>
                  </span>
                  <ConfidenceBar value={Math.abs(p.preference)} />
                </div>
              ))}
            </Group>

            <Group label="评价时看重" count={criteria.length}>
              {criteria.map((c, i) => (
                <div className="mem-item" key={i}>
                  <span className="mem-item__value">{c.name}</span>
                  <ConfidenceBar value={c.weight} />
                </div>
              ))}
            </Group>

            <Group label="用户相信" count={beliefs.length}>
              {beliefs.map((b, i) => (
                <div className="mem-item mem-item--stack" key={i}>
                  <span className="mem-item__value">{b.topic}</span>
                  <span className="mem-item__sub">{b.proposition}</span>
                  <ConfidenceBar value={b.confidence} />
                </div>
              ))}
            </Group>

            <Group label="行为习惯" count={habits.length}>
              {habits.map((h, i) => (
                <div className="mem-item mem-item--stack" key={i}>
                  <span className="mem-item__value">{h.behavior}</span>
                  {h.context && <span className="mem-item__sub">{h.context}</span>}
                  <ConfidenceBar value={h.confidence} />
                </div>
              ))}
            </Group>

            {interests.length + dislikes.length + prefs.length +
              criteria.length + beliefs.length + habits.length === 0 && (
              <p className="empty-hint">
                还没有沉淀任何记忆。多聊几句，或在上方「记住」里显式告诉 Jev 你的偏好。
              </p>
            )}
          </>
        )}
      </div>
    </aside>
  );
}
