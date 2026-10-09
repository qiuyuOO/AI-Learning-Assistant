"""按会话聚合 usage 账本。

回答的问题：一次典型的多轮客服问答，花了几次模型调用、多少 token、多少钱、P95 延迟多少？

跑：.venv\\Scripts\\python.exe -m a2_error_taxonomy.usage_report
"""
import json
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

from llm_client import _DEFAULT_USAGE_LOG

# 单价：元 / 百万 token（来源见 A2 查过的官方价格页）
# ★ 改价时只改这里 —— 这是"价格表不进 runtime"的落地
PRICE = {
    "peak": {"cache": 0.04, "miss": 2.0, "out": 8.0},
    "off":  {"cache": 0.02, "miss": 1.0, "out": 4.0},
}

CST = timezone(timedelta(hours=8))


def _display_width(s: str) -> int:
    """中文字符占 2 列，用它算填充宽度 —— 否则表会歪。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s: str, width: int) -> str:
    return s + " " * max(0, width - _display_width(s))


def is_peak(ts: float) -> bool:
    """高峰：北京时间 周一至周五 9:00-12:00 / 14:00-18:00。"""
    dt = datetime.fromtimestamp(ts, CST)
    if dt.weekday() >= 5:
        return False
    return (9 <= dt.hour < 12) or (14 <= dt.hour < 18)


def cost_of(r: dict) -> float:
    """一次调用的成本。★ 用 ts 判时段，不用"现在" —— 历史记录必须按当时时段算。"""
    p = PRICE["peak" if is_peak(r["ts"]) else "off"]
    return (r["cache_read_tokens"] * p["cache"]
            + r["input_tokens"] * p["miss"]
            + r["output_tokens"] * p["out"]) / 1_000_000


def pctl(values: list[float], q: float) -> float:
    """分位数。★ 用 P95 而不是平均值 —— 延迟是长尾分布。"""
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(int(len(s) * q), len(s) - 1)
    return s[idx]


def load(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"账本不存在：{path}")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def main() -> None:
    records = load(Path(_DEFAULT_USAGE_LOG))
    print(f"账本：{_DEFAULT_USAGE_LOG}   共 {len(records)} 条调用记录\n")

    # ---- 按会话聚合 ----
    sessions: dict[str, list[dict]] = {}
    for r in records:
        sessions.setdefault(r["session_id"], []).append(r)

    header = ["session", "调用", "未命中in", "缓存命中", "输出",
              "命中率", "成本(元)", "P95延迟"]
    widths = [16, 5, 9, 9, 7, 7, 10, 9]
    print("  ".join(_pad(h, w) for h, w in zip(header, widths)))
    print("-" * (sum(widths) + 2 * len(widths)))

    total = {"calls": 0, "miss": 0, "cache": 0, "out": 0, "cost": 0.0}
    all_latency: list[float] = []

    for sid in sorted(sessions):
        rows = sorted(sessions[sid], key=lambda r: r["ts"])
        miss = sum(r["input_tokens"] for r in rows)
        cache = sum(r["cache_read_tokens"] for r in rows)
        out = sum(r["output_tokens"] for r in rows)
        cost = sum(cost_of(r) for r in rows)
        lat = [r["latency_ms"] for r in rows]
        all_latency.extend(lat)
        rate = cache / (cache + miss) * 100 if (cache + miss) else 0.0

        cells = [sid, str(len(rows)), f"{miss:,}", f"{cache:,}", f"{out:,}",
                 f"{rate:.0f}%", f"{cost:.5f}", f"{pctl(lat, 0.95):.0f}ms"]
        print("  ".join(_pad(c, w) for c, w in zip(cells, widths)))

        total["calls"] += len(rows)
        total["miss"] += miss
        total["cache"] += cache
        total["out"] += out
        total["cost"] += cost

    print("-" * (sum(widths) + 2 * len(widths)))
    rate = total["cache"] / (total["cache"] + total["miss"]) * 100
    cells = ["合计", str(total["calls"]), f"{total['miss']:,}",
             f"{total['cache']:,}", f"{total['out']:,}", f"{rate:.0f}%",
             f"{total['cost']:.5f}", f"{pctl(all_latency, 0.95):.0f}ms"]
    print("  ".join(_pad(c, w) for c, w in zip(cells, widths)))

    # ---- 延迟分位数：为什么看 P95 不看均值 ----
    print(f"\n延迟分布（{len(all_latency)} 次调用）")
    for q, label in ((0.5, "P50"), (0.9, "P90"), (0.95, "P95"), (1.0, "max")):
        print(f"  {label:<4} {pctl(all_latency, q):>8.0f} ms")
    print(f"  {'均值':<4} {sum(all_latency) / len(all_latency):>8.0f} ms"
          "   ← 被长尾抬高，不代表体感")

    # ---- 单会话样本：一次典型多轮问答的构成 ----
    if sessions:
        sid = sorted(sessions)[-1]
        rows = sorted(sessions[sid], key=lambda r: r["ts"])
        print(f"\n样本会话 {sid}（{len(rows)} 次调用，按时间顺序）")
        print(f"  {'序':<3}{'未命中in':>9}{'缓存命中':>10}{'输出':>8}"
              f"{'时段':>6}{'成本(元)':>11}{'延迟':>9}")
        for i, r in enumerate(rows, 1):
            print(f"  {i:<3}{r['input_tokens']:>9,}{r['cache_read_tokens']:>10,}"
                  f"{r['output_tokens']:>8,}"
                  f"{'高峰' if is_peak(r['ts']) else '空闲':>6}"
                  f"{cost_of(r):>11.5f}{r['latency_ms']:>8.0f}ms")


if __name__ == "__main__":
    main()