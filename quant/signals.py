"""当日信号快照：每个策略今天触发了哪些票、排序分数、哪些进前 5、市场状态如何。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import datasource, engine
from .config import Config
from .strategies import STRATEGIES


def market_status(cfg: Config, dates: pd.DatetimeIndex) -> dict:
    """市场状态：直接按当前数据算，不依赖上次回测落下的 regime_state.csv。

    这样「只选股」也能拿到真实状态，而不是默认 OFF / 0 天。
    """
    idx = datasource.load_index_close(cfg).reindex(dates).ffill()
    window = cfg.regime_window
    ma = idx.rolling(window).mean()

    state = engine.build_regime(cfg, dates)          # 顺带刷新 data/regime_state.csv
    arr = state.to_numpy()
    cur = int(arr[-1]) if len(arr) else 0
    j = len(arr) - 1
    while j > 0 and arr[j] == arr[j - 1]:
        j -= 1
    since = str(dates[j].date()) if len(arr) else str(dates[-1].date())
    ndays = len(arr) - j if len(arr) else 0
    return {
        "on": bool(cur == 1),
        "since": since,
        "days": int(ndays),
        "index_close": round(float(idx.iloc[-1]), 2),
        "ma": round(float(ma.iloc[-1]), 2),
        "gap_pct": round(float((idx.iloc[-1] / ma.iloc[-1] - 1) * 100), 2),
        "window": window,
    }


def snapshot(cfg: Config, frames: dict[str, pd.DataFrame],
             signals: dict[str, dict[str, pd.DataFrame]]) -> dict:
    close = frames["close"]
    names = datasource.load_names(cfg)
    last = close.index[-1]
    prev = close.index[-2] if len(close.index) > 1 else last

    out = {"as_of": str(last.date()), "strategies": []}
    for st in STRATEGIES:
        enter = signals[st.key]["enter"].loc[last]
        rank = signals[st.key]["rank"].loc[last]
        px = close.loc[last]
        chg = px / close.loc[prev] - 1.0

        mask = enter.fillna(0).astype(bool) & px.notna()
        items = []
        for sym in mask[mask].index:
            score = rank.get(sym, np.nan)
            items.append({
                "symbol": sym,
                "code": sym[2:],
                "name": names.get(sym, ""),
                "close": round(float(px[sym]), 2),
                "day_chg_pct": round(float(chg.get(sym, np.nan) * 100), 2),
                "score": None if pd.isna(score) else round(float(score), 4),
            })
        items.sort(key=lambda x: (x["score"] is None, -(x["score"] or 0)))
        for i, it in enumerate(items):
            it["rank"] = i + 1
            it["selected"] = i < cfg.max_positions

        out["strategies"].append({
            "key": st.key, "name": st.name, "kind": st.kind, "prefix": st.prefix,
            "rank_rule": st.rank_desc,
            "n_signal": len(items),
            "n_selected": min(len(items), cfg.max_positions),
            "candidates": items,
        })
    return out


def recent_trades(cfg: Config, out_dir: Path, per_strategy: int = 3, limit: int = 15) -> list[dict]:
    rows = []
    for st in STRATEGIES:
        path = out_dir / f"{st.prefix}_regx_trades.csv"
        if not path.exists():
            path = out_dir / f"{st.prefix}_trades.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        for _, r in df.sort_values("entry_date").tail(per_strategy).iterrows():
            rows.append({
                "strategy": f"{st.key} {st.name}",
                "symbol": r["symbol"],
                "name": r.get("symbol_name", ""),
                "entry_date": r["entry_date"],
                "exit_date": r["exit_date"],
                "entry_price": round(float(r["entry_price"]), 2),
                "exit_price": round(float(r["exit_price"]), 2),
                "pnl_pct": round(float(r["pnl_pct"]), 2),
                "holding_bars": int(r["holding_bars"]),
            })
    rows.sort(key=lambda x: x["entry_date"], reverse=True)
    return rows[:limit]


def write(cfg: Config, frames: dict[str, pd.DataFrame],
          signals: dict[str, dict[str, pd.DataFrame]],
          out_dir: Path, analysis: dict | None = None) -> dict:
    snap = snapshot(cfg, frames, signals)
    status = market_status(cfg, frames["close"].index)
    trades = recent_trades(cfg, out_dir)

    payload = {
        "as_of": snap["as_of"],
        "status": status,
        "snapshot": snap,
        "recent_trades": trades,
        "analysis": analysis or {},
    }
    (out_dir / "signal_snapshot.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- Markdown 版 ----
    lines = [f"# 信号快照（{snap['as_of']} 收盘）", ""]
    state_txt = "ON（允许开新仓）" if status["on"] else "OFF（禁止开新仓）"
    lines.append(
        f"> **市场状态：{state_txt}**　基准指数 {status['index_close']} vs "
        f"MA{status['window']} {status['ma']}（{status['gap_pct']:+.2f}%），"
        f"已持续 {status['days']} 个交易日（自 {status['since']}）")
    lines.append("")
    for s in snap["strategies"]:
        lines += [f"## {s['key']} {s['name']}　排序依据：{s['rank_rule']}", ""]
        if not s["candidates"]:
            lines += ["**今日无信号**", ""]
            continue
        lines += [f"触发 {s['n_signal']} 只，最多买 {cfg.max_positions} 只：", ""]
        lines.append("| 排名 | 代码 | 名称 | 收盘 | 当日涨幅 | 排序分数 | 动作 |")
        lines.append("|---|---|---|---|---|---|---|")
        for c in s["candidates"]:
            score = "—" if c["score"] is None else f"{c['score']:.4f}"
            action = ("买入" if status["on"] else "待命") if c["selected"] else "排队"
            lines.append(
                f"| {c['rank']} | {c['code']} | {c['name']} | {c['close']:.2f} | "
                f"{c['day_chg_pct']:+.2f}% | {score} | {action} |")
        lines.append("")

    if analysis and analysis.get("top1"):
        lines += [f"## 排序规则有效性（后验 {cfg.fwd_days} 个交易日）", ""]
        lines.append("| 策略 | 排序第1 均值 | 全信号均值 | 超额 | 跑赢比例 | t 值 | 判定 |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in analysis["top1"]:
            if r.get("insufficient"):
                lines.append(f"| {r['key']} | — | — | — | — | — | 样本不足 |")
                continue
            lines.append(
                f"| {r['key']} {r['name']} | {r['top1_mean_pct']:+.2f}% | "
                f"{r['all_mean_pct']:+.2f}% | {r['excess_pp']:+.2f}pp | "
                f"{r['top1_winrate_pct']}% | {r['t_stat']:+.2f} | {r['verdict']} |")
        lines += ["", "> t 值绝对值 < 2 说明超额在统计上不显著，排序与随机挑选差别不大。"]

    (out_dir / "signal_snapshot.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  今日状态：{'ON 允许开仓' if status['on'] else 'OFF 禁止开仓'} "
          f"（已 {status['days']} 个交易日），"
          f"信号数：{ {s['key']: s['n_signal'] for s in snap['strategies']} }")
    return payload


def regime_series(cfg: Config) -> pd.Series:
    return engine.load_regime(cfg)
