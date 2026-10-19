"""回测引擎：A股多头、次日开盘成交、T+1、涨跌停不可成交、可选市场状态过滤。

时间口径（全套统一）：
  第 i 日收盘   → 用收盘价确认信号 / 确认市场状态
  第 i+1 日开盘 → 按开盘价成交
因此任何决策都不会用到未来信息。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import datasource
from .config import Config
from .vendor.export_results import export_results


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def to_std_symbol(code: str) -> str:
    """sh600519 -> 600519.SH"""
    if code.startswith("sh"):
        return f"{code[2:]}.SH"
    if code.startswith("sz"):
        return f"{code[2:]}.SZ"
    return code


def limit_pct(code: str) -> float:
    body = code[2:] if code[:2] in ("sh", "sz") else code
    if body.startswith(("300", "688")):
        return 0.20
    if body.startswith(("8", "4")):
        return 0.30
    return 0.10


def resolve_window(cfg: Config, index: pd.DatetimeIndex) -> tuple[str, str]:
    """评估区间：起点取配置，终点默认取数据最后一天。"""
    start = cfg.eval_start.strip()
    end = cfg.eval_end.strip() or index.max().strftime("%Y-%m-%d")
    if start > end:
        raise ValueError(f"评估区间非法：{start} > {end}")
    if pd.Timestamp(end) < index.min() or pd.Timestamp(start) > index.max():
        raise ValueError(
            f"评估区间 [{start}, {end}] 与数据范围 "
            f"[{index.min().date()}, {index.max().date()}] 没有交集"
        )
    return start, end


def build_regime(cfg: Config, dates: pd.DatetimeIndex) -> pd.Series:
    """市场状态：基准指数收盘价 > 自身 N 日均线 → ON(1)，否则 OFF(0)。

    当日收盘价与当日均线比较，收盘确认、次日生效。
    """
    idx = datasource.load_index_close(cfg)
    ma = idx.rolling(cfg.regime_window).mean()
    state = (idx > ma).astype(float).where(ma.notna(), np.nan)
    state = state.reindex(dates).ffill()
    state.to_frame("regime_on").to_csv(cfg.data_path / "regime_state.csv")
    return state


def load_regime(cfg: Config) -> pd.Series:
    path = cfg.data_path / "regime_state.csv"
    if not path.exists():
        raise FileNotFoundError(f"缺少市场状态文件 {path}，请先执行 backtest。")
    s = pd.read_csv(path, parse_dates=["date"])
    return s.set_index("date")["regime_on"]


# ---------------------------------------------------------------------------
# 单策略回测
# ---------------------------------------------------------------------------
def run_backtest(
    cfg: Config,
    frames: dict[str, pd.DataFrame],
    sig: dict[str, pd.DataFrame],
    name: str,
    prefix: str,
    out_dir: Path,
    max_hold: int | None = None,
    regime: pd.Series | None = None,
    regime_exit: bool = False,
) -> dict:
    open_, high, low, close, volume = (
        frames["open"], frames["high"], frames["low"], frames["close"], frames["volume"],
    )
    names = datasource.load_names(cfg)
    eval_start, eval_end = resolve_window(cfg, close.index)

    dates = list(close.index)
    symbols = list(close.columns)
    start_idx = next(i for i, d in enumerate(dates) if d.strftime("%Y-%m-%d") >= eval_start)
    end_idx = max(i for i, d in enumerate(dates) if d.strftime("%Y-%m-%d") <= eval_end)

    enter = sig["enter"].to_numpy()
    exit_ = sig["exit"].to_numpy()
    rank = sig["rank"].to_numpy()
    o = open_.to_numpy()
    c = close.to_numpy()

    reg = None
    if regime is not None:
        reg = regime.reindex(close.index).fillna(0.0).to_numpy()

    cash = cfg.initial_cash
    positions: dict[int, dict] = {}
    last_close = np.full(len(symbols), np.nan)
    for j in range(start_idx):
        row = c[j]
        valid = ~np.isnan(row)
        last_close[valid] = row[valid]

    prev_close_arr = last_close.copy()   # 首日也能正确判断涨跌停
    pending_buy: list[int] = []
    pending_sell: set[int] = set()
    equity_curve: list[dict] = []
    trades: list[dict] = []

    for i in range(start_idx, end_idx + 1):
        date = dates[i]

        # ---------- 1. 执行昨日挂单：卖出 ----------
        # force_exit 是"持续挂单"：当日跌停/停牌卖不出，次日继续挂
        sell_today = set(pending_sell) | {j for j, p in positions.items() if p.get("force_exit")}
        for j in sorted(sell_today):
            if j not in positions:
                continue
            pos = positions[j]
            if i <= pos["entry_bar"]:          # T+1
                continue
            px = o[i, j]
            if np.isnan(px) or px <= 0:
                continue                        # 停牌
            pc = prev_close_arr[j]
            lim = limit_pct(symbols[j])
            if not np.isnan(pc) and px <= pc * (1 - lim) * 0.995:
                continue                        # 开盘跌停，卖不出
            size = pos["size"]
            gross = size * px
            fee = gross * (cfg.sell_comm + cfg.sell_tax)
            cash += gross - fee
            cost = size * pos["entry_price"] * (1 + cfg.buy_comm)
            pnl = gross - fee - cost
            trades.append({
                "entry_date": pos["entry_date"],
                "exit_date": date.strftime("%Y-%m-%d"),
                "side": "long",
                "size": int(size),
                "symbol": to_std_symbol(symbols[j]),
                "symbol_name": names.get(symbols[j], ""),
                "entry_price": round(pos["entry_price"], 4),
                "exit_price": round(px, 4),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl / cost * 100, 4),
                "holding_bars": int(i - pos["entry_bar"]),
            })
            del positions[j]
        pending_sell.clear()

        # ---------- 2. 执行昨日挂单：买入 ----------
        equity_now = cash + sum(
            p["size"] * (last_close[j] if not np.isnan(last_close[j]) else p["entry_price"])
            for j, p in positions.items()
        )
        for j in pending_buy:
            if i == end_idx:                    # 最后一根 K 线不再开新仓
                break
            if len(positions) >= cfg.max_positions:
                break
            if j in positions:
                continue
            px = o[i, j]
            if np.isnan(px) or px <= 0:
                continue
            pc = prev_close_arr[j]
            lim = limit_pct(symbols[j])
            if not np.isnan(pc) and px >= pc * (1 + lim) * 0.995:
                continue                        # 开盘涨停，买不进
            target = equity_now * cfg.weight
            size = int(target / (px * (1 + cfg.buy_comm)) // cfg.lot) * cfg.lot
            if size <= 0:
                continue
            cost = size * px * (1 + cfg.buy_comm)
            if cost > cash:
                size = int(cash / (px * (1 + cfg.buy_comm)) // cfg.lot) * cfg.lot
                if size <= 0:
                    continue
                cost = size * px * (1 + cfg.buy_comm)
            cash -= cost
            positions[j] = {
                "size": size,
                "entry_price": px,
                "entry_bar": i,
                "entry_date": date.strftime("%Y-%m-%d"),
            }
        pending_buy = []

        # ---------- 3. 收盘估值 ----------
        for j in range(len(symbols)):
            if not np.isnan(c[i, j]):
                last_close[j] = c[i, j]
        equity = cash + sum(p["size"] * last_close[j] for j, p in positions.items())
        equity_curve.append({"date": date.strftime("%Y-%m-%d"), "value": round(equity, 2)})

        # ---------- 4. 收盘后生成信号 ----------
        if i == end_idx:
            prev_close_arr = np.where(np.isnan(c[i]), prev_close_arr, c[i])
            continue

        for j, pos in list(positions.items()):
            if i <= pos["entry_bar"]:
                continue
            want_exit = bool(exit_[i, j]) if not math.isnan(exit_[i, j]) else False
            if max_hold is not None and (i - pos["entry_bar"]) >= max_hold:
                want_exit = True
            if want_exit:
                pending_sell.add(j)

        # 市场状态刚转为 OFF：全部持仓标记强制清仓（次日开盘起持续挂单）
        regime_off_today = reg is not None and reg[i] == 0
        if regime_exit and regime_off_today and i > 0 and reg[i - 1] == 1:
            for j in positions:
                positions[j]["force_exit"] = True
                pending_sell.add(j)

        candidates = []
        for j in range(len(symbols)):
            if j in positions or j in pending_sell:
                continue
            e = enter[i, j]
            if e is None or math.isnan(e) or e <= 0:
                continue
            if np.isnan(c[i, j]):
                continue
            candidates.append((rank[i, j] if not math.isnan(rank[i, j]) else -1e9, j))
        candidates.sort(reverse=True)
        pending_buy = [j for _, j in candidates[: cfg.max_positions * 4]]
        if regime_off_today:
            pending_buy = []

        prev_close_arr = np.where(np.isnan(c[i]), prev_close_arr, c[i])

    # ---------- 5. 期末强制平仓 ----------
    last_i = end_idx
    for j, pos in list(positions.items()):
        px = c[last_i, j] if not np.isnan(c[last_i, j]) else last_close[j]
        size = pos["size"]
        gross = size * px
        fee = gross * (cfg.sell_comm + cfg.sell_tax)
        cash += gross - fee
        cost = size * pos["entry_price"] * (1 + cfg.buy_comm)
        pnl = gross - fee - cost
        trades.append({
            "entry_date": pos["entry_date"],
            "exit_date": dates[last_i].strftime("%Y-%m-%d"),
            "side": "long",
            "size": int(size),
            "symbol": to_std_symbol(symbols[j]),
            "symbol_name": names.get(symbols[j], ""),
            "entry_price": round(pos["entry_price"], 4),
            "exit_price": round(px, 4),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / cost * 100, 4),
            "holding_bars": int(last_i - pos["entry_bar"]),
        })
        del positions[j]
    if equity_curve:
        equity_curve[-1]["value"] = round(cash, 2)

    files = export_results(
        equity_curve=equity_curve,
        trade_history=trades,
        prefix=prefix,
        initial_cash=cfg.initial_cash,
        start=eval_start,
        end=eval_end,
        market="china_a",
        output_dir=out_dir,
        strategy_name=name,
        symbol=f"{cfg.universe_index} 成分股",
        is_flat_at_end=True,
    )
    summary = json.loads(Path(files["summary"]).read_text(encoding="utf-8"))
    return {
        "prefix": prefix,
        "name": name,
        "equity": equity_curve,
        "trades": trades,
        "summary": summary["summary"],
        "meta": summary["meta"],
        "files": {k: str(v) for k, v in files.items()},
        "eval_start": eval_start,
        "eval_end": eval_end,
    }


# ---------------------------------------------------------------------------
# 基准
# ---------------------------------------------------------------------------
def benchmark_curves(cfg: Config, frames: dict[str, pd.DataFrame],
                     eval_start: str, eval_end: str) -> dict[str, list[dict]]:
    close = frames["close"]
    dates = [d for d in close.index if eval_start <= d.strftime("%Y-%m-%d") <= eval_end]
    if not dates:
        return {}

    sub = close.loc[dates]
    norm = sub.ffill() / sub.ffill().iloc[0]
    ew = norm.mean(axis=1) * cfg.initial_cash
    curves = {
        "等权持有成分股": [
            {"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 2)} for d, v in ew.items()
        ]
    }

    try:
        idx = datasource.load_index_close(cfg)
    except datasource.DataSourceError:
        return curves

    idx = idx[(idx.index >= pd.Timestamp(eval_start)) & (idx.index <= pd.Timestamp(eval_end))]
    if len(idx) > 1:
        base = idx.iloc[0]
        curves["沪深300指数"] = [
            {"date": d.strftime("%Y-%m-%d"), "value": round(float(v) / base * cfg.initial_cash, 2)}
            for d, v in idx.items()
        ]
    return curves


# ---------------------------------------------------------------------------
# 辅助指标
# ---------------------------------------------------------------------------
def exposure_pct(trades: pd.DataFrame, dates: pd.DatetimeIndex,
                 eval_start: str, eval_end: str) -> float:
    """有持仓的交易日占比（%），衡量资金参赛时间。"""
    pos = {d: i for i, d in enumerate(dates)}
    lo = pos.get(pd.Timestamp(eval_start))
    hi = pos.get(pd.Timestamp(eval_end))
    if lo is None or hi is None:
        return 0.0
    held: set[int] = set()
    for _, r in trades.iterrows():
        a = pos.get(pd.Timestamp(r.entry_date))
        b = pos.get(pd.Timestamp(r.exit_date))
        if a is None or b is None:
            continue
        held.update(range(a, b))
    total = max(hi - lo, 1)
    return len([d for d in held if lo <= d <= hi]) / total * 100


def regime_diagnostics(cfg: Config, regime: pd.Series, dates: pd.DatetimeIndex,
                       eval_start: str, eval_end: str) -> dict:
    r = regime.reindex(dates).ffill()
    mask = (r.index >= pd.Timestamp(eval_start)) & (r.index <= pd.Timestamp(eval_end))
    ev = r[mask]

    idx = datasource.load_index_close(cfg).reindex(dates).ffill()
    ev_idx = idx[(idx.index >= pd.Timestamp(eval_start)) & (idx.index <= pd.Timestamp(eval_end))]

    yearly = []
    for yr, chunk in ev.groupby(ev.index.year):
        ic = ev_idx[ev_idx.index.year == yr]
        yearly.append({
            "year": int(yr),
            "days": int(len(chunk)),
            "on_days": int((chunk == 1).sum()),
            "on_share_pct": round(float((chunk == 1).mean() * 100), 1),
            "index_ret_pct": round(float(ic.iloc[-1] / ic.iloc[0] - 1) * 100, 1) if len(ic) > 1 else None,
        })

    return {
        "window": cfg.regime_window,
        "on_share_pct": round(float((ev == 1).mean() * 100), 1),
        "off_days": int((ev == 0).sum()),
        "switches": int((ev.diff() == -1).sum()),
        "yearly": yearly,
    }
