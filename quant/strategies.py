"""策略定义：5 个纯技术面选股策略。

约定（与 engine 一致）：
  enter / exit  布尔矩阵（当日收盘确认，次日开盘执行）
  rank          排序分数，越大越优先；同一交易日信号数超过持仓上限时按它挑
  max_hold      最长持有交易日数，None 表示不限制
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .indicators import bollinger, donchian, rsi_frame


@dataclass(frozen=True)
class Strategy:
    key: str
    name: str
    prefix: str
    kind: str
    rank_desc: str
    max_hold: int | None


STRATEGIES: tuple[Strategy, ...] = (
    Strategy("S1", "双均线趋势跟踪", "s1_ma_cross", "趋势跟随",
             "近20日涨幅（越强越优先）", None),
    Strategy("S2", "唐奇安通道突破", "s2_donchian_breakout", "趋势突破",
             "近20日涨幅（越强越优先）", None),
    Strategy("S3", "RSI超卖反转", "s3_rsi_reversion", "均值回归",
             "RSI 由低到高（越超卖越优先）", 20),
    Strategy("S4", "布林下轨回归", "s4_bollinger_reversion", "均值回归",
             "收盘偏离下轨幅度（跌得越狠越优先）", 30),
    Strategy("S5", "量价放量突破", "s5_volume_breakout", "事件驱动",
             "当日涨幅（越猛越优先）", 15),
)

BY_KEY = {s.key: s for s in STRATEGIES}


def build_signals(frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, pd.DataFrame]]:
    close, high, low, volume = frames["close"], frames["high"], frames["low"], frames["volume"]

    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ret20 = close / close.shift(20) - 1.0
    pct_chg = close / close.shift(1) - 1.0
    vol_ma20 = volume.rolling(20).mean()

    upper20, lower10 = donchian(high, low, up=20, dn=10)
    lower_band, mid, _ = bollinger(close, 20, 2.0)
    rsi = rsi_frame(close, 14)

    signals: dict[str, dict[str, pd.DataFrame]] = {
        "S1": {
            "enter": (ma20.shift(1) <= ma60.shift(1)) & (ma20 > ma60),
            "exit": (ma20.shift(1) >= ma60.shift(1)) & (ma20 < ma60),
            "rank": ret20,
        },
        "S2": {
            "enter": close > upper20,
            "exit": close < lower10,
            "rank": ret20,
        },
        "S3": {
            "enter": rsi < 30,
            "exit": rsi > 60,
            "rank": -rsi,
        },
        "S4": {
            "enter": close < lower_band,
            "exit": close > mid,
            "rank": -(close / lower_band - 1.0),
        },
        "S5": {
            "enter": (pct_chg > 0.03) & (volume > 2 * vol_ma20),
            "exit": close < ma10,
            "rank": pct_chg,
        },
    }

    for spec in signals.values():
        for key in ("enter", "exit", "rank"):
            spec[key] = spec[key].reindex_like(close).astype(float)
    return signals
