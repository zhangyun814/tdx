"""技术指标：全部按列（单只股票）计算，返回与输入同形状的 DataFrame。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi_wilder(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder RSI。前 window 根为 NaN。"""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50.0).where(avg_gain.notna(), np.nan)


def rsi_frame(close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    return pd.DataFrame({c: rsi_wilder(close[c], window) for c in close.columns},
                        index=close.index)


def bollinger(close: pd.DataFrame, window: int = 20, k: float = 2.0):
    mid = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=0)
    return mid - k * std, mid, mid + k * std


def donchian(high: pd.DataFrame, low: pd.DataFrame, up: int = 20, dn: int = 10):
    """通道用「前一根为止」的极值，避免把当日自身算进突破阈值。"""
    upper = high.rolling(up).max().shift(1)
    lower = low.rolling(dn).min().shift(1)
    return upper, lower
