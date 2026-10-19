"""运行归档：每次执行把 reports/ 里的产物复制一份带时间戳的快照，并记一行历史。

设计取舍：
- `reports/` 根目录始终保持「最新一次」的位置不变（用户习惯直接打开 regime_report.html）
- 历史产物进 `reports/runs/<时间戳>/`，索引在 `reports/history.csv`
- 只归档产物（html/png/csv/json/md），不归档 .pkl —— 单次 700KB，攒几十次没必要
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import Config

SNAPSHOT_SUFFIXES = (".html", ".png", ".csv", ".json", ".md")
HISTORY_COLUMNS = [
    "run_at", "as_of", "eval_start", "eval_end",
    "regime_on", "avg_annual_pct", "avg_dd_pct", "avg_sharpe",
    "best_strategy", "best_annual_pct", "n_signals", "run_dir",
]


def _reports_root(cfg: Config) -> Path:
    return cfg.reports_path


def _prune(runs_dir: Path, keep: int) -> None:
    if keep <= 0:
        return
    dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], reverse=True)
    for old in dirs[keep:]:
        shutil.rmtree(old, ignore_errors=True)


def summarize(runtime: dict, signal_payload: dict | None) -> dict:
    """从 runtime 里抽一行摘要，用于 history.csv 与跨次对比。"""
    regx = [g["modes"]["regx"] for g in runtime["grid"]]
    annual = [m["summary"]["annual_return_pct"] for m in regx]
    dd = [m["summary"]["max_drawdown_pct"] for m in regx]
    sharpe = [m["summary"]["sharpe"] for m in regx]
    best_i = max(range(len(regx)), key=lambda i: sharpe[i])

    n_signals = None
    regime_on = None
    as_of = None
    if signal_payload:
        n_signals = sum(s["n_signal"] for s in signal_payload["snapshot"]["strategies"])
        regime_on = bool(signal_payload["status"]["on"])
        as_of = signal_payload.get("as_of")

    return {
        "as_of": as_of,
        "eval_start": runtime["eval_start"],
        "eval_end": runtime["eval_end"],
        "regime_on": regime_on,
        "avg_annual_pct": round(sum(annual) / len(annual), 2),
        "avg_dd_pct": round(sum(dd) / len(dd), 2),
        "avg_sharpe": round(sum(sharpe) / len(sharpe), 3),
        "best_strategy": f"{regx[best_i]['key']} {regx[best_i]['name']}",
        "best_annual_pct": round(annual[best_i], 2),
        "n_signals": n_signals,
    }


def snapshot_run(cfg: Config, runtime: dict, signal_payload: dict | None,
                 tag: str = "") -> Path:
    """把当前 reports/ 的产物复制进带时间戳的子目录，并追加一行 history.csv。"""
    root = _reports_root(cfg)
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{tag}" if tag else stamp
    run_dir = runs_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for f in root.iterdir():
        if not f.is_file() or f.suffix.lower() not in SNAPSHOT_SUFFIXES:
            continue
        if f.name == "history.csv":
            continue
        shutil.copy2(f, run_dir / f.name)
        copied += 1

    info = summarize(runtime, signal_payload)
    info["run_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info["run_dir"] = str(run_dir.relative_to(root))
    (run_dir / "run_meta.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    hist_path = root / "history.csv"
    row = pd.DataFrame([info])[HISTORY_COLUMNS]
    if hist_path.exists():
        old = pd.read_csv(hist_path, encoding="utf-8")
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(hist_path, index=False, encoding="utf-8")

    _prune(runs_dir, cfg.keep_runs)
    print(f"  归档 {copied} 个文件 → {run_dir}")
    return run_dir


def diff_last(cfg: Config, info: dict) -> str | None:
    """跟历史里上一条（同区间）比一下，输出人话摘要。"""
    hist_path = _reports_root(cfg) / "history.csv"
    if not hist_path.exists():
        return None
    df = pd.read_csv(hist_path, encoding="utf-8")
    if len(df) < 2:
        return None
    prev = df.iloc[-2]
    if str(prev.get("eval_end")) == str(info.get("eval_end")):
        return None                          # 同一天同一区间重跑，比较没意义

    def arrow(cur, old, unit=""):
        delta = float(cur) - float(old)
        return f"{float(old):.2f}{unit} → {float(cur):.2f}{unit}（{delta:+.2f}）"

    return (f"上次 {prev['run_at']}（数据到 {prev['eval_end']}）："
            f"年化 {arrow(info['avg_annual_pct'], prev['avg_annual_pct'], '%')}，"
            f"夏普 {arrow(info['avg_sharpe'], prev['avg_sharpe'])}，"
            f"信号数 {int(prev['n_signals'])} → {info['n_signals']}")
