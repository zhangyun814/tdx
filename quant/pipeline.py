"""流水线：把抓数、回测、检验、信号、报表串起来。"""

from __future__ import annotations

import json
import pickle
import time
import unicodedata
from pathlib import Path

import pandas as pd

from . import analysis, archive, board, datasource, engine, report, signals
from .config import Config
from .strategies import STRATEGIES, build_signals

MODES = (
    ("base", "原版", {"regime": False, "regime_exit": False}),
    ("reg", "只过滤开仓", {"regime": True, "regime_exit": False}),
    ("regx", "过滤+清仓", {"regime": True, "regime_exit": True}),
)


def _step(title: str) -> None:
    print(f"\n▶ {title}")


# ---------------------------------------------------------------------------
def cmd_fetch(cfg: Config, workers: int = 6) -> None:
    cfg.ensure_dirs()
    _step("抓取股票池")
    uni = datasource.fetch_universe(cfg)
    print(f"  {cfg.universe_index} 成分股 {len(uni)} 只 → {cfg.data_path / 'universe.csv'}")

    _step(f"下载日线（{cfg.max_bars} 根/只，前复权）")
    codes = list(uni["code"]) + [cfg.universe_index]
    datasource.fetch_klines(cfg, codes, workers=workers)

    _step("合并面板")
    datasource.build_panel(cfg)


def cmd_backtest(cfg: Config) -> dict:
    cfg.ensure_dirs()
    out_dir = cfg.reports_path

    _step("加载面板")
    frames = datasource.load_panel(cfg)
    close = frames["close"]
    eval_start, eval_end = engine.resolve_window(cfg, close.index)
    print(f"  {close.shape[0]} 交易日 × {close.shape[1]} 标的　"
          f"评估区间 {eval_start} ~ {eval_end}")

    _step("计算信号与市场状态")
    sig = build_signals(frames)
    regime = engine.build_regime(cfg, close.index)
    mask = (regime.index >= pd.Timestamp(eval_start)) & (regime.index <= pd.Timestamp(eval_end))
    print(f"  ON 占比 {regime[mask].mean() * 100:.1f}%")

    _step("回测：5 策略 × 3 用法")
    grid = []
    for st in STRATEGIES:
        per_mode = {}
        for mode, label, kw in MODES:
            prefix = st.prefix if mode == "base" else f"{st.prefix}_{mode}"
            res = engine.run_backtest(
                cfg, frames, sig[st.key], st.name, prefix, out_dir,
                max_hold=st.max_hold,
                regime=regime if kw["regime"] else None,
                regime_exit=kw["regime_exit"],
            )
            trades = pd.DataFrame(res["trades"])
            exposure = engine.exposure_pct(trades, close.index, eval_start, eval_end) if len(trades) else 0.0
            per_mode[mode] = {
                "key": st.key, "name": st.name, "mode": mode, "mode_label": label,
                "prefix": prefix, "summary": res["summary"], "meta": res["meta"],
                "equity": res["equity"], "exposure_pct": exposure,
            }
            s = res["summary"]
            print(f"  {st.key} {label:6s} 年化 {s['annual_return_pct']:7.2f}% | "
                  f"回撤 {s['max_drawdown_pct']:6.2f}% | 夏普 {s['sharpe']:6.3f} | "
                  f"{s['total_trades']:4d} 笔 | 持仓 {exposure:3.0f}%")
        grid.append({"key": st.key, "name": st.name, "prefix": st.prefix, "modes": per_mode})

    bench = engine.benchmark_curves(cfg, frames, eval_start, eval_end)
    diag = engine.regime_diagnostics(cfg, regime, close.index, eval_start, eval_end)

    runtime = {"grid": grid, "benchmark": bench, "regime": diag,
               "eval_start": eval_start, "eval_end": eval_end}
    with (out_dir / "backtest_runtime.pkl").open("wb") as fh:
        pickle.dump(runtime, fh)

    summary_out = {
        "eval_start": eval_start, "eval_end": eval_end,
        "universe": cfg.universe_index, "regime": diag,
        "grid": [{"key": g["key"], "name": g["name"],
                  "modes": {m: {"summary": v["summary"], "exposure_pct": v["exposure_pct"]}
                            for m, v in g["modes"].items()}} for g in grid],
    }
    (out_dir / "backtest_results.json").write_text(
        json.dumps(summary_out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return runtime


def load_runtime(cfg: Config) -> dict:
    path = cfg.reports_path / "backtest_runtime.pkl"
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}，请先执行 `python run.py backtest`。")
    with path.open("rb") as fh:
        return pickle.load(fh)


def cmd_analysis(cfg: Config, runtime: dict | None = None) -> dict:
    _step("排序有效性检验")
    runtime = runtime or load_runtime(cfg)
    frames = datasource.load_panel(cfg)
    sig = build_signals(frames)
    return analysis.run(cfg, frames, sig, runtime["eval_start"], runtime["eval_end"],
                        cfg.reports_path)


def cmd_signal(cfg: Config, analysis_payload: dict | None = None) -> dict:
    _step("生成当日信号快照")
    frames = datasource.load_panel(cfg)
    sig = build_signals(frames)
    if analysis_payload is None:                     # 单独执行时复用已有检验结果
        cached = cfg.reports_path / "rank_analysis.json"
        if cached.exists():
            analysis_payload = json.loads(cached.read_text(encoding="utf-8"))
    return signals.write(cfg, frames, sig, cfg.reports_path, analysis_payload)


def cmd_report(cfg: Config, runtime: dict | None = None) -> Path:
    _step("渲染对比报告")
    runtime = runtime or load_runtime(cfg)
    regime = engine.load_regime(cfg)
    out = report.build_dashboard(cfg, runtime["grid"], runtime["benchmark"], runtime["regime"],
                                 regime, cfg.reports_path,
                                 runtime["eval_start"], runtime["eval_end"])
    print(f"  仪表盘 → {out}")
    return out


def cmd_board(cfg: Config) -> Path:
    _step("渲染信号看板")
    out = board.build(cfg, cfg.reports_path)
    print(f"  看板 → {out}")
    return out


def cmd_all(cfg: Config, skip_fetch: bool = False, no_archive: bool = False) -> None:
    t0 = time.time()
    if not skip_fetch:
        cmd_fetch(cfg)
    else:
        _step("跳过抓数，直接使用本地数据")

    runtime = cmd_backtest(cfg)
    analysis_payload = cmd_analysis(cfg, runtime)
    signal_payload = cmd_signal(cfg, analysis_payload)
    cmd_report(cfg, runtime)
    cmd_board(cfg)

    if not no_archive:
        _step("归档本次运行")
        info = archive.summarize(runtime, signal_payload)
        gap = archive.diff_last(cfg, info)
        archive.snapshot_run(cfg, runtime, signal_payload)
        if gap:
            print(f"  对比上次：{gap}")
        print(f"  历史索引 → {cfg.reports_path / 'history.csv'}")

    print(f"\n✅ 全部完成，用时 {time.time() - t0:.1f}s，结果在 {cfg.reports_path}")


def cmd_history(cfg: Config, limit: int = 15) -> None:
    path = cfg.reports_path / "history.csv"
    if not path.exists():
        print("还没有历史记录，先跑一次 `python run.py all`。")
        return

    def pad(text: str, width: int) -> str:
        """按显示宽度对齐（中文算 2 格）。"""
        text = str(text)
        w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
        return text + " " * max(0, width - w)

    df = pd.read_csv(path, encoding="utf-8")
    show = df.tail(limit)
    print(f"\n历次运行（最近 {len(show)} 次，共 {len(df)} 次）\n")
    print(pad("运行时间", 20) + pad("数据截至", 12) + pad("状态", 6)
          + pad("年化", 9) + pad("回撤", 9) + pad("夏普", 8) + pad("信号", 6) + "最优策略")
    print("-" * 92)
    for _, r in show.iterrows():
        state = "ON" if str(r.get("regime_on")) == "True" else "OFF"
        print(pad(r["run_at"], 20) + pad(r["eval_end"], 12) + pad(state, 6)
              + pad(f"{r['avg_annual_pct']:.2f}%", 9) + pad(f"{r['avg_dd_pct']:.2f}%", 9)
              + pad(f"{r['avg_sharpe']:.3f}", 8) + pad(int(r["n_signals"]), 6)
              + f"{r['best_strategy']}（年化 {r['best_annual_pct']:.2f}%）")
    print(f"\n快照目录：{cfg.reports_path / 'runs'}")
