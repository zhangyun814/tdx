#!/usr/bin/env python3
"""命令行入口。

用法：
    python run.py pick                # 只选股：补数据 + 出今日候选 + 看板（日常用这个）
    python run.py all                 # 抓数 + 回测 + 检验 + 信号 + 报告（一条命令全流程）
    python run.py fetch               # 只更新数据（成分股 + 日线 + 面板）
    python run.py backtest            # 只用本地数据重跑回测
    python run.py analysis            # 排序规则有效性检验
    python run.py signal              # 当日信号快照
    python run.py report              # 渲染对比报告
    python run.py board               # 渲染信号看板
    python run.py history             # 查看历次运行摘要（年化/回撤/信号数变化）
    python run.py doctor              # 环境自检（数据源 / 数据 / 输出目录）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quant import datasource, pipeline  # noqa: E402
from quant.config import describe, load_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="A股多策略选股回测与信号系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("command", choices=["pick", "all", "fetch", "backtest", "analysis", "signal",
                                       "report", "board", "history", "doctor"],
                   help="要执行的动作")
    p.add_argument("--config", default=None, help="配置文件路径（默认 config.toml）")
    p.add_argument("--start", default=None, help="评估起点，如 2022-07-01")
    p.add_argument("--end", default=None, help="评估终点，默认取数据最后一天")
    p.add_argument("--index", default=None, help="股票池指数代码，如 sh000300")
    p.add_argument("--max-positions", type=int, default=None, help="最大持仓数量")
    p.add_argument("--regime-window", type=int, default=None, help="市场状态均线窗口")
    p.add_argument("--refresh", action="store_true", help="强制重新下载全部日线（默认只补落后的）")
    p.add_argument("--skip-fetch", action="store_true", help="all 时跳过抓数")
    p.add_argument("--no-archive", action="store_true", help="all 时不归档历史快照")
    p.add_argument("--workers", type=int, default=6, help="下载并发数")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    overrides = {
        "eval_start": args.start,
        "eval_end": args.end,
        "universe_index": args.index,
        "max_positions": args.max_positions,
        "regime_window": args.regime_window,
        "refresh": True if args.refresh else None,
    }
    cfg = load_config(args.config, overrides)
    cfg.ensure_dirs()

    if args.command == "doctor":
        info = datasource.availability(cfg)
        print("环境自检")
        print(f"  数据源 CLI : {info['westock_dir']}")
        print(f"  node       : {info['node_bin']}")
        print(f"  股票池数量 : {info['universe']}")
        print(f"  已下载文件 : {info['raw_files']}")
        print(f"  数据目录   : {cfg.data_path}")
        print(f"  输出目录   : {cfg.reports_path}")
        panel = cfg.data_path / "panel.csv.gz"
        if panel.exists():
            try:
                start, end = datasource.panel_span(cfg)
                print(f"  面板区间   : {start} ~ {end}")
            except Exception as exc:  # noqa: BLE001
                print(f"  面板读取失败：{exc}")
        else:
            print("  面板       : 尚未生成（执行 python run.py fetch）")
        print("\n生效参数：")
        for k, v in describe(cfg).items():
            print(f"  {k:18s} = {v}")
        return 0

    try:
        if args.command == "pick":
            pipeline.cmd_pick(cfg, skip_fetch=args.skip_fetch)

        elif args.command == "fetch":
            pipeline.cmd_fetch(cfg, workers=args.workers)

        elif args.command == "backtest":
            pipeline.cmd_backtest(cfg)

        elif args.command == "analysis":
            pipeline.cmd_analysis(cfg)

        elif args.command == "signal":
            pipeline.cmd_signal(cfg)

        elif args.command == "report":
            pipeline.cmd_report(cfg)

        elif args.command == "board":
            pipeline.cmd_board(cfg)

        elif args.command == "history":
            pipeline.cmd_history(cfg)

        elif args.command == "all":
            pipeline.cmd_all(cfg, skip_fetch=args.skip_fetch,
                             no_archive=args.no_archive)
    except KeyboardInterrupt:
        print("\n已中断。")
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌ {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
