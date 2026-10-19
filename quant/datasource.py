"""数据层：调用 westock-data CLI 拉日线、解析 markdown、构建面板。

数据源是可替换的：只要提供同样格式的 markdown 日线文件（date/open/close/high/low/volume），
本模块后面的部分就不依赖任何外部服务。
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .config import Config

ROW_RE = re.compile(
    r"^\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*([-\d.]+)\s*\|\s*([-\d.]+)\s*\|\s*([-\d.]+)\s*\|"
    r"\s*([-\d.]+)\s*\|\s*([-\d.]+)\s*\|\s*([-\d.]+)\s*\|\s*([-\d.]*)\s*\|"
)
CONSTITUENT_RE = re.compile(r"^\|\s*((?:sh|sz)\d{6})\s*\|\s*([^|]*?)\s*\|")

# 常见的 westock-data 安装位置（自动探测用）
_SEARCH_ROOTS = [
    Path.home() / ".workbuddy" / "plugins",
    Path.home() / ".workbuddy" / "skills",
    Path.home() / ".codebuddy" / "plugins",
]


class DataSourceError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# 外部 CLI 定位
# ---------------------------------------------------------------------------
def resolve_westock(cfg: Config) -> Path:
    if cfg.westock_dir:
        p = Path(cfg.westock_dir).expanduser()
        if (p / "scripts" / "index.js").exists():
            return p
        raise DataSourceError(f"westock_dir 指向的目录里没有 scripts/index.js：{p}")

    for root in _SEARCH_ROOTS:
        if not root.exists():
            continue
        for hit in root.rglob("westock-data/scripts/index.js"):
            return hit.parent.parent
    raise DataSourceError(
        "找不到 westock-data 数据源。请在 config.toml 里设置 [data] westock_dir = \"...\"，"
        "或设置环境变量 WESTOCK_DATA_DIR 指向包含 scripts/index.js 的目录。"
    )


def resolve_node(cfg: Config) -> str:
    if cfg.node_bin:
        return cfg.node_bin
    found = shutil.which("node")
    if found:
        return found
    # 兜底：WorkBuddy 托管运行时
    for cand in sorted(Path.home().glob(".workbuddy/binaries/node/versions/*/bin/node"), reverse=True):
        if cand.exists():
            return str(cand)
    raise DataSourceError("找不到 node 可执行文件，请在 config.toml 的 [data] node_bin 中指定路径。")


def _run(cfg: Config, args: list[str], timeout: int = 120) -> str:
    ws = resolve_westock(cfg)
    node = resolve_node(cfg)
    proc = subprocess.run(
        [node, "scripts/index.js", *args],
        cwd=str(ws),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise DataSourceError(f"数据源命令失败：{' '.join(args)}\n{proc.stderr[:400]}")
    return proc.stdout


# ---------------------------------------------------------------------------
# 股票池
# ---------------------------------------------------------------------------
def fetch_universe(cfg: Config) -> pd.DataFrame:
    """取指数成分股清单，保存为 data/universe.csv。"""
    out = _run(cfg, ["index", cfg.universe_index], timeout=180)
    rows = []
    for line in out.splitlines():
        m = CONSTITUENT_RE.match(line.strip())
        if m:
            rows.append({"code": m.group(1), "name": m.group(2).strip()})
    if not rows:
        raise DataSourceError("未能解析出成分股清单，数据源返回格式可能已变化。")
    df = pd.DataFrame(rows).drop_duplicates("code")
    df.to_csv(cfg.data_path / "universe.csv", index=False, encoding="utf-8")
    return df


def load_universe(cfg: Config) -> pd.DataFrame:
    p = cfg.data_path / "universe.csv"
    if p.exists():
        return pd.read_csv(p, dtype={"code": str})
    return pd.DataFrame(columns=["code", "name"])


def load_names(cfg: Config) -> dict[str, str]:
    df = load_universe(cfg)
    if df.empty:
        return {}
    return dict(zip(df["code"], df["name"].fillna("")))


# ---------------------------------------------------------------------------
# 日线
# ---------------------------------------------------------------------------
def parse_kline_file(path: Path) -> pd.DataFrame:
    rows = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        m = ROW_RE.match(line.strip())
        if not m:
            continue
        rows.append({
            "date": m.group(1),
            "open": float(m.group(2)),
            "close": float(m.group(3)),
            "high": float(m.group(4)),
            "low": float(m.group(5)),
            "volume": float(m.group(6)),
            "amount": float(m.group(7)),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset="date").sort_values("date")
    return df.reset_index(drop=True)


def _last_bar_date(path: Path) -> str:
    """快速读取一个日线文件里的最新交易日。

    数据源文件是新在前、老在后（也可能被别的数据源改成升序），
    所以头尾各扫一段取日期最大值，两种排列都能覆盖，且不用解析整份文件。
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            head = fh.read(8192).decode("utf-8", errors="ignore")
            if size > 8192:
                fh.seek(max(0, size - 8192))
                tail = fh.read().decode("utf-8", errors="ignore")
            else:
                tail = ""
    except OSError:
        return ""
    dates = re.findall(r"\|\s*(\d{4}-\d{2}-\d{2})\s*\|", head + tail)
    return max(dates) if dates else ""


def _download_one(cfg: Config, code: str, target_date: str = "") -> tuple[str, bool, str]:
    """下载单只票。target_date 为「应达到的最新交易日」，缓存落后于它才重下。"""
    target = cfg.raw_path / f"{code}.md"
    if target.exists() and target.stat().st_size > 2000 and not cfg.refresh:
        if not target_date or _last_bar_date(target) >= target_date:
            return code, True, "cached"
    for attempt in range(2):
        try:
            text = _run(cfg, ["kline", code, "--period", "day",
                              "--limit", str(cfg.max_bars), "--fq", "qfq"], timeout=120)
            target.write_text(text, encoding="utf-8")
            df = parse_kline_file(target)
            if len(df) >= 2:
                return code, True, f"{len(df)} bars"
        except Exception as exc:  # noqa: BLE001
            if attempt == 1:
                return code, False, str(exc)[:120]
    return code, False, "解析为空"


def _refresh_benchmark(cfg: Config) -> str:
    """先刷新基准指数（只有 1 只，成本极低），用它最后一天当作「应有最新日」。"""
    code = cfg.universe_index
    target = cfg.raw_path / f"{code}.md"
    try:
        text = _run(cfg, ["kline", code, "--period", "day",
                          "--limit", str(cfg.max_bars), "--fq", "qfq"], timeout=120)
        target.write_text(text, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        if not target.exists():
            raise DataSourceError(f"基准指数 {code} 下载失败：{exc}") from exc
        print(f"  ⚠️ 基准指数刷新失败，沿用本地文件（{exc}）")
    return _last_bar_date(target)


def fetch_klines(cfg: Config, codes: list[str], workers: int = 6) -> dict[str, str]:
    """并发下载日线，返回 {code: 状态}。

    默认增量更新：缓存文件的最新日期已经追平基准指数最后一个交易日就跳过，
    落后才重下 —— 这样每天跑一次既不会重复下载，也不会拿到过期数据。
    """
    target_date = "" if cfg.refresh else _refresh_benchmark(cfg)
    if target_date:
        print(f"  应有最新交易日：{target_date}（基准 {cfg.universe_index}）")

    status: dict[str, str] = {}
    total = len(codes)
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_download_one, cfg, c, target_date): c for c in codes}
        for fut in as_completed(futures):
            code, ok, msg = fut.result()
            status[code] = ("ok " + msg) if ok else ("fail " + msg)
            done += 1
            if done % 25 == 0 or done == total:
                print(f"  下载进度 {done}/{total}", flush=True)

    cached = sum(1 for s in status.values() if s.endswith("cached"))
    if cached:
        print(f"  复用缓存 {cached} 只，更新 {total - cached} 只")
    bad = [c for c, s in status.items() if s.startswith("fail")]
    if bad:
        print(f"  ⚠️ {len(bad)} 只下载失败：{bad[:8]}{' ...' if len(bad) > 8 else ''}")
    return status


def build_panel(cfg: Config) -> pd.DataFrame:
    """把 data/raw/*.md 合并成 data/panel.csv.gz（长表）。"""
    uni = load_universe(cfg)
    codes = list(uni["code"]) if not uni.empty else [
        p.stem for p in cfg.raw_path.glob("*.md") if p.stem[:2] in ("sh", "sz")
    ]
    frames, missing, short = [], [], []
    for code in codes:
        path = cfg.raw_path / f"{code}.md"
        if not path.exists():
            missing.append(code)
            continue
        df = parse_kline_file(path)
        if df.empty or len(df) < cfg.min_bars:
            short.append(code)
            continue
        df["symbol"] = code
        frames.append(df)

    if not frames:
        raise DataSourceError("没有任何可用日线数据，请先执行 fetch。")

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    panel.to_csv(cfg.data_path / "panel.csv.gz", index=False, compression="gzip")

    print(f"  universe={len(codes)} 载入={len(frames)} 缺失={len(missing)} 过短={len(short)}")
    if short:
        print(f"  剔除（<{cfg.min_bars} 根）：{short[:10]}")
    print(f"  面板：{len(panel):,} 行，{panel['date'].min().date()} ~ {panel['date'].max().date()}")
    return panel


def load_panel(cfg: Config) -> dict[str, pd.DataFrame]:
    path = cfg.data_path / "panel.csv.gz"
    if not path.exists():
        raise DataSourceError(f"缺少面板文件 {path}，请先执行 `python run.py fetch`。")
    panel = pd.read_csv(path, parse_dates=["date"])
    out = {}
    for field in ("open", "high", "low", "close", "volume"):
        out[field] = panel.pivot(index="date", columns="symbol", values=field).sort_index()
    return out


def load_index_close(cfg: Config) -> pd.Series:
    """基准指数收盘价（用于市场状态判断与净值对照）。"""
    path = cfg.raw_path / f"{cfg.universe_index}.md"
    if not path.exists():
        raise DataSourceError(f"缺少基准指数数据 {path}，请先执行 `python run.py fetch`。")
    df = parse_kline_file(path)
    if df.empty:
        raise DataSourceError(f"基准指数文件无法解析：{path}")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].astype(float)


def panel_span(cfg: Config) -> tuple[str, str]:
    panel = load_panel(cfg)
    idx = panel["close"].index
    return idx.min().strftime("%Y-%m-%d"), idx.max().strftime("%Y-%m-%d")


def availability(cfg: Config) -> dict:
    """用于 run.py 的启动自检。"""
    info = {"westock_dir": None, "node_bin": None, "universe": 0, "raw_files": 0}
    try:
        info["westock_dir"] = str(resolve_westock(cfg))
    except DataSourceError as exc:
        info["westock_dir"] = f"未找到（{exc}）"
    try:
        info["node_bin"] = resolve_node(cfg)
    except DataSourceError as exc:
        info["node_bin"] = f"未找到（{exc}）"
    info["universe"] = len(load_universe(cfg))
    info["raw_files"] = len(list(cfg.raw_path.glob("*.md")))
    return info


if __name__ == "__main__":  # 便于单独调试
    from .config import load_config as _lc

    _cfg = _lc()
    _cfg.ensure_dirs()
    print(availability(_cfg))
    sys.exit(0)
