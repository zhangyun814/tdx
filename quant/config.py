"""配置：默认值 + config.toml 覆盖 + 命令行覆盖。"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config.toml"


@dataclass
class Config:
    # ---- 数据 ----
    westock_dir: str = ""          # westock-data CLI 目录，留空自动探测
    node_bin: str = ""             # node 可执行文件，留空自动探测
    universe_index: str = "sh000300"   # 股票池：指数成分股
    max_bars: int = 1300           # 单只股票最多拉多少根日线
    min_bars: int = 250            # 少于该根数剔除（次新股）
    refresh: bool = False          # True = 重新下载已有文件

    # ---- 回测参数 ----
    initial_cash: float = 1_000_000.0
    max_positions: int = 5
    weight: float = 0.20
    buy_comm: float = 0.0003
    sell_comm: float = 0.0003
    sell_tax: float = 0.0005
    lot: int = 100
    eval_start: str = "2022-07-01"  # 评估起点（此前数据仅用于指标预热）
    eval_end: str = ""              # 评估终点，留空 = 数据最后一个交易日

    # ---- 市场状态过滤 ----
    regime_window: int = 60

    # ---- 排序规则检验 ----
    fwd_days: int = 20              # 后验收益观察窗口（交易日）
    min_candidates: int = 15        # 分层检验时要求当日候选数 >= 该值
    random_trials: int = 200        # 随机基准的抽样次数

    # ---- 输出 ----
    data_dir: str = ""              # 留空 = 项目下 data/
    reports_dir: str = ""           # 留空 = 项目下 reports/
    keep_runs: int = 60             # reports/runs/ 保留几份历史快照，0 = 不清理

    # ------------------------------------------------------------------
    @property
    def data_path(self) -> Path:
        return Path(self.data_dir).expanduser() if self.data_dir else PROJECT_ROOT / "data"

    @property
    def raw_path(self) -> Path:
        return self.data_path / "raw"

    @property
    def reports_path(self) -> Path:
        return Path(self.reports_dir).expanduser() if self.reports_dir else PROJECT_ROOT / "reports"

    def ensure_dirs(self) -> None:
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.reports_path.mkdir(parents=True, exist_ok=True)

    def eval_end_or(self, fallback: str) -> str:
        return self.eval_end.strip() or fallback


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    """读取 config.toml（若存在），再用 overrides 覆盖。未知键会被忽略。"""
    cfg = Config()
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if cfg_path.exists() and tomllib is not None:
        with cfg_path.open("rb") as fh:
            raw = tomllib.load(fh)
        flat: dict[str, object] = {}
        for section in raw.values():
            if isinstance(section, dict):
                flat.update(section)
        valid = {f.name for f in fields(Config)}
        for key, value in flat.items():
            if key in valid:
                setattr(cfg, key, value)

    for key, value in (overrides or {}).items():
        if value is not None and hasattr(cfg, key):
            setattr(cfg, key, value)

    # 环境变量优先级最高，方便 CI / 临时替换数据源
    if os.environ.get("WESTOCK_DATA_DIR"):
        cfg.westock_dir = os.environ["WESTOCK_DATA_DIR"]
    if os.environ.get("QUANT_NODE_BIN"):
        cfg.node_bin = os.environ["QUANT_NODE_BIN"]
    return cfg


def describe(cfg: Config) -> dict:
    d = asdict(cfg)
    d["data_dir"] = str(cfg.data_path)
    d["reports_dir"] = str(cfg.reports_path)
    return d
