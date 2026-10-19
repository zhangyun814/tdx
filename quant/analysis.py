"""排序有效性检验：回答「同一天多只票触发，怎么确定哪只更好」。

三个层次：
  A. top1 口径   —— 只买排序第 1 名，是否优于当天全部信号的平均
  B. top5 口径   —— 实盘买满 5 只，对比 6 种排序规则
  C. 分层检验     —— 第 6 名之后（排队名单）是不是真的更差，并控制样本偏差
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .strategies import STRATEGIES, Strategy

RULES = ("current", "reverse", "lowvol", "volratio", "relstr", "random")
RULE_LABEL = {
    "current": "当前规则",
    "reverse": "反向",
    "lowvol": "低波动优先",
    "volratio": "量比最高",
    "relstr": "相对强度",
    "random": "随机",
}


def _zscore(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def _paired_t(diff: np.ndarray) -> float:
    if len(diff) < 2:
        return 0.0
    sd = diff.std(ddof=1)
    return float(diff.mean() / (sd / np.sqrt(len(diff)))) if sd > 0 else 0.0


def _factor_panels(frames: dict[str, pd.DataFrame], cfg: Config) -> dict[str, pd.DataFrame]:
    close, volume = frames["close"], frames["volume"]
    pct = close.pct_change()
    mom20 = close / close.shift(20) - 1.0
    vola20 = pct.rolling(20).std()
    vol_ratio = volume / volume.rolling(20).mean()

    from . import datasource
    try:
        idx_close = datasource.load_index_close(cfg).reindex(close.index).ffill()
        idx_mom20 = idx_close / idx_close.shift(20) - 1.0
        relstr = mom20.sub(idx_mom20, axis=0)
    except Exception:  # noqa: BLE001 - 指数缺失时该因子降级
        relstr = mom20
    return {
        "current": pd.DataFrame(),
        "reverse": pd.DataFrame(),
        "lowvol": -vola20,
        "volratio": vol_ratio,
        "relstr": relstr,
        "random": pd.DataFrame(),
    }


# ---------------------------------------------------------------------------
# A. top1 口径
# ---------------------------------------------------------------------------
def _rankdata(x: np.ndarray) -> np.ndarray:
    """平均秩（与 pandas.Series.rank() 一致），用于计算 Spearman IC。"""
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def top1_skill(cfg: Config, frames: dict[str, pd.DataFrame],
               signals: dict[str, dict[str, pd.DataFrame]],
               eval_start: str, eval_end: str) -> list[dict]:
    close = frames["close"]
    dates = close.index
    fwd_df = close.shift(-cfg.fwd_days) / close - 1.0
    fwd = fwd_df.to_numpy()
    in_range = (dates >= pd.Timestamp(eval_start)) & (dates <= pd.Timestamp(eval_end))
    day_idx = np.flatnonzero(in_range)

    rows = []
    for st in STRATEGIES:
        E = signals[st.key]["enter"].fillna(0).astype(bool)
        valid = (E & close.notna() & fwd_df.notna()).to_numpy()
        R = signals[st.key]["rank"].to_numpy()

        top_rets, bench_rets, ic_list = [], [], []
        for i in day_idx:
            idx = np.flatnonzero(valid[i])
            if idx.size < 2:
                continue
            r = R[i, idx]
            ok = ~np.isnan(r)
            if ok.sum() < 2:
                continue
            r2 = r[ok]
            f2 = fwd[i, idx][ok]
            top_rets.append(float(f2[np.argmax(r2)]))
            bench_rets.append(float(f2.mean()))
            ic = _spearman(r2, f2)
            if not np.isnan(ic):
                ic_list.append(ic)

        if len(top_rets) < 5:
            rows.append({"key": st.key, "name": st.name, "rank_rule": st.rank_desc,
                         "n_days": len(top_rets), "insufficient": True})
            continue

        top = np.array(top_rets)
        bench = np.array(bench_rets)
        diff = top - bench
        rows.append({
            "key": st.key, "name": st.name, "rank_rule": st.rank_desc,
            "n_days": len(top),
            "top1_mean_pct": round(float(top.mean()) * 100, 2),
            "all_mean_pct": round(float(bench.mean()) * 100, 2),
            "excess_pp": round(float(diff.mean()) * 100, 2),
            "top1_winrate_pct": round(float((diff > 0).mean()) * 100, 1),
            "t_stat": round(_paired_t(diff), 2),
            "mean_ic": round(float(np.mean(ic_list)), 3) if ic_list else None,
            "ic_positive_share_pct": round(float(np.mean([x > 0 for x in ic_list]) * 100), 1) if ic_list else None,
            "insufficient": False,
        })
    return rows


# ---------------------------------------------------------------------------
# B. top5 口径：6 种排序规则
# ---------------------------------------------------------------------------
def compare_rules(cfg: Config, frames: dict[str, pd.DataFrame],
                  signals: dict[str, dict[str, pd.DataFrame]],
                  eval_start: str, eval_end: str) -> list[dict]:
    close = frames["close"]
    dates = close.index
    fwd_df = close.shift(-cfg.fwd_days) / close - 1.0
    fwd = fwd_df.to_numpy()
    in_range = ((dates >= pd.Timestamp(eval_start)) & (dates <= pd.Timestamp(eval_end)))
    day_idx = np.flatnonzero(in_range)
    factors = _factor_panels(frames, cfg)

    k = cfg.max_positions
    trials = max(1, cfg.random_trials)
    rng = np.random.default_rng(20261019)
    rows: list[dict] = []

    for st in STRATEGIES:
        E = signals[st.key]["enter"].fillna(0).astype(bool)
        valid = (E & close.notna() & fwd_df.notna()).to_numpy()
        base = signals[st.key]["rank"].to_numpy()
        rank_arrays = {
            "current": base,
            "reverse": -base,
            "lowvol": factors["lowvol"].to_numpy(),
            "volratio": factors["volratio"].to_numpy(),
            "relstr": factors["relstr"].to_numpy(),
        }
        acc: dict[str, tuple[list[float], list[float]]] = {r: ([], []) for r in RULES}

        for i in day_idx:
            idx = np.flatnonzero(valid[i])
            if idx.size == 0:
                continue
            f = fwd[i, idx]
            bench = float(f.mean())
            for rule in RULES:
                if rule == "random":
                    n = min(k, idx.size)
                    sample = np.empty(trials)
                    for t in range(trials):
                        sample[t] = f[rng.choice(idx.size, size=n, replace=False)].mean()
                    top = float(sample.mean())
                else:
                    r = rank_arrays[rule][i, idx]
                    ok = ~np.isnan(r)
                    if not ok.any():
                        continue
                    pos = np.flatnonzero(ok)
                    sel = pos[np.argsort(-r[ok], kind="mergesort")[:k]]
                    top = float(f[sel].mean())
                acc[rule][0].append(top)
                acc[rule][1].append(bench)

        for rule in RULES:
            top = np.array(acc[rule][0], dtype=float)
            bench = np.array(acc[rule][1], dtype=float)
            if len(top) < 10:
                continue
            diff = top - bench
            rows.append({
                "strategy": st.key, "name": st.name, "rule": rule,
                "rule_label": RULE_LABEL[rule],
                "top_mean_pct": round(float(top.mean()) * 100, 2),
                "bench_mean_pct": round(float(bench.mean()) * 100, 2),
                "excess_pp": round(float(diff.mean()) * 100, 2),
                "win_pct": round(float((diff > 0).mean()) * 100, 1),
                "t_stat": round(_paired_t(diff), 2),
                "n_days": int(len(top)),
            })
    return rows


# ---------------------------------------------------------------------------
# C. 分层检验（控制样本偏差）
# ---------------------------------------------------------------------------
def band_analysis(cfg: Config, frames: dict[str, pd.DataFrame],
                  signals: dict[str, dict[str, pd.DataFrame]],
                  eval_start: str, eval_end: str) -> tuple[list[dict], list[dict]]:
    close = frames["close"]
    dates = close.index
    fwd_df = close.shift(-cfg.fwd_days) / close - 1.0
    fwd = fwd_df.to_numpy()
    in_range = ((dates >= pd.Timestamp(eval_start)) & (dates <= pd.Timestamp(eval_end)))
    day_idx = np.flatnonzero(in_range)
    min_cand = cfg.min_candidates

    bands_out: list[dict] = []
    confound_out: list[dict] = []

    for st in STRATEGIES:
        E = signals[st.key]["enter"].fillna(0).astype(bool)
        valid = (E & close.notna() & fwd_df.notna()).to_numpy()
        R = signals[st.key]["rank"].to_numpy()

        bands: dict[str, list[float]] = {"1-5": [], "6-10": [], "11+": []}
        dup_a, dup_b = [], []
        cand_cnt: list[int] = []

        for i in day_idx:
            idx = np.flatnonzero(valid[i])
            if idx.size == 0:
                continue
            r = R[i, idx]
            ok = ~np.isnan(r)
            if ok.sum() < min_cand:
                continue
            pos = np.flatnonzero(ok)
            f = fwd[i, idx]
            order = pos[np.argsort(-r[ok], kind="mergesort")]
            cand_cnt.append(int(order.size))
            bands["1-5"].append(float(f[order[:5]].mean()))
            bands["6-10"].append(float(f[order[5:10]].mean()))
            if order.size > 10:
                bands["11+"].append(float(f[order[10:]].mean()))
            if order.size >= 10:
                dup_a.append(float(f[order[:5]].mean()))
                dup_b.append(float(f[order[5:10]].mean()))

        if not bands["1-5"]:
            bands_out.append({"key": st.key, "name": st.name, "n_days": 0})
            continue

        rec = {"key": st.key, "name": st.name,
               "n_days": len(bands["1-5"]),
               "avg_candidates": round(float(np.mean(cand_cnt)), 1)}
        for band, vals in bands.items():
            rec[band] = round(float(np.mean(vals) * 100), 2) if vals else None
            rec[f"n_{band}"] = len(vals)
        if dup_a:
            a, b = np.array(dup_a), np.array(dup_b)
            rec["top5_beat_6_10_pct"] = round(float((a > b).mean()) * 100, 1)
            rec["top5_minus_6_10_pp"] = round(float((a - b).mean()) * 100, 2)
        bands_out.append(rec)

        # 混淆检验：信号密集日 vs 稀疏日的后验收益
        counts, avg_rets = [], []
        for i in day_idx:
            idx = np.flatnonzero(valid[i])
            if idx.size < 2:
                continue
            counts.append(int(idx.size))
            avg_rets.append(float(fwd[i, idx].mean()))
        if counts:
            c_arr, r_arr = np.array(counts), np.array(avg_rets)
            hi = r_arr[c_arr >= np.percentile(c_arr, 75)]
            lo = r_arr[c_arr <= np.percentile(c_arr, 25)]
            confound_out.append({
                "key": st.key, "name": st.name,
                "dense_pct": round(float(hi.mean()) * 100, 2),
                "sparse_pct": round(float(lo.mean()) * 100, 2),
                "diff_pp": round(float((hi.mean() - lo.mean()) * 100), 2),
            })
    return bands_out, confound_out


# ---------------------------------------------------------------------------
# 结论生成（不写死数字，随数据变化）
# ---------------------------------------------------------------------------
def verdict_rank(rec: dict) -> tuple[str, str]:
    """排序规则是否有效的文字判定。"""
    if rec.get("insufficient") or rec.get("t_stat") is None:
        return "样本不足", "信号太稀疏，排序几乎不起作用"
    excess, t = rec["excess_pp"], rec["t_stat"]
    win = rec.get("top1_winrate_pct") or 0
    if t >= 2 and excess > 0:
        return "有效", f"超额 {excess:+.2f}pp、t={t}，排序确实挑得比平均好"
    if t <= -2:
        return "方向反了", f"超额 {excess:+.2f}pp、t={t}，反着排更好"
    if win < 47:
        return "接近无效", f"跑赢比例仅 {win}%、t={t}，排序更接近噪声"
    return "弱有效", f"超额 {excess:+.2f}pp 但 t={t}<2，统计上不显著"


def verdict_band(rec: dict) -> tuple[str, str]:
    n_days = rec.get("n_days") or 0
    a, b, c = rec.get("1-5"), rec.get("6-10"), rec.get("11+")
    if n_days == 0:
        return "样本不足", "没有满足条件的交易日"
    if a is None or b is None:
        return "样本不足", "候选数不足 10 只，无法分层"
    if n_days < 15:
        return "样本不足", f"仅 {n_days} 个交易日满足条件，不足以判断"

    tail_vals = [v for v in (b, c) if v is not None]
    tail = sum(tail_vals) / len(tail_vals)
    spread = a - tail
    if spread > 0.3:
        if c is not None and c < b - 0.15:
            return "单调递减", f"越靠前越好：前 5 名 {a:+.2f}%、之后平均 {tail:+.2f}%，排序有梯度"
        return "前 5 名有优势", f"前 5 名 {a:+.2f}% 比之后 {tail:+.2f}% 高 {spread:.2f}pp，第 6 名之后基本持平"
    if spread < -0.3:
        return "正好相反", f"第 5 名之后反而更好：前 5 名 {a:+.2f}%、之后平均 {tail:+.2f}%，排序方向错了"
    return "无差别", f"前 5 名 {a:+.2f}% 与之后 {tail:+.2f}% 基本持平，队内次序没有信息"


def run(cfg: Config, frames: dict[str, pd.DataFrame],
        signals: dict[str, dict[str, pd.DataFrame]],
        eval_start: str, eval_end: str, out_dir: Path) -> dict:
    print("  [1/3] top1 口径：排序第 1 vs 当日信号均值")
    top1 = top1_skill(cfg, frames, signals, eval_start, eval_end)
    pd.DataFrame(top1).to_csv(out_dir / "sort_skill.csv", index=False, encoding="utf-8")

    print("  [2/3] top5 口径：6 种排序规则横向对比")
    rules = compare_rules(cfg, frames, signals, eval_start, eval_end)
    pd.DataFrame(rules).to_csv(out_dir / "sort_rule_compare.csv", index=False, encoding="utf-8")
    (out_dir / "sort_rule_compare.json").write_text(
        json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  [3/3] 分层检验：只看当日候选 >= {cfg.min_candidates} 只的交易日")
    bands, confound = band_analysis(cfg, frames, signals, eval_start, eval_end)
    pd.DataFrame(bands).to_csv(out_dir / "rank_band_controlled.csv", index=False, encoding="utf-8")

    for row in top1:
        v, note = verdict_rank(row)
        row["verdict"], row["verdict_note"] = v, note
    for row in bands:
        v, note = verdict_band(row)
        row["verdict"], row["verdict_note"] = v, note

    payload = {"fwd_days": cfg.fwd_days, "min_candidates": cfg.min_candidates,
               "top1": top1, "rules": rules, "bands": bands, "confound": confound}
    (out_dir / "rank_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for row in top1:
        if row.get("insufficient"):
            print(f"    {row['key']}: 样本不足")
        else:
            print(f"    {row['key']} 超额 {row['excess_pp']:+.2f}pp t={row['t_stat']:+.2f} "
                  f"→ {row['verdict']}")
    return payload
