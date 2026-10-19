"""对比报告：三种用法（原版 / 只过滤 / 过滤+清仓）的图表与仪表盘。

所有文案里的数字都从本次回测结果计算，重跑后不会出现"文字与图表不一致"。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .config import Config  # noqa: E402
from .vendor.render_dashboard import build_dashboard_data, render_dashboard  # noqa: E402

for _cand in ("PingFang SC", "Heiti SC", "Songti SC", "Arial Unicode MS", "STHeiti",
              "Noto Sans CJK SC", "Microsoft YaHei"):
    if any(f.name == _cand for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _cand
        break
plt.rcParams["axes.unicode_minus"] = False

MODES = ("base", "reg", "regx")
MODE_LABEL = {"base": "原版", "reg": "只过滤开仓", "regx": "过滤+清仓"}
MODE_COLOR = {"base": "#9aa0a6", "reg": "#4a90d9", "regx": "#e8743b"}


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def fmt(v, digits: int = 2, suffix: str = "") -> str:
    if v is None:
        return "—"
    try:
        return f"{v:,.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "—"


def nav_series(equity: list[dict], base: float = 100.0) -> pd.Series:
    s = pd.Series([p["value"] for p in equity],
                  index=pd.to_datetime([p["date"] for p in equity]))
    return s / s.iloc[0] * base


def yearly_return(equity: list[dict]) -> pd.Series:
    s = nav_series(equity)
    last_by_year = s.groupby(s.index.year).last()
    prev = last_by_year.shift(1)
    prev.iloc[0] = 100.0
    return (last_by_year / prev - 1) * 100


def off_spans(regime: pd.Series, lo, hi, min_len: int = 1):
    s = regime.loc[lo:hi]
    spans, start = [], None
    for d, v in s.items():
        if v == 0 and start is None:
            start = d
        elif v == 1 and start is not None:
            spans.append((start, d, len(s.loc[start:d])))
            start = None
    if start is not None:
        spans.append((start, s.index[-1], len(s.loc[start:])))
    return [(a, b) for a, b, n in spans if n >= min_len]


def slim(s: pd.Series, max_n: int = 300) -> pd.Series:
    if len(s) <= max_n:
        return s
    step = len(s) / (max_n - 1)
    idx = sorted(set([int(round(i * step)) for i in range(max_n - 1)] + [len(s) - 1]))
    return s.iloc[idx]


def points_of(s: pd.Series) -> list[dict]:
    s = slim(s)
    return [{"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 4)} for d, v in s.items()]


def _avg(grid: list[dict], mode: str, key: str):
    vals = [g["modes"][mode]["summary"].get(key) for g in grid]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


# ---------------------------------------------------------------------------
# 图表
# ---------------------------------------------------------------------------
def chart_before_after(grid, out: Path) -> None:
    keys = [g["key"] for g in grid]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), dpi=140)
    for ax, metric, title in ((axes[0], "annual_return_pct", "年化收益（%）"),
                              (axes[1], "max_drawdown_pct", "最大回撤（%）")):
        x = range(len(keys))
        w = 0.26
        flat = []
        for k, m in enumerate(MODES):
            vals = [g["modes"][m]["summary"].get(metric) or 0 for g in grid]
            flat += vals
            bars = ax.bar([i + (k - 1) * w for i in x], vals, w,
                          label=MODE_LABEL[m], color=MODE_COLOR[m], alpha=0.9)
            for b, v in zip(bars, vals):
                if metric == "max_drawdown_pct":
                    ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v),
                                ha="center", va="center", fontsize=6.6, rotation=90,
                                color="#ffffff" if m != "base" else "#333333",
                                xytext=(0, 11), textcoords="offset points")
                else:
                    ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v),
                                ha="center", va="bottom" if v >= 0 else "top",
                                fontsize=7, rotation=90,
                                xytext=(0, 3 if v >= 0 else -3), textcoords="offset points")
        ax.set_xticks(list(x))
        ax.set_xticklabels(keys, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax.axhline(0, color="#444", lw=0.8)
        lo_, hi_ = min(flat + [0]), max(flat + [0])
        if metric == "max_drawdown_pct":
            ax.set_ylim(0, hi_ * 1.10)
            ax.invert_yaxis()
        else:
            pad = (hi_ - lo_) * 0.24
            ax.set_ylim(lo_ - pad, hi_ + pad)
    axes[0].legend(fontsize=8, ncol=3, loc="upper left")
    fig.suptitle("市场状态过滤前后：收益与回撤（同池、同成本口径）", fontsize=12)
    plt.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def chart_nav_grid(grid, bench, regime, out: Path, lo, hi) -> None:
    fig = plt.figure(figsize=(13.2, 7.4), dpi=140)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.13], hspace=0.42, wspace=0.22)
    axes = [fig.add_subplot(gs[r, c]) for r in (0, 1) for c in range(3)]
    ribbon = fig.add_subplot(gs[2, :], sharex=axes[0])

    idx_nav = nav_series(bench["沪深300指数"]) if "沪深300指数" in bench else None

    for ax, g in zip(axes, grid):
        for m in MODES:
            s = nav_series(g["modes"][m]["equity"])
            ax.plot(s.index, s.values, lw=1.35, color=MODE_COLOR[m],
                    label=MODE_LABEL[m], ls="--" if m == "base" else "-")
        if idx_nav is not None:
            ax.plot(idx_nav.index, idx_nav.values, lw=1.0, color="#7f7f7f", ls=":", label="沪深300")
        ax.set_title(f"{g['key']} {g['name'][:5]}", fontsize=10)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax = axes[len(grid)]
    ax.axis("off")
    handles = [plt.Line2D([], [], color=MODE_COLOR[m], lw=1.6,
                          ls="--" if m == "base" else "-", label=MODE_LABEL[m]) for m in MODES]
    handles += [plt.Line2D([], [], color="#7f7f7f", lw=1.2, ls=":", label="沪深300"),
                Patch(facecolor="#2e7d32", alpha=0.55, label="状态 ON（可开仓）"),
                Patch(facecolor="#c9c9c9", alpha=0.75, label="状态 OFF（禁止开仓）")]
    ax.legend(handles=handles, loc="center", fontsize=10, frameon=False)

    st = regime.loc[lo:hi]
    ribbon.axvspan(lo, hi, color="#2e7d32", alpha=0.5, lw=0)
    for a, b in off_spans(st, st.index[0], st.index[-1], min_len=1):
        ribbon.axvspan(a, b, color="#c9c9c9", alpha=0.98, lw=0)
    ribbon.set_xlim(lo, hi)
    ribbon.set_ylim(0, 1)
    ribbon.set_yticks([])
    ribbon.set_ylabel("市场状态", fontsize=8, rotation=0, ha="right", va="center")
    ribbon.tick_params(labelsize=8)
    ribbon.xaxis.set_major_locator(mdates.YearLocator())
    ribbon.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for sp in ribbon.spines.values():
        sp.set_visible(False)

    fig.suptitle("五策略净值：原版 / 只过滤 / 过滤+清仓（起点=100，底部为指数相对均线的状态带）",
                 fontsize=12)
    fig.subplots_adjust(top=0.90, bottom=0.06, left=0.05, right=0.985)
    fig.savefig(out)
    plt.close(fig)


def chart_yearly(grid, diag, out: Path) -> None:
    years = [r["year"] for r in diag["yearly"]]
    if not years:
        return
    base_avg, regx_avg, idx_ret, on_share = [], [], [], []
    for r in diag["yearly"]:
        yr = r["year"]
        base_avg.append(sum(float(yearly_return(g["modes"]["base"]["equity"]).get(yr, 0))
                            for g in grid) / len(grid))
        regx_avg.append(sum(float(yearly_return(g["modes"]["regx"]["equity"]).get(yr, 0))
                            for g in grid) / len(grid))
        idx_ret.append(r["index_ret_pct"] or 0)
        on_share.append(r["on_share_pct"])

    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=140)
    x = range(len(years))
    w = 0.26
    ax.bar([i - w for i in x], base_avg, w, label="五策略平均·原版", color=MODE_COLOR["base"])
    ax.bar(list(x), regx_avg, w, label="五策略平均·过滤+清仓", color=MODE_COLOR["regx"])
    ax.bar([i + w for i in x], idx_ret, w, label="沪深300", color="#78909c",
           edgecolor="#37474f", hatch="//", lw=0.6)
    for i, (b, r) in enumerate(zip(base_avg, regx_avg)):
        ax.annotate(f"{b:.0f}", (i - w, b), ha="center", va="bottom" if b >= 0 else "top",
                    fontsize=7, xytext=(0, 2 if b >= 0 else -2), textcoords="offset points")
        ax.annotate(f"{r:.0f}", (i, r), ha="center", va="bottom" if r >= 0 else "top",
                    fontsize=7, xytext=(0, 2 if r >= 0 else -2), textcoords="offset points")

    ax2 = ax.twinx()
    ax2.plot(list(x), on_share, color="#2e7d32", marker="o", ms=4, lw=1.4, label="状态 ON 占比")
    ax2.set_ylim(0, 130)
    ax2.set_ylabel("状态 ON 交易日占比（%）", fontsize=9, color="#2e7d32")
    ax2.tick_params(labelsize=8, colors="#2e7d32")

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(y) for y in years])
    ax.set_ylabel("年度收益（%）")
    ax.axhline(0, color="#444", lw=0.8)
    ax.grid(axis="y", alpha=0.25)
    vals = base_avg + regx_avg + idx_ret
    lo_, hi_ = min(vals + [0]), max(vals + [0])
    pad = (hi_ - lo_) * 0.22
    ax.set_ylim(lo_ - pad, hi_ + pad)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, ncol=4, loc="upper left")
    ax.set_title("分年度：过滤在哪些年份加分、哪些年份拖后腿", fontsize=12)
    plt.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 仪表盘
# ---------------------------------------------------------------------------
def build_dashboard(cfg: Config, grid: list[dict], bench: dict, diag: dict,
                    regime: pd.Series, out_dir: Path, eval_start: str, eval_end: str) -> Path:
    lo, hi = pd.Timestamp(eval_start), pd.Timestamp(eval_end)

    chart_before_after(grid, out_dir / "regime_before_after.png")
    chart_nav_grid(grid, bench, regime, out_dir / "regime_nav.png", lo, hi)
    chart_yearly(grid, diag, out_dir / "regime_yearly.png")

    def cell(v, digits=2, suffix="", tone=False):
        d = {"main": fmt(v, digits, suffix)}
        if tone and v is not None:
            d["raw"] = v
        return d

    def compare_table(metric_key, title, subtitle, digits=2, suffix="%", tone=False):
        rows = []
        for g in grid:
            b = g["modes"]["base"]["summary"].get(metric_key)
            r = g["modes"]["reg"]["summary"].get(metric_key)
            x = g["modes"]["regx"]["summary"].get(metric_key)
            delta = (x - b) if (x is not None and b is not None) else None
            rows.append({
                "metric": f"{g['key']} {g['name']}",
                "values": [
                    cell(b, digits, suffix, tone), cell(r, digits, suffix, tone),
                    cell(x, digits, suffix, tone),
                    {"main": ("+" if (delta or 0) > 0 else "") + fmt(delta, digits, "pp")},
                ],
            })
        b_avg, r_avg, x_avg = (_avg(grid, "base", metric_key), _avg(grid, "reg", metric_key),
                               _avg(grid, "regx", metric_key))
        d_avg = (x_avg - b_avg) if (b_avg is not None and x_avg is not None) else None
        rows.append({
            "metric": "五策略平均",
            "values": [cell(b_avg, digits, suffix, tone), cell(r_avg, digits, suffix, tone),
                       cell(x_avg, digits, suffix, tone),
                       {"main": ("+" if (d_avg or 0) > 0 else "") + fmt(d_avg, digits, "pp")}],
        })
        return {"type": "metric_table", "tab": "compare", "title": title,
                "subtitle": subtitle,
                "columns": ["策略", MODE_LABEL["base"], MODE_LABEL["reg"],
                            MODE_LABEL["regx"], "变化（清仓 vs 原版）"],
                "rows": rows}

    # ---- 净值曲线 ----
    series = [{"name": f"{g['key']} {g['name'][:4]}",
               "points": points_of(nav_series(g["modes"]["regx"]["equity"]))} for g in grid]
    if "沪深300指数" in bench:
        series.append({"name": "沪深300", "points": points_of(nav_series(bench["沪深300指数"]))})

    # ---- 分年度表 ----
    year_rows = []
    for r in diag["yearly"]:
        yr = r["year"]
        b = sum(float(yearly_return(g["modes"]["base"]["equity"]).get(yr, 0)) for g in grid) / len(grid)
        x = sum(float(yearly_return(g["modes"]["regx"]["equity"]).get(yr, 0)) for g in grid) / len(grid)
        year_rows.append({"metric": str(yr), "values": [
            cell(r["index_ret_pct"], 1, "%"),
            {"main": f"{r['on_share_pct']:.0f}%"},
            cell(b, 1, "%"), cell(x, 1, "%"),
        ]})

    # ---- 动态结论 ----
    b_ann, x_ann = _avg(grid, "base", "annual_return_pct"), _avg(grid, "regx", "annual_return_pct")
    b_dd, x_dd = _avg(grid, "base", "max_drawdown_pct"), _avg(grid, "regx", "max_drawdown_pct")
    b_sh, x_sh = _avg(grid, "base", "sharpe"), _avg(grid, "regx", "sharpe")
    r_sh = _avg(grid, "reg", "sharpe")
    r_dd = _avg(grid, "reg", "max_drawdown_pct")
    x_expo = sum(g["modes"]["regx"]["exposure_pct"] for g in grid) / len(grid)

    by_gain = sorted(grid, key=lambda g: (g["modes"]["regx"]["summary"]["annual_return_pct"] or -999)
                     - (g["modes"]["base"]["summary"]["annual_return_pct"] or -999), reverse=True)
    best = max(grid, key=lambda g: g["modes"]["regx"]["summary"].get("sharpe") or -9)
    worst = min(grid, key=lambda g: (g["modes"]["regx"]["summary"]["annual_return_pct"] or -999))

    yr_rows = []
    for r in diag["yearly"]:
        yr = r["year"]
        b = sum(float(yearly_return(g["modes"]["base"]["equity"]).get(yr, 0)) for g in grid) / len(grid)
        x = sum(float(yearly_return(g["modes"]["regx"]["equity"]).get(yr, 0)) for g in grid) / len(grid)
        yr_rows.append({"year": yr, "base": b, "regx": x, "delta": x - b,
                        "index": r["index_ret_pct"], "on": r["on_share_pct"]})
    yr_best = max(yr_rows, key=lambda r: r["delta"]) if yr_rows else None
    yr_worst = min(yr_rows, key=lambda r: r["delta"]) if yr_rows else None

    def line_change(g) -> str:
        s_b, s_x = g["modes"]["base"]["summary"], g["modes"]["regx"]["summary"]
        return (f"- {g['key']} {g['name']}：年化 {fmt(s_b['annual_return_pct'])}% → "
                f"{fmt(s_x['annual_return_pct'])}%，夏普 {fmt(s_b['sharpe'], 3)} → "
                f"{fmt(s_x['sharpe'], 3)}，回撤 {fmt(s_b['max_drawdown_pct'])}% → "
                f"{fmt(s_x['max_drawdown_pct'])}%")

    conclusion = "\n".join([
        f"**一句话结论**：五策略平均年化 {fmt(b_ann)}% → {fmt(x_ann)}%，"
        f"最大回撤 {fmt(b_dd)}% → {fmt(x_dd)}%，夏普 {fmt(b_sh, 3)} → {fmt(x_sh, 3)}。"
        f"代价是资金在场时间被压缩到 {x_expo:.0f}%。",
        "",
        f"**加分最多的策略**：{by_gain[0]['key']} {by_gain[0]['name']}",
        line_change(by_gain[0]),
        "",
        "**各策略变化（原版 → 过滤+清仓）**：",
        *[line_change(g) for g in grid],
        "",
        f"**只过滤开仓是不够的**：平均夏普 {fmt(b_sh, 3)} → {fmt(r_sh, 3)}，"
        f"平均回撤 {fmt(b_dd)}% → {fmt(r_dd)}%。不开新仓却让旧持仓继续留在下跌段里，"
        f"只挡住了新错误、没挡住旧错误，必须配合强制清仓才兑现回撤改善。",
        "",
        f"**不是全年有效**：{yr_best['year']} 年加分最多"
        f"（平均 {yr_best['base']:+.1f}% vs {yr_best['regx']:+.1f}%，"
        f"ON 占比 {yr_best['on']:.0f}%）；"
        + (f"{yr_worst['year']} 年反而拖后腿"
           f"（平均 {yr_worst['base']:+.1f}% vs {yr_worst['regx']:+.1f}%，"
           f"ON 占比 {yr_worst['on']:.0f}%）。" if yr_worst else "")
        + "均线能识别趋势性下跌，但对震荡市和单边急涨都会反复打脸。",
        "",
        f"**当前最优组合**：{best['key']} {best['name']} + 过滤开仓 + 强制清仓，"
        f"年化 {fmt(best['modes']['regx']['summary']['annual_return_pct'])}%，"
        f"回撤 {fmt(best['modes']['regx']['summary']['max_drawdown_pct'])}%，"
        f"夏普 {fmt(best['modes']['regx']['summary']['sharpe'], 3)}。"
        + (f"最差的是 {worst['key']} {worst['name']}，"
           f"年化 {fmt(worst['modes']['regx']['summary']['annual_return_pct'])}%——"
           f"均值回归类不宜共用这套开关。" if worst is not best else ""),
    ])

    regime_text = "\n".join([
        "**规则定义**：",
        f"- 状态判断：基准指数当日收盘价 > 该指数 {diag['window']} 日均线 → ON（允许开新仓）；否则 OFF",
        "- 判断用当日收盘价与当日均线，收盘确认、次日开盘执行，与策略信号同一时间口径，不含未来信息",
        "- 只过滤：OFF 时不开新仓，已持仓仍按原策略规则退出",
        "- 过滤+清仓：OFF 时不开新仓，且状态由 ON 转 OFF 当日收盘挂单，"
        "次日开盘起持续挂单清仓（遇跌停/停牌顺延，直到卖出为止）",
        "",
        f"**区间统计**：ON 占比 {diag['on_share_pct']:.0f}%，OFF 交易日 {diag['off_days']} 天，"
        f"ON→OFF 切换 {diag['switches']} 次。",
        "",
        "切换次数越多，震荡市里的来回进出越频繁，滑点与冲击成本对实盘的侵蚀越大——这是过滤最现实的代价。",
    ])

    assumptions = "\n".join([
        f"- 标的池：{cfg.universe_index} 当前成分股（剔除上市不足 {cfg.min_bars} 根 K 线的标的），存在幸存者偏差",
        f"- 区间：{eval_start} ~ {eval_end}，之前的数据仅用于指标预热",
        "- 执行：信号当日收盘确认，次日开盘成交；A 股 T+1；100 股整手",
        f"- 仓位：最多同时持有 {cfg.max_positions} 只，单票目标权重 {cfg.weight:.0%}，等权",
        f"- 成本：双边佣金 {cfg.buy_comm * 1e4:.0f}bp，卖出印花税 {cfg.sell_tax * 1e4:.0f}bp，未计滑点与冲击成本",
        "- 涨停不买、跌停不卖；强制清仓为持续挂单，遇跌停/停牌顺延；期末强制平仓",
        "- 三个版本共用同一套引擎与数据，唯一差别就是市场状态判断",
    ])

    limitations = "\n".join([
        "- **幸存者偏差**：用当前成分股回溯历史，收益被系统性高估（三个版本同受影响）",
        "- **夏普不可直接横比**：过滤版有大量空仓日、日收益为 0，波动率被摊薄，夏普天然偏高。"
        "同口径比较时应同时看持仓占比",
        f"- **参数未寻优**：MA{diag['window']} 是教科书默认值，换窗口结果会不同；"
        "本报告只做单点检验，未做参数稳健性扫描",
        f"- **切换成本被低估**：区间内 ON→OFF 切换 {diag['switches']} 次，实盘滑点与择时误差会侵蚀收益",
        "- **单点失效风险**：效果高度依赖「指数均线能识别熊市」这一前提，震荡市与单边急涨都会失效",
        "- **均值回归类不适用**：这类策略需要单独的状态定义，甚至方向应该是反的",
        "- 未考虑融资成本、税费差异、停牌复牌跳空等细节",
    ])

    optimize = "\n".join([
        "- 给均值回归类单独设计状态条件（如「指数在均线下方 + 个股 RSI 极低」才开仓），而不是共用开关",
        "- 对均线窗口（MA20/60/120）做参数与稳健性扫描，检验是否只是运气",
        "- 用波动率目标（按 ATR 缩放仓位）替代一次性清仓，减少来回切换成本",
        "- 把状态条件从单一均线扩展为「均线 + 指数动量 + 行业宽度」的组合，降低假信号",
    ])

    modules = [
        {"type": "line_chart", "tab": "compare",
         "title": "五策略净值（过滤+清仓版）",
         "subtitle": "起点归一为 100；三个版本的逐策略对比见下方各策略页",
         "series": series},
        compare_table("annual_return_pct", "年化收益对比", "单位：%，变化列为百分点差"),
        compare_table("max_drawdown_pct", "最大回撤对比", "单位：%，越小越好；变化列为负表示回撤改善"),
        compare_table("sharpe", "夏普比率对比", "按日收益计算，无风险利率取 0", digits=3, suffix="", tone=True),
        {"type": "text", "tab": "compare", "title": "结论速览", "text": conclusion},
        {"type": "metric_table", "tab": "regime",
         "title": "市场状态逐年分布",
         "subtitle": f"OFF = 基准指数收盘价跌破 {diag['window']} 日均线",
         "columns": ["年度", "沪深300 收益", "ON 交易日占比", "五策略平均·原版", "五策略平均·过滤+清仓"],
         "rows": year_rows},
        {"type": "text", "tab": "regime", "title": "状态规则与统计", "text": regime_text},
        {"type": "text", "tab": "notes", "title": "回测口径与假设", "text": assumptions},
        {"type": "text", "tab": "notes", "title": "已知偏差与局限", "text": limitations},
        {"type": "text", "tab": "notes", "title": "下一步可做的事", "text": optimize},
        {"type": "text", "tab": "notes", "title": "免责声明",
         "text": "⚠️ 本报告由程序基于公开行情数据自动生成，仅供参考，不构成任何投资建议或个股推荐。"
                 "投资有风险，决策需谨慎。"},
    ]

    tabs, tab_modules = [], []
    for i, g in enumerate(grid):
        tab_id = f"s{i + 1}"
        tabs.append({"id": tab_id, "label": f"{g['key']} {g['name']}"})
        sers = [{"name": MODE_LABEL[m], "points": points_of(nav_series(g["modes"][m]["equity"]))}
                for m in MODES]
        if "沪深300指数" in bench:
            sers.append({"name": "沪深300", "points": points_of(nav_series(bench["沪深300指数"]))})
        tab_modules.append({"type": "line_chart", "tab": tab_id,
                            "title": f"{g['key']} {g['name']}：三种用法净值对比",
                            "subtitle": "起点归一为 100", "series": sers})

        rows = []
        for label, key, digits, suffix, tone in (
            ("累计收益", "total_return_pct", 2, "%", False),
            ("年化收益", "annual_return_pct", 2, "%", True),
            ("最大回撤", "max_drawdown_pct", 2, "%", False),
            ("夏普比率", "sharpe", 3, "", True),
            ("交易笔数", "total_trades", 0, "", False),
            ("胜率", "win_rate_pct", 1, "%", False),
        ):
            rows.append({"metric": label, "values": [
                cell(g["modes"][m]["summary"].get(key), digits, suffix, tone) for m in MODES]})
        rows.append({"metric": "持仓交易日占比", "values": [
            {"main": f"{g['modes'][m]['exposure_pct']:.0f}%"} for m in MODES]})
        tab_modules.append({"type": "metric_table", "tab": tab_id, "title": "关键指标",
                            "columns": ["指标"] + [MODE_LABEL[m] for m in MODES], "rows": rows})

    base_prefix = grid[0]["modes"]["base"]["prefix"]
    rep = build_dashboard_data(
        equity_csv=out_dir / f"{base_prefix}_equity.csv",
        trades_csv=out_dir / f"{base_prefix}_trades.csv",
        summary_json=out_dir / f"{base_prefix}_summary.json",
        language="zh",
        market="china_a",
    )
    rep["meta"]["strategy_name"] = f"市场状态过滤对比（五策略 · {cfg.universe_index}）"
    rep["modules"] = modules + tab_modules
    rep["ui"] = {
        "subtitle": f"{cfg.eval_start} ~ {eval_end}　·　{cfg.universe_index} 成分股　·　三种用法同池同成本",
        "active_tab": "compare",
        "tabs": ([{"id": "compare", "label": "过滤前后对比"}] + tabs
                 + [{"id": "regime", "label": "市场状态"}, {"id": "notes", "label": "口径与局限"}]),
        "topbar_menu": [],
        "window_status": "",
        "language": "zh",
        "i18n": rep["ui"]["i18n"],
    }
    out = Path(render_dashboard(rep, output_path=out_dir / "regime_report.html"))
    return out
