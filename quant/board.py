"""信号看板：今日候选清单 + 排序有效性检验 + 最近实际成交，输出单页 HTML。"""

from __future__ import annotations

import json
from pathlib import Path

from .config import Config

CSS = """
  :root{--bg:#f6f7f9;--card:#fff;--line:#e3e6ea;--ink:#1a1d21;--dim:#8a8f98;
        --up:#c0392b;--dn:#2e7d32;--accent:#1f4e79}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:14px/1.6 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}
  .wrap{max-width:1180px;margin:0 auto;padding:40px 24px 72px}
  h1{font-size:26px;margin:0 0 6px;letter-spacing:-.3px}
  .sub{color:var(--dim);font-size:13px;margin-bottom:8px}
  .lead{background:#fff;border:1px solid var(--line);border-left:3px solid var(--accent);
    border-radius:8px;padding:16px 18px;margin:22px 0 28px;font-size:13.5px;color:#3c4148}
  .lead b{color:var(--ink)}
  .status{background:#fff;border:1px solid var(--line);border-radius:10px;
    padding:14px 18px;margin:18px 0 0;border-left:3px solid #999}
  .status.on{border-left-color:var(--up);background:#fdf7f6}
  .status.off{border-left-color:#5f6b7a;background:#f4f6f8}
  .st-line{display:flex;align-items:center;gap:8px;font-size:15px}
  .st-line .dot{font-size:11px}
  .status.on .st-line .dot{color:var(--up)}
  .status.off .st-line .dot{color:#5f6b7a}
  .st-tag{font-size:11px;font-weight:700;letter-spacing:.5px;border-radius:3px;
    padding:1px 6px;background:#e8ebef;color:#4a5563}
  .status.on .st-tag{background:#fbe9e7;color:var(--up)}
  .st-detail{color:#5a6068;font-size:12.5px;margin-top:4px}
  .st-advice{color:var(--ink);font-size:13px;margin-top:7px}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start}
  @media(max-width:900px){.grid{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 16px 14px}
  .card header{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px}
  .title{font-size:15px;font-weight:600}
  .k{display:inline-block;background:#eef2f6;color:var(--accent);font-weight:700;
    border-radius:4px;padding:1px 6px;margin-right:8px;font-size:12px;letter-spacing:.3px}
  .verdict{font-size:12px;font-weight:600;white-space:nowrap}
  .meta{color:var(--dim);font-size:12px;margin-bottom:10px}
  table{width:100%;border-collapse:collapse;font-size:12.5px}
  th{text-align:left;color:var(--dim);font-weight:500;padding:5px 6px;
    border-bottom:1px solid var(--line);font-size:11.5px;white-space:nowrap}
  td{padding:5px 6px;border-bottom:1px solid #f0f2f4;white-space:nowrap}
  tr.sel{background:#fdf6f5}
  .rank{color:var(--dim);width:26px;font-variant-numeric:tabular-nums}
  .code{font-variant-numeric:tabular-nums;color:#5a6068;letter-spacing:.2px}
  .num{text-align:right;font-variant-numeric:tabular-nums}
  .dim{color:var(--dim)}
  .up{color:var(--up)} .dn{color:var(--dn)}
  tr:last-child td{border-bottom:none}
  tbody tr:nth-child(n+11){display:none}
  .card.expanded tbody tr:nth-child(n+11){display:table-row}
  .tag{font-size:11px;border-radius:3px;padding:1px 5px}
  .tag.buy{background:#fbe9e7;color:var(--up)}
  .tag.hold{background:#eef1f4;color:#5f6b7a;border:1px solid #dde2e7}
  .tag.wait{background:#f0f1f3;color:var(--dim)}
  .note{margin-top:10px;padding-top:9px;border-top:1px dashed var(--line);
    color:var(--dim);font-size:12px}
  .empty{color:var(--dim);font-size:12.5px;padding:12px 0;margin:0}
  .more{margin-top:8px;font-size:12px;color:var(--accent);cursor:pointer;user-select:none}
  h2{font-size:17px;margin:40px 0 6px}
  .hint{color:var(--dim);font-size:12.5px;margin-bottom:12px}
  .cmp{background:#fff;border:1px solid var(--line);border-radius:10px;padding:6px 4px;overflow-x:auto}
  .cmp table{font-size:12.5px}
  .cmp th{padding:8px 10px}
  .cmp td{padding:8px 10px}
  td.hot{font-weight:700;background:#fdf6f5}
  .foot{margin-top:34px;color:var(--dim);font-size:11.5px;border-top:1px solid var(--line);padding-top:14px}
"""

SCRIPT = """
document.querySelectorAll('.card').forEach(function(c){
  var rows = c.querySelectorAll('tbody tr');
  if (rows.length > 10) {
    var b = document.createElement('div');
    b.className = 'more';
    b.textContent = '展开全部 ' + rows.length + ' 只 ▾';
    b.onclick = function(){
      c.classList.toggle('expanded');
      b.textContent = c.classList.contains('expanded')
        ? '收起 ▴' : '展开全部 ' + rows.length + ' 只 ▾';
    };
    c.appendChild(b);
  }
});
"""

RULES = ("current", "reverse", "lowvol", "volratio", "relstr", "random")
RULE_LABEL = {"current": "当前规则", "reverse": "反向", "lowvol": "低波动优先",
              "volratio": "量比最高", "relstr": "相对强度", "random": "随机"}
VERDICT_COLOR = {"有效": "#c0392b", "弱有效": "#b7791f", "接近无效": "#8a8f98",
                 "方向反了": "#2e7d32", "样本不足": "#8a8f98",
                 "单调递减": "#c0392b", "前 5 名有优势": "#c0392b",
                 "正好相反": "#2e7d32", "无差别": "#8a8f98"}


def _status_html(status: dict) -> str:
    if not status:
        return ""
    on = status["on"]
    cls = "on" if on else "off"
    headline = "允许开新仓" if on else "禁止开新仓"
    detail = (f"基准指数 {status['index_close']} vs MA{status['window']} {status['ma']}"
              f"（{status['gap_pct']:+.2f}%）· 已持续 {status['days']} 个交易日"
              f"（自 {status['since']}）")
    advice = ("市场状态为 ON，下面的信号可以直接执行。" if on else
              "市场状态为 OFF：按这套规则，今天所有信号都<b>不执行</b>，"
              "清单里标「待命」的只作观察记录，等状态回到 ON 才下单。")
    return f"""
  <div class="status {cls}">
    <div class="st-line"><span class="dot">●</span><b>{headline}</b>
      <span class="st-tag">{'ON' if on else 'OFF'}</span></div>
    <div class="st-detail">{detail}</div>
    <div class="st-advice">{advice}</div>
  </div>"""


def _cards(snapshot: dict, regime_on: bool, top1_map: dict) -> str:
    cards = []
    for s in snapshot["strategies"]:
        verdict = top1_map.get(s["key"], {})
        vtext = verdict.get("verdict", "—")
        vnote = verdict.get("verdict_note", "")
        color = VERDICT_COLOR.get(vtext, "#8a8f98")

        rows = []
        for c in s["candidates"]:
            cls = "sel" if c["selected"] else ""
            if c["selected"]:
                badge = ('<span class="tag buy">买入</span>' if regime_on
                         else '<span class="tag hold">待命</span>')
            else:
                badge = '<span class="tag wait">排队</span>'
            chg_cls = "up" if c["day_chg_pct"] >= 0 else "dn"
            score = "—" if c["score"] is None else f"{c['score']:.4f}"
            rows.append(
                f'<tr class="{cls}"><td class="rank">{c["rank"]}</td>'
                f'<td class="code">{c["code"]}</td><td>{c["name"]}</td>'
                f'<td class="num">{c["close"]:.2f}</td>'
                f'<td class="num {chg_cls}">{c["day_chg_pct"]:+.2f}%</td>'
                f'<td class="num dim">{score}</td><td>{badge}</td></tr>')

        table = ('<table><thead><tr><th>#</th><th>代码</th><th>名称</th><th>收盘</th>'
                 '<th>当日</th><th>排序分数</th><th>动作</th></tr></thead><tbody>'
                 + "".join(rows) + "</tbody></table>") if rows else '<p class="empty">今日无信号</p>'

        cards.append(f"""
        <section class="card">
          <header>
            <div class="title"><span class="k">{s['key']}</span>{s['name']}</div>
            <div class="verdict" style="color:{color}">{vtext}</div>
          </header>
          <div class="meta">排序依据：{s['rank_rule']}　·　触发 <b>{s['n_signal']}</b> 只，
            买 <b>{s['n_selected']}</b> 只　·　{s['kind']}</div>
          {table}
          <div class="note">{vnote}</div>
        </section>""")
    return "".join(cards)


def _rules_table(rules: list[dict], top1_map: dict) -> str:
    if not rules:
        return ""
    by_strategy: dict[str, dict] = {}
    for r in rules:
        by_strategy.setdefault(r["strategy"], {})[r["rule"]] = r

    head = "".join(f"<th>{RULE_LABEL[r]}</th>" for r in RULES)
    body = []
    for key in sorted(by_strategy):
        vals = {r: by_strategy[key].get(r, {}).get("excess_pp") for r in RULES}
        best = max((v for v in vals.values() if v is not None), default=None)
        cells = []
        for r in RULES:
            v = vals.get(r)
            if v is None:
                cells.append("<td class='num'>—</td>")
                continue
            hot = " hot" if (best is not None and abs(v - best) < 1e-9) else ""
            clr = "up" if v > 0 else ("dn" if v < 0 else "")
            cells.append(f"<td class='num {clr}{hot}'>{v:+.2f}</td>")
        name = by_strategy[key][RULES[0]]["name"]
        t1 = top1_map.get(key, {})
        body.append(
            f"<tr><td><b>{key}</b> {name}</td>{''.join(cells)}"
            f"<td class='num dim'>{t1.get('top1_winrate_pct', '—')}%</td>"
            f"<td class='num dim'>{t1.get('mean_ic', '—')}</td></tr>")

    return f"""
  <h2>排序规则到底有没有用</h2>
  <div class="hint">
    检验方式：把历史上每个交易日的全部触发信号，按不同规则各挑 5 只，观察其后 N 个交易日的收益，
    再减去「当天所有信号的平均收益」。数值为正说明这种排法确实挑得比平均水平好，单位：百分点（pp）。
    后两列是「只买排序第 1 名」的口径，用于判断排序是否单调，与「买满 5 只」不可直接相加比较。
  </div>
  <div class="cmp"><table>
    <thead><tr><th>策略</th>{head}<th>top1 跑赢比例</th><th>平均 IC</th></tr></thead>
    <tbody>{''.join(body)}</tbody>
  </table></div>"""


def _band_table(bands: list[dict], fwd_days: int, min_candidates: int) -> str:
    if not bands:
        return ""
    rows = []
    for r in bands:
        if not r.get("n_days"):
            rows.append(f"<tr><td><b>{r['key']}</b> {r.get('name', '')}</td>"
                        f"<td colspan='3' class='dim'>样本不足</td>"
                        f"<td class='num dim'>0 天</td><td class='dim'>—</td><td class='dim'>—</td></tr>")
            continue
        cells = []
        for band in ("1-5", "6-10", "11+"):
            v = r.get(band)
            txt = "—" if v is None else f"{v:+.2f}%"
            cls = "num hot" if band == "1-5" and v is not None else "num"
            cells.append(f'<td class="{cls}">{txt}</td>')
        verdict = r.get("verdict", "—")
        m = " dn" if verdict in ("正好相反", "方向反了") else ""
        rows.append(
            f"<tr><td><b>{r['key']}</b> {r['name']}</td>{''.join(cells)}"
            f"<td class='num dim'>{r['n_days']} 天</td>"
            f"<td class='{m.strip()}'>{verdict}</td>"
            f"<td class='dim' style='white-space:normal'>{r.get('verdict_note', '')}</td></tr>")

    top5_beat = [r for r in bands if r.get("top5_beat_6_10_pct") is not None]
    avg_beat = (sum(r["top5_beat_6_10_pct"] for r in top5_beat) / len(top5_beat)) if top5_beat else None
    tail = ""
    if avg_beat is not None:
        tail = (f"前 5 名跑赢第 6-10 名的平均频率只有 {avg_beat:.1f}%——"
                f"均值上的优势来自少数大赢家，不是稳定胜率。")

    return f"""
  <h2>排在第 6 名之后，真的更差吗</h2>
  <div class="hint">
    只统计「当日触发信号 ≥ {min_candidates} 只」的交易日，让各名次层级的样本可比。
    这个限制很关键：不加限制时，靠后的名次只在信号极度密集的日子才有样本，
    而那些日子往往是大跌后的反弹前夜，会把结论带偏。
    数值为后验 {fwd_days} 个交易日的平均收益。
  </div>
  <div class="cmp"><table>
    <thead><tr><th>策略</th><th>第 1-5 名（买入）</th><th>第 6-10 名</th>
    <th>第 11 名之后</th><th>样本</th><th>结论</th><th>说明</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table></div>
  <div class="hint" style="margin-top:12px">
    「排队」不是质量差，而是持仓上限挤出来的候补席——它用于涨停买不进、已持有、资金不足时的顺延。
    {tail}
  </div>"""


def _recent_table(trades: list[dict], status: dict) -> str:
    if not trades:
        return ""
    rows = []
    for t in trades:
        cls = "up" if t["pnl_pct"] >= 0 else "dn"
        rows.append(
            f'<tr><td class="dim">{t["strategy"]}</td>'
            f'<td class="code">{t["symbol"]}</td><td>{t["name"]}</td>'
            f'<td class="dim">{t["entry_date"]}</td><td class="dim">{t["exit_date"]}</td>'
            f'<td class="num">{t["entry_price"]:.2f}</td><td class="num">{t["exit_price"]:.2f}</td>'
            f'<td class="num {cls}">{t["pnl_pct"]:+.2f}%</td>'
            f'<td class="num dim">{t["holding_bars"]}日</td></tr>')

    halt = ""
    if status and not status["on"]:
        halt = (f'<br><b>为什么最近成交停在 {status["since"][:7]}：</b>'
                f'市场状态自 {status["since"]} 起持续 OFF，过滤版此后不再开新仓，'
                f'所以最后一笔交易停在这个时点。不加过滤的原版至今仍在正常交易。')

    return f"""
  <h2>最近的实际成交</h2>
  <div class="hint">这是回测引擎真实下单的记录（过滤+清仓版本），不是信号，
    可以直接对照前面的信号清单验证「信号 → 成交」这一环有没有偏差。{halt}</div>
  <div class="cmp"><table><thead><tr><th>策略</th><th>代码</th><th>名称</th>
  <th>买入日</th><th>卖出日</th><th>买入价</th><th>卖出价</th><th>收益</th><th>持有</th>
  </tr></thead><tbody>{''.join(rows)}</tbody></table></div>"""


def build(cfg: Config, out_dir: Path) -> Path:
    snap_path = out_dir / "signal_snapshot.json"
    if not snap_path.exists():
        raise FileNotFoundError(f"缺少 {snap_path}，请先执行 `python run.py signal`。")
    payload = json.loads(snap_path.read_text(encoding="utf-8"))
    analysis = payload.get("analysis") or {}
    if not analysis.get("rules"):          # 单独跑 board 时，直接读检验结果文件
        cached = out_dir / "rank_analysis.json"
        if cached.exists():
            analysis = json.loads(cached.read_text(encoding="utf-8"))
    status = payload.get("status") or {}
    regime_on = bool(status.get("on", True))
    top1_map = {r["key"]: r for r in (analysis.get("top1") or [])}

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>信号看板 · {payload['as_of']}</title>
<style>{CSS}</style></head>
<body><div class="wrap">
  <h1>信号看板</h1>
  <div class="sub">数据截止 {payload['as_of']} 收盘　·　信号当日收盘确认、次日开盘成交　·　
    单策略最多 {cfg.max_positions} 只、等权 {cfg.weight:.0%}</div>
{_status_html(status)}

  <div class="lead">
    <b>怎么读这张表：</b>「排序分数」是策略内部给所有触发信号打的唯一分数，分高的排在前面。
    前 {cfg.max_positions} 名（浅红底）是该策略当前的优先名单，第 {cfg.max_positions + 1} 名之后是排队候补——
    如果前面有涨停买不进、已经持有的，就按排名往下顺延。
    分数本身不是收益预测，它只在「同一天触发的多个信号里挑谁先买」这一件事上起作用。
  </div>

  <div class="grid">{_cards(payload['snapshot'], regime_on, top1_map)}</div>
{_rules_table(analysis.get('rules') or [], top1_map)}
{_band_table(analysis.get('bands') or [], cfg.fwd_days, cfg.min_candidates)}
{_recent_table(payload.get('recent_trades') or [], status)}

  <div class="foot">
    后验收益用收盘价计算，与实际「次日开盘成交」存在小幅差异，仅用于横向比较不同排序规则。
    样本为 {cfg.universe_index} 当前成分股，存在幸存者偏差。<br>
    ⚠️ 本看板由程序基于公开行情数据自动生成，仅供参考，不构成任何投资建议或个股推荐。
  </div>
</div>
<script>{SCRIPT}</script>
</body></html>"""

    out = out_dir / "signal_board.html"
    out.write_text(html, encoding="utf-8")
    return out
