from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _aggregate_by_engine(derived_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in derived_rows:
        grouped.setdefault(str(row["engine"]), []).append(row)

    result: list[dict[str, Any]] = []
    for engine, rows in grouped.items():
        count = len(rows)
        result.append(
            {
                "engine": engine,
                "avg_score": sum(float(row.get("avg_score", 0.0)) for row in rows) / count if count else 0.0,
                "success_rate": sum(float(row.get("success_rate", 0.0)) for row in rows) / count if count else 0.0,
                "avg_latency_seconds": sum(float(row.get("avg_latency_seconds", 0.0)) for row in rows) / count
                if count
                else 0.0,
                "score_per_second": sum(float(row.get("score_per_second", 0.0)) for row in rows) / count
                if count
                else 0.0,
            }
        )
    return sorted(result, key=lambda row: row["engine"])


def _build_table_html(derived_rows: list[dict[str, Any]]) -> str:
    headers = [
        ("run_id", "Run"),
        ("dataset", "Dataset"),
        ("engine", "Engine"),
        ("avg_score", "Avg Score"),
        ("success_rate", "Success Rate"),
        ("avg_latency_seconds", "Avg Latency (s)"),
        ("avg_iterations", "Avg Iterations"),
        ("score_per_second", "Score/sec"),
        ("failure_rate", "Failure Rate"),
        ("timeout_rate", "Timeout Rate"),
    ]

    head = "".join(f'<th onclick="sortTable(\'{key}\')" data-key="{key}">{label}</th>' for key, label in headers)
    body_rows: list[str] = []
    for row in derived_rows:
        body_rows.append(
            "<tr>"
            f"<td>{row.get('run_id', '')}</td>"
            f"<td>{row.get('dataset', '')}</td>"
            f"<td>{row.get('engine', '')}</td>"
            f"<td>{float(row.get('avg_score', 0.0)):.4f}</td>"
            f"<td>{float(row.get('success_rate', 0.0)):.2%}</td>"
            f"<td>{float(row.get('avg_latency_seconds', 0.0)):.2f}</td>"
            f"<td>{float(row.get('avg_iterations', 0.0)):.2f}</td>"
            f"<td>{float(row.get('score_per_second', 0.0)):.4f}</td>"
            f"<td>{float(row.get('failure_rate', 0.0)):.2%}</td>"
            f"<td>{float(row.get('timeout_rate', 0.0)):.2%}</td>"
            "</tr>"
        )
    return f"""
<table id="metrics-table">
  <thead><tr>{head}</tr></thead>
  <tbody>
    {"".join(body_rows)}
  </tbody>
</table>
"""


def _js_array(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _render_charts(derived_rows: list[dict[str, Any]], niah_context_rows: list[dict[str, Any]]) -> str:
    by_engine = _aggregate_by_engine(derived_rows)
    engines = [row["engine"] for row in by_engine]
    avg_scores = [row["avg_score"] for row in by_engine]
    success_rates = [row["success_rate"] for row in by_engine]
    avg_latency = [row["avg_latency_seconds"] for row in by_engine]

    scatter_x = [float(row.get("avg_latency_seconds", 0.0)) for row in derived_rows]
    scatter_y = [float(row.get("avg_score", 0.0)) for row in derived_rows]
    scatter_text = [f"{row.get('run_id')} | {row.get('dataset')} | {row.get('engine')}" for row in derived_rows]

    niah_by_engine: dict[str, list[dict[str, Any]]] = {}
    for row in niah_context_rows:
        niah_by_engine.setdefault(str(row.get("engine", "unknown")), []).append(row)

    niah_traces = []
    for engine, rows in sorted(niah_by_engine.items()):
        rows_sorted = sorted(rows, key=lambda value: int(value.get("context_length", 0)))
        niah_traces.append(
            {
                "x": [int(value.get("context_length", 0)) for value in rows_sorted],
                "y": [float(value.get("avg_score", 0.0)) for value in rows_sorted],
                "mode": "lines+markers",
                "name": engine,
            }
        )

    return f"""
<script>
  const engines = {_js_array(engines)};
  const avgScores = {_js_array(avg_scores)};
  const successRates = {_js_array(success_rates)};
  const avgLatency = {_js_array(avg_latency)};

  Plotly.newPlot('chart-score', [{{ type: 'bar', x: engines, y: avgScores, name: 'Avg Score' }}], {{
    title: 'Average Score by Engine'
  }});

  Plotly.newPlot('chart-success', [{{ type: 'bar', x: engines, y: successRates, name: 'Success Rate' }}], {{
    title: 'Success Rate by Engine',
    yaxis: {{ tickformat: '.0%' }}
  }});

  Plotly.newPlot('chart-latency', [{{ type: 'bar', x: engines, y: avgLatency, name: 'Latency (s)' }}], {{
    title: 'Average Latency by Engine'
  }});

  Plotly.newPlot('chart-frontier', [{{
    type: 'scatter',
    mode: 'markers',
    x: {_js_array(scatter_x)},
    y: {_js_array(scatter_y)},
    text: {_js_array(scatter_text)},
    hovertemplate: '%{{text}}<br>latency=%{{x:.2f}}<br>score=%{{y:.3f}}<extra></extra>'
  }}], {{
    title: 'Score vs Latency (Efficiency Frontier)',
    xaxis: {{ title: 'Average Latency (s)' }},
    yaxis: {{ title: 'Average Score' }}
  }});

  const niahTraces = {_js_array(niah_traces)};
  if (niahTraces.length > 0) {{
    Plotly.newPlot('chart-niah', niahTraces, {{
      title: 'S-NIAH Score Degradation by Context Length',
      xaxis: {{ title: 'Context Length (tokens)' }},
      yaxis: {{ title: 'Average Score' }}
    }});
  }} else {{
    document.getElementById('chart-niah').innerHTML = '<p>No S-NIAH context-length summary was found in selected runs.</p>';
  }}
</script>
"""


def render_html_report(analysis: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    insights = analysis.get("insights", [])
    derived_rows = analysis.get("derived_rows", [])
    niah_context_rows = analysis.get("niah_context_rows", [])

    top = max(derived_rows, key=lambda row: float(row.get("avg_score", 0.0)), default=None)
    avg_success = (
        sum(float(row.get("success_rate", 0.0)) for row in derived_rows) / len(derived_rows) if derived_rows else 0.0
    )
    total_rows = len(derived_rows)

    insight_html = "".join(f"<li>{text}</li>" for text in insights)
    table_html = _build_table_html(derived_rows)
    charts_js = _render_charts(derived_rows, niah_context_rows)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Benchmark Report</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; color: #1f2937; }}
    .cards {{ display: grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap: 12px; margin-bottom: 20px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; background: #fafafa; }}
    .chart {{ height: 360px; margin-bottom: 16px; border: 1px solid #e5e7eb; border-radius: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px; text-align: left; font-size: 14px; }}
    th {{ cursor: pointer; background: #f9fafb; }}
  </style>
</head>
<body>
  <h1>Benchmark Analytics Report</h1>
  <p>Generated: {generated_at}</p>
  <div class="cards">
    <div class="card"><strong>Rows analyzed</strong><div>{total_rows}</div></div>
    <div class="card"><strong>Average success rate</strong><div>{avg_success:.2%}</div></div>
    <div class="card"><strong>Top score</strong><div>{(top["engine"] + " @ " + top["dataset"]) if top else "n/a"}</div></div>
  </div>

  <h2>Key Insights</h2>
  <ul>{insight_html}</ul>

  <h2>Engine Comparison</h2>
  {table_html}

  <h2>Charts</h2>
  <div id="chart-score" class="chart"></div>
  <div id="chart-success" class="chart"></div>
  <div id="chart-latency" class="chart"></div>
  <div id="chart-frontier" class="chart"></div>
  <div id="chart-niah" class="chart"></div>

  <script>
    function sortTable(key) {{
      const table = document.getElementById('metrics-table');
      const headers = Array.from(table.querySelectorAll('thead th'));
      const idx = headers.findIndex(h => h.dataset.key === key);
      if (idx < 0) return;
      const tbody = table.querySelector('tbody');
      const rows = Array.from(tbody.querySelectorAll('tr'));
      const asc = !(table.dataset.sortKey === key && table.dataset.sortDir === 'asc');
      rows.sort((a, b) => {{
        const va = a.children[idx].innerText;
        const vb = b.children[idx].innerText;
        const na = parseFloat(va.replace('%','')) ;
        const nb = parseFloat(vb.replace('%','')) ;
        if (!Number.isNaN(na) && !Number.isNaN(nb)) return asc ? na - nb : nb - na;
        return asc ? va.localeCompare(vb) : vb.localeCompare(va);
      }});
      rows.forEach(r => tbody.appendChild(r));
      table.dataset.sortKey = key;
      table.dataset.sortDir = asc ? 'asc' : 'desc';
    }}
  </script>
  {charts_js}
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path
