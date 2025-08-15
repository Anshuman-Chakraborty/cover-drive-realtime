import os
from jinja2 import Template

TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Cover Drive Analysis Report</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; padding: 24px; }
    h1 { margin-top: 0; }
    .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
    .card { border: 1px solid #e5e5e5; border-radius: 12px; padding: 16px; }
    .score { font-size: 28px; font-weight: 700; }
    .badge { padding: 4px 8px; border-radius: 999px; background: #efefef; display: inline-block; font-size: 12px; }
    table { width: 100%; border-collapse: collapse; margin-top: 8px; }
    th, td { text-align: left; border-bottom: 1px solid #eee; padding: 8px; }
    code { background: #f7f7f7; padding: 2px 6px; border-radius: 6px; }
  </style>
</head>
<body>
  <h1>Cover Drive Analysis Report</h1>
  <p><span class="badge">{{grade}}</span> Overall Score: <span class="score">{{overall|round(1)}}</span> / 10</p>

  <div class="grid">
  {% for k,v in scores.items() %}
    <div class="card">
      <h3>{{k}}</h3>
      <div class="score">{{v|round(1)}}</div>
      <div>{{feedback[k]}}</div>
    </div>
  {% endfor %}
  </div>

  <h2>Metadata</h2>
  <table>
    <tr><th>Video</th><td>{{meta.video_path}}</td></tr>
    <tr><th>Frames</th><td>{{meta.frames}}</td></tr>
    <tr><th>FPS</th><td>{{meta.fps|round(2)}}</td></tr>
  </table>

  {% if plots %}
  <h2>Metric Plots</h2>
  <p>See generated plot images inside <code>./output</code> if enabled.</p>
  {% endif %}
</body>
</html>
"""
def write_report(output_dir, evaluation, meta, include_plots=True):
    os.makedirs(output_dir, exist_ok=True)
    t = Template(TEMPLATE)
    html = t.render(scores=evaluation["scores"], overall=evaluation["overall"], grade=evaluation["grade"], feedback=evaluation["feedback"], meta=meta, plots=include_plots)
    path = os.path.join(output_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
