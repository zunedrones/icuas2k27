"""Compare several race runs: `ros2 run icuas27_race race_compare results/*` writes
compare.html/compare.csv next to the runs (organisers run each solution at least 3 times and keep
the run with the lowest T_adj, so consistency matters as much as speed)."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import math
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .metrics import compute_metrics, lap_table  # noqa: E402
from .report import CSS, fmt  # noqa: E402
from .rundata import find_runs, load_run  # noqa: E402

COLUMNS = [('run', 'Run'), ('valid', 'Valid'), ('elapsed_time', 'T [s]'),
           ('penalty_time', 'Penalty [s]'), ('adjusted_time', 'T_adj [s]'),
           ('gates_passed', 'Gates'), ('hits', 'Hits'), ('missed_entries', 'Missed'),
           ('best_lap', 'Best lap [s]'), ('mean_lap', 'Mean lap [s]'), ('lap_std', 'Lap std [s]'),
           ('max_speed', 'Max v [m/s]'), ('mean_speed', 'Mean v [m/s]'),
           ('min_clearance', 'Min clear [m]'), ('mean_gate_offset', 'Mean offset [m]'),
           ('tracking_rmse', 'Track RMSE [m]'), ('reason', 'Reason')]


def b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs', nargs='+', help='run directories, globs or a parent directory')
    ap.add_argument('--out', default='', help='output directory (default: parent of the first run)')
    args = ap.parse_args(argv)
    run_dirs = find_runs(args.runs)
    if not run_dirs:
        print('no runs found', file=sys.stderr)
        return 1
    rows, laps = [], []
    for d in run_dirs:
        try:
            run = load_run(d)
        except (OSError, ValueError, KeyError) as exc:
            print(f'skipping {d}: {exc}', file=sys.stderr)
            continue
        m = compute_metrics(run)
        rows.append(m)
        laps.append([r['lap_time'] for r in lap_table(run)])
    out = args.out or os.path.dirname(os.path.abspath(run_dirs[0]))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'compare.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow([k for k, _ in COLUMNS])
        for r in rows:
            w.writerow([r.get(k) for k, _ in COLUMNS])
    valid = [r for r in rows if r.get('valid')]
    best = min(valid, key=lambda r: r['adjusted_time']) if valid else None
    adj = [r['adjusted_time'] for r in valid]
    stats = {
        'runs': len(rows), 'valid_runs': len(valid),
        'success_rate': len(valid) / len(rows) if rows else 0.0,
        'best_run': best['run'] if best else None,
        'best_adjusted_time': best['adjusted_time'] if best else None,
        'mean_adjusted_time': float(np.mean(adj)) if adj else None,
        'std_adjusted_time': float(np.std(adj)) if adj else None,
    }
    with open(os.path.join(out, 'compare.json'), 'w', encoding='utf-8') as fh:
        json.dump({'stats': stats, 'runs': rows}, fh, indent=2, default=float)

    figs = []
    if rows:
        fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(rows) + 2), 3.8))
        names = [r['run'] for r in rows]
        vals = [r['adjusted_time'] if r.get('valid') else 0.0 for r in rows]
        colors = ['#2ca02c' if r.get('valid') else '#d62728' for r in rows]
        ax.bar(range(len(rows)), vals, color=colors)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('T_adj [s] (red = invalid)')
        ax.grid(alpha=0.3, axis='y')
        ax.set_title('Adjusted race time per run')
        figs.append(b64(fig))
    if any(laps):
        fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(rows) + 2), 3.8))
        data = [l if l else [math.nan] for l in laps]
        ax.boxplot(data, labels=[r['run'] for r in rows])
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.set_ylabel('lap time [s]')
        ax.grid(alpha=0.3, axis='y')
        ax.set_title('Lap time distribution per run')
        figs.append(b64(fig))
    head = ''.join(f'<th>{html.escape(lab)}</th>' for _, lab in COLUMNS)
    body = ''.join('<tr>' + ''.join(
        f'<td>{fmt(r.get(k), 3) if not isinstance(r.get(k), str) else html.escape(r.get(k))}</td>'
        for k, _ in COLUMNS) + '</tr>' for r in rows)
    imgs = ''.join(f'<section><img src="data:image/png;base64,{f}"/></section>' for f in figs)
    stats_html = ''.join(f'<div class="card"><div class="k">{html.escape(k)}</div>'
                         f'<div class="v">{fmt(v, 3) if not isinstance(v, str) else html.escape(v)}</div></div>'
                         for k, v in stats.items())
    page = (f'<!doctype html><html><head><meta charset="utf-8"><title>Race comparison</title>'
            f'<style>{CSS}</style></head><body><header><h1>Race comparison</h1>'
            f'<p>{len(rows)} runs</p></header><main><div class="cards">{stats_html}</div>{imgs}'
            f'<section><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
            f'</section></main></body></html>')
    path = os.path.join(out, 'compare.html')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(page)
    print(json.dumps(stats, indent=2, default=float))
    print(path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
