#!/usr/bin/env python3
import csv
import json
import math
import re
from pathlib import Path

ROOT = Path('docs/int8-kv-audit')
DATA = ROOT / 'data'
PLOTS = ROOT / 'plots'
DATA.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)


def _svg_line_plot(path, title, x_vals, series, x_label, y_label, y_min=None, y_max=None):
    width, height = 980, 520
    ml, mr, mt, mb = 90, 30, 70, 70
    pw, ph = width - ml - mr, height - mt - mb

    if not x_vals:
        raise ValueError('empty x_vals')

    x_min, x_max = min(x_vals), max(x_vals)
    if x_max == x_min:
        x_max += 1

    ys = []
    for s in series:
        ys.extend([v for v in s['y'] if v is not None])
    if not ys:
        ys = [0.0, 1.0]

    y_min_user = y_min is not None
    y_max_user = y_max is not None
    if y_min is None:
        y_min = min(ys)
    if y_max is None:
        y_max = max(ys)
    if y_max == y_min:
        y_max += 1

    pad = (y_max - y_min) * 0.08
    if not y_min_user:
        y_min -= pad
    if not y_max_user:
        y_max += pad

    def sx(x):
        return ml + (x - x_min) / (x_max - x_min) * pw

    def sy(y):
        return mt + (y_max - y) / (y_max - y_min) * ph

    out = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    out.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    out.append(f'<text x="{width/2}" y="34" text-anchor="middle" font-size="22" font-family="sans-serif">{title}</text>')

    # axes
    out.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#111" stroke-width="1.5"/>')
    out.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#111" stroke-width="1.5"/>')

    # y ticks
    for i in range(6):
        yv = y_min + (y_max - y_min) * i / 5.0
        py = sy(yv)
        out.append(f'<line x1="{ml-6}" y1="{py:.2f}" x2="{ml+pw}" y2="{py:.2f}" stroke="#e8e8e8" stroke-width="1"/>')
        out.append(f'<text x="{ml-10}" y="{py+4:.2f}" text-anchor="end" font-size="12" font-family="monospace" fill="#444">{yv:.3f}</text>')

    # x ticks
    x_ticks = x_vals
    if len(x_ticks) > 12:
        step = math.ceil(len(x_ticks) / 12)
        x_ticks = x_ticks[::step]
        if x_vals[-1] not in x_ticks:
            x_ticks.append(x_vals[-1])
    for xv in x_ticks:
        px = sx(xv)
        out.append(f'<line x1="{px:.2f}" y1="{mt}" x2="{px:.2f}" y2="{mt+ph+6}" stroke="#e8e8e8" stroke-width="1"/>')
        out.append(f'<text x="{px:.2f}" y="{mt+ph+22}" text-anchor="middle" font-size="12" font-family="monospace" fill="#444">{xv}</text>')

    # lines + points
    legend_x = ml + 8
    legend_y = mt - 30
    for idx, s in enumerate(series):
        points = []
        for xv, yv in zip(x_vals, s['y']):
            if yv is None:
                continue
            points.append((sx(xv), sy(yv)))
        if len(points) >= 2:
            pts = ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)
            out.append(f'<polyline points="{pts}" fill="none" stroke="{s["color"]}" stroke-width="2.4"/>')
        for x, y in points:
            out.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.8" fill="{s["color"]}"/>')

        ly = legend_y + idx * 18
        out.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x+22}" y2="{ly}" stroke="{s["color"]}" stroke-width="2.6"/>')
        out.append(f'<text x="{legend_x+28}" y="{ly+4}" font-size="13" font-family="sans-serif" fill="#222">{s["name"]}</text>')

    out.append(f'<text x="{ml+pw/2}" y="{height-18}" text-anchor="middle" font-size="14" font-family="sans-serif">{x_label}</text>')
    out.append(f'<text transform="translate(20 {mt+ph/2}) rotate(-90)" text-anchor="middle" font-size="14" font-family="sans-serif">{y_label}</text>')

    out.append('</svg>')
    path.write_text('\n'.join(out))


def build_layer_scales_plot():
    inp = DATA / 'per_layer_scales_gemma3_1b.csv'
    rows = list(csv.DictReader(inp.open()))
    parsed = []
    for r in rows:
        m = re.search(r'layers\.(\d+)\.', r['layer_name'])
        if not m:
            continue
        parsed.append((int(m.group(1)), float(r['k_scale']), float(r['v_scale']), float(r['k_absmax_seen']), float(r['v_absmax_seen'])))
    parsed.sort(key=lambda t: t[0])

    out_csv = DATA / 'per_layer_scales_gemma3_1b_with_layer_idx.csv'
    with out_csv.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['layer_idx', 'k_scale', 'v_scale', 'k_absmax_seen', 'v_absmax_seen'])
        for row in parsed:
            w.writerow(row)

    x = [r[0] for r in parsed]
    k = [r[1] for r in parsed]
    v = [r[2] for r in parsed]
    _svg_line_plot(
        PLOTS / 'per_layer_scales_gemma3_1b.svg',
        'Per-layer INT8 KV scales (Gemma-3 1B, first measured pass)',
        x,
        [
            {'name': 'k_scale', 'y': k, 'color': '#d1495b'},
            {'name': 'v_scale', 'y': v, 'color': '#00798c'},
        ],
        x_label='Layer index',
        y_label='Scale value (float32)',
        y_min=0.0,
    )


def load_cfg(path):
    d = json.loads(Path(path).read_text())
    cfg = d['results']['4b-w4a16']['configs']
    out = {}
    for v in cfg.values():
        out[int(v['context_len'])] = float(v['max_throughput'])
    return out


def build_throughput_plot():
    tp2_bf16 = load_cfg('results/throughput-grid-20260317-174312.json')
    tp2_int8 = load_cfg('results/throughput-grid-20260317-194424.json')
    dp2_bf16 = load_cfg('results/throughput-grid-20260317-190716.json')
    dp2_int8 = load_cfg('results/throughput-grid-20260317-192257.json')

    contexts = sorted(set(tp2_bf16) | set(tp2_int8) | set(dp2_bf16) | set(dp2_int8))

    out_csv = DATA / 'throughput_4b_configs.csv'
    with out_csv.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['context_len', 'tp2_bf16', 'tp2_int8', 'dp2_bf16', 'dp2_int8'])
        for c in contexts:
            w.writerow([c, tp2_bf16.get(c, ''), tp2_int8.get(c, ''), dp2_bf16.get(c, ''), dp2_int8.get(c, '')])

    _svg_line_plot(
        PLOTS / 'throughput_4b_configs.svg',
        '4B throughput by context and KV dtype (max tok/s)',
        contexts,
        [
            {'name': 'TP=2 BF16', 'y': [tp2_bf16.get(c) for c in contexts], 'color': '#444444'},
            {'name': 'TP=2 INT8', 'y': [tp2_int8.get(c) for c in contexts], 'color': '#c08a00'},
            {'name': 'DP=2 BF16', 'y': [dp2_bf16.get(c) for c in contexts], 'color': '#2e6f40'},
            {'name': 'DP=2 INT8', 'y': [dp2_int8.get(c) for c in contexts], 'color': '#2f59a7'},
        ],
        x_label='Context length (tokens)',
        y_label='Throughput (tok/s)',
    )


def build_27b_scales_plot():
    inp = DATA / 'per_layer_scales_gemma3_27b_tp2.csv'
    if not inp.exists():
        return
    rows = list(csv.DictReader(inp.open()))
    parsed = []
    for r in rows:
        parsed.append((int(r['layer_idx']), float(r['k_scale']), float(r['v_scale'])))
    parsed.sort(key=lambda t: t[0])
    x = [r[0] for r in parsed]
    k = [r[1] for r in parsed]
    v = [r[2] for r in parsed]
    _svg_line_plot(
        PLOTS / 'per_layer_scales_gemma3_27b_tp2.svg',
        'Per-layer INT8 KV scales (Gemma-3 27B TP=2, first measured pass)',
        x,
        [
            {'name': 'k_scale', 'y': k, 'color': '#d1495b'},
            {'name': 'v_scale', 'y': v, 'color': '#00798c'},
        ],
        x_label='Layer index',
        y_label='Scale value (float32)',
        y_min=0.0,
    )


def build_quant_error_table():
    # default scale path discussed in code: absmax=20 -> scale=20/127
    scale = 20.0 / 127.0
    vals = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 20.0]
    out_csv = DATA / 'default_scale_quant_error_examples.csv'
    with out_csv.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['value', 'quantized_int8', 'dequantized', 'abs_error', 'rel_error_pct'])
        for v in vals:
            q = int(round(v / scale))
            q = max(-128, min(127, q))
            dq = q * scale
            err = abs(v - dq)
            rel = 0.0 if v == 0 else err / abs(v) * 100
            w.writerow([v, q, dq, err, rel])


def main():
    build_layer_scales_plot()
    build_27b_scales_plot()
    build_throughput_plot()
    build_quant_error_table()
    print('Generated artifacts in', ROOT)


if __name__ == '__main__':
    main()
