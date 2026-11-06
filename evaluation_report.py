"""PDF report of an evaluation: everything the explorer shows for the current filters, in one file.

Used by metrics_explorer.py (button "Build PDF report"); needs matplotlib, and pypdf to append the mirror plots
(examples.pdf). Without pypdf the report is complete except for those plots.

Pages
    1  title page: run, inputs, filters, key numbers
    2+ conversion and comparison reports (measured_report.txt, summary.txt) and summary.png
       score distribution and b/y balance; score by length and by charge; length x charge
       terminal residues (N1 / C1)
       retention time (if predicted): predicted vs measured, error by length, RT outliers
       worst and best predicted spectra
       mirror plots (examples.pdf)
"""
import io
import textwrap
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages               # noqa: E402
import numpy as np                                                 # noqa: E402
import pandas as pd                                                # noqa: E402

A4 = (8.27, 11.69)
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
TEXT_W_PT, TEXT_H_PT = 0.86 * A4[0] * 72, (0.93 - 0.06) * A4[1] * 72      # printable text block in points
SHORT = {'n_measured_ions': 'ions', 'top_ion_measured': 'top meas', 'top_ion_predicted': 'top pred',
         'rt_measured_min': 'RT meas', 'rt_predicted_min': 'RT pred', 'rt_error_min': 'RT error', 'bond_cosine': 'bond cos',
         'spectral_angle': 'spec angle', 'source_row': 'row', 'length': 'len'}


# ---------------------------------------------------------------- page helpers
def _page(title=None):
    fig = plt.figure(figsize=A4)
    fig.patch.set_facecolor('white')
    if title:
        fig.text(0.07, 0.955, title, fontsize=14, fontweight='bold', color=INK)
    return fig


def _footer(fig, text):
    fig.text(0.07, 0.025, text, fontsize=7, color=INK2)
    fig.text(0.93, 0.025, time.strftime('%Y-%m-%d %H:%M'), fontsize=7, color=INK2, ha='right')


def _text_pages(pdf, title, text, footer, wrap=118):
    """Monospace text over as many pages as needed. wrap=None keeps lines whole (tables) and shrinks the font so the
    widest line fits; the number of lines per page follows the font size."""
    lines = []
    for raw in text.splitlines():
        lines += (textwrap.wrap(raw, wrap, subsequent_indent='    ', drop_whitespace=False) or ['']) if wrap else [raw]
    widest = max((len(x) for x in lines), default=1)
    size = max(4.5, min(7.6, TEXT_W_PT / (0.61 * widest)))         # monospace glyph ~0.6 em wide
    per_page = int(TEXT_H_PT / (size * 1.22))
    for i in range(0, max(len(lines), 1), per_page):
        fig = _page(title if i == 0 else f'{title} (continued)')
        fig.text(0.07, 0.93, '\n'.join(lines[i:i + per_page]), va='top', family='monospace', fontsize=size, color=INK)
        _footer(fig, footer)
        pdf.savefig(fig)
        plt.close(fig)


def _figure_to_axes(src_fig, ax):
    """Draw an existing matplotlib figure into an axes of the page (as an image, keeps its own layout)."""
    buf = io.BytesIO()
    src_fig.patch.set_alpha(1)
    src_fig.patch.set_facecolor('white')
    src_fig.savefig(buf, format='png', dpi=220, facecolor='white', bbox_inches='tight')
    plt.close(src_fig)
    buf.seek(0)
    ax.imshow(plt.imread(buf))
    ax.axis('off')


def _figures_page(pdf, title, items, footer, notes=None):
    """items: list of (heading, matplotlib figure or None); up to 4 per page, 2 per row."""
    items = [(h, f) for h, f in items if f is not None]
    if not items:
        return
    n = len(items)
    rows = 1 if n == 1 else (n + 1) // 2
    fig = _page(title)
    top, bottom = 0.92, 0.12 if notes else 0.06
    h = min((top - bottom) / rows, 0.42)                           # a single figure does not fill the whole page
    for i, (heading, src) in enumerate(items):
        r, c = divmod(i, 1 if n == 1 else 2)
        w = 0.86 if n == 1 else 0.43
        left = 0.07 + c * 0.45
        y0 = top - (r + 1) * h
        fig.text(left, y0 + h - 0.012, heading, fontsize=10, fontweight='bold', color=INK)
        ax = fig.add_axes([left, y0, w, h - 0.03])
        _figure_to_axes(src, ax)
    if notes:
        fig.text(0.07, min(0.10, top - rows * h - 0.02), '\n'.join(textwrap.fill(n, 120) for n in notes), va='top', fontsize=8, color=INK2)
    _footer(fig, footer)
    pdf.savefig(fig)
    plt.close(fig)


def _table_text(df, cols, n, float_fmt='{:.3f}'):
    t = df[cols].head(n).copy().rename(columns=SHORT)
    for c in t.columns:
        if pd.api.types.is_float_dtype(t[c]):
            t[c] = t[c].map(lambda v: '' if pd.isna(v) else float_fmt.format(v))
    return t.to_string(index=False, max_colwidth=26)


def _plain(md):
    return md.replace('**', '')


# ---------------------------------------------------------------- report
def build_report(d, df, metric, label, bounds, min_n, min_effect, filters_text, run_dir=None, title=None,
                 explorer=None):
    """Returns the PDF as bytes. d = filtered metrics, df = all metrics, explorer = the metrics_explorer module."""
    mx = explorer
    th = mx.THEMES['light']
    run_dir = Path(run_dir) if run_dir else None
    title = title or (run_dir.name if run_dir else 'Evaluation')
    footer = f'pepMS2-Check evaluation report | {title}'
    overall = float(d[metric].median())
    has_charge = 'charge' in d.columns and d['charge'].notna().any()
    has_rt = mx.has_rt(d)
    out = io.BytesIO()
    with PdfPages(out) as pdf:
        # ---- 1 title page
        fig = _page()
        fig.text(0.07, 0.90, 'Evaluation report', fontsize=22, fontweight='bold', color=INK)
        fig.text(0.07, 0.865, 'measured vs predicted MS/MS spectra' + (' and retention times' if has_rt else ''),
                 fontsize=12, color=INK2)
        info = [('Run', title), ('Created', time.strftime('%Y-%m-%d %H:%M'))]
        if run_dir is not None and (run_dir / 'input').is_dir():
            files = sorted(p.name for p in (run_dir / 'input').iterdir())
            info.append(('Input files', ', '.join(files)))
        info += [('Spectra in the evaluation', f'{len(df):,}'),
                 ('Spectra in this report', f'{len(d):,} ({100 * len(d) / max(len(df), 1):.1f}%)'),
                 ('Filters', filters_text or 'none'), ('Score shown', label)]
        y = 0.80
        for k, v in info:
            fig.text(0.07, y, k, fontsize=9.5, color=INK2)
            wrapped = textwrap.wrap(str(v), 78) or ['']
            fig.text(0.34, y, '\n'.join(wrapped), fontsize=9.5, color=INK, va='top')
            y -= 0.028 * max(len(wrapped), 1) + 0.008
        y -= 0.03
        fig.text(0.07, y, 'Key numbers', fontsize=12, fontweight='bold', color=INK)
        y -= 0.04
        kpis = [(f'median {label}', f'{overall:.3f}'), (f'mean {label}', f'{d[metric].mean():.3f}'),
                (f'share with {label} >= 0.8', f'{100 * (d[metric] >= 0.8).mean():.1f}%')]
        if 'bond_cosine' in d and metric != 'bond_cosine':
            kpis.append(('median bond-level cosine', f'{d["bond_cosine"].median():.3f}'))
        if has_rt:
            r = d[np.isfinite(pd.to_numeric(d['rt_abs_error_min'], errors='coerce'))]
            if 'rt_early' in r:
                kpis.append(('early-eluting (not scored)', f'{int((r["rt_early"] == 1).sum()):,} spectra'))
                r = r[r['rt_early'] != 1]
            kpis += [('median RT error', f'{r["rt_abs_error_min"].median():.2f} min'),
                     ('95% of spectra within', f'{r["rt_abs_error_min"].quantile(0.95):.2f} min'),
                     ('RT outliers', f'{int((r["rt_outlier"] == 1).sum()):,}' if 'rt_outlier' in r else '-')]
            if 'rt_calibration' in r and r['rt_calibration'].notna().any():
                kpis.append(('RT calibration', ', '.join(map(str, r['rt_calibration'].dropna().unique()))))
        for k, v in kpis:
            fig.text(0.09, y, k, fontsize=10, color=INK2)
            fig.text(0.42, y, v, fontsize=10, color=INK, fontweight='bold')
            y -= 0.03
        fig.text(0.07, 0.08, textwrap.fill(
            'Scores compare the measured y/b ion intensities with the model prediction (1 = identical). Group plots: '
            'dot = median, bar = 95% confidence interval of the median, dashed line = overall median; groups with '
            f'fewer than {min_n} spectra are left out. "Clearly lower/higher": the whole interval lies on one side '
            f'of the overall median and the difference is at least {min_effect:g}.', 125), fontsize=7.8, color=INK2,
            va='top')
        _footer(fig, footer)
        pdf.savefig(fig)
        plt.close(fig)

        # ---- 2 run reports
        if run_dir is not None:
            parts = []
            for name, path in (('CONVERSION (format_measured.py)', run_dir / 'measured_report.txt'),
                               ('COMPARISON (compare_spectra.py, all spectra, no filters)', run_dir / 'comparison' / 'summary.txt')):
                if path.is_file():
                    parts.append(f'{name}\n{"-" * len(name)}\n' + path.read_text(encoding='utf-8', errors='replace'))
            if parts:
                _text_pages(pdf, 'Run reports', '\n\n'.join(parts), footer)
            png = run_dir / 'comparison' / 'summary.png'
            if png.is_file():
                fig = _page('Overview (all spectra)')
                ax = fig.add_axes([0.05, 0.55, 0.9, 0.36])
                ax.imshow(plt.imread(str(png)))
                ax.axis('off')
                _footer(fig, footer)
                pdf.savefig(fig)
                plt.close(fig)

        # ---- 3 distribution, balance, length, charge
        items = [(f'Distribution of {label}', mx.hist_fig(d[metric], th, label, overall, bounds[0]))]
        if {'b_share_measured', 'b_share_predicted'} <= set(d.columns):
            items.append(('b/y balance (measured vs predicted share of b ions)',
                          mx.balance_fig(d.sample(min(len(d), 5000), random_state=0), th)))
        len_stats = mx.group_stats(d, 'length', metric, overall, min_effect)
        items.append(('By peptide length', mx.dot_fig(len_stats, th, 'peptide length (residues)', label, overall, bounds,
                                                     numeric=True, min_n=min_n)))
        notes = [_plain(mx.findings(len_stats[len_stats['n'] >= min_n], 'Length', fmt=lambda v: str(int(v))))]
        if has_charge:
            ch_stats = mx.group_stats(d, 'charge', metric, overall, min_effect)
            items.append(('By precursor charge', mx.dot_fig(ch_stats, th, 'precursor charge', label, overall, bounds,
                                                           numeric=True, min_n=min_n, fmt=lambda v: f'{int(v)}+')))
            notes.append(_plain(mx.findings(ch_stats[ch_stats['n'] >= min_n], 'Charge', fmt=lambda v: f'{int(v)}+')))
        _figures_page(pdf, f'Prediction quality: {label}', items, footer, notes)
        if has_charge:
            pv = mx.pair_table(d, 'length', 'charge', metric, min_n)
            hf = mx.heat_fig(pv.T if pv is not None else None, th, f'median {label}', 'precursor charge', 'peptide length',
                             row_fmt=lambda v: f'{v:.0f}+', col_fmt=lambda v: str(int(v)))
            _figures_page(pdf, 'Length and charge together', [(f'median {label} per length and charge', hf)], footer)

        # ---- 4 terminal residues
        n_stats = mx.group_stats(d, 'N1', metric, overall, min_effect)
        c_stats = mx.group_stats(d, 'C1', metric, overall, min_effect)
        items = [('By N-terminal residue (N1)', mx.dot_fig(n_stats, th, 'N1 residue', label, overall, bounds, min_n=min_n,
                                                          sort_desc=True)),
                 ('By C-terminal residue (C1)', mx.dot_fig(c_stats, th, 'C1 residue', label, overall, bounds, min_n=min_n,
                                                          sort_desc=True))]
        n_ok, c_ok = n_stats[n_stats['n'] >= min_n], c_stats[c_stats['n'] >= min_n]
        pv = mx.pair_table(d, 'N1', 'C1', metric, min_n, row_order=list(n_ok.sort_values('median', ascending=False)['group']),
                           col_order=list(c_ok.sort_values('median', ascending=False)['group']))
        items.append(('N1 and C1 together', mx.heat_fig(pv, th, f'median {label}', 'N1 residue', 'C1 residue')))
        _figures_page(pdf, 'Terminal residues', items, footer,
                      [_plain(mx.findings(n_ok, 'N1')), _plain(mx.findings(c_ok, 'C1')),
                       'Residue effects can also be length or charge effects in disguise.'])

        # ---- 5 retention time
        if has_rt:
            r_all = d[np.isfinite(pd.to_numeric(d['rt_abs_error_min'], errors='coerce'))]
            early = r_all['rt_early'] == 1 if 'rt_early' in r_all else pd.Series(False, index=r_all.index)
            r = r_all[~early]
            ae = r['rt_abs_error_min']
            ov = float(ae.median())
            st_len = mx.group_stats(r, 'length', 'rt_abs_error_min', ov, min_effect)
            items = [('Predicted against measured RT', mx.rt_fig(r_all.sample(min(len(r_all), 8000), random_state=0), th)),
                     ('Absolute RT error by peptide length',
                      mx.dot_fig(st_len, th, 'peptide length (residues)', 'absolute RT error (min)', ov, (0.0, float('inf')),
                                 numeric=True, min_n=min_n))]
            notes = [f'{len(r):,} spectra with a predicted RT' + (' outside the early-elution zone' if early.any() else '')
                     + f': median error {ov:.2f} min, 95% within {ae.quantile(0.95):.2f} min, '
                     f'mean signed error {r["rt_error_min"].mean():+.2f} min.']
            if early.any():
                notes.append(f'Early-elution zone (grey): {int(early.sum()):,} spectra before '
                             f'{r_all.loc[early, "rt_measured_min"].max():.2f} min elute around the void time (weak '
                             f'retention) and cannot be predicted (median error {r_all.loc[early, "rt_abs_error_min"].median():.2f} '
                             'min); a chromatography effect, not wrong identifications, so not scored, and flagged as '
                             'outliers only when extreme even for that zone.')
            notes += [
                     _plain(mx.findings(st_len[st_len['n'] >= min_n], 'Length', fmt=lambda v: str(int(v)))) +
                     ' (lower is better here)']
            if 'rt_model_seen' in r and r['rt_model_seen'].astype(bool).any():
                seen = r['rt_model_seen'].astype(bool)
                notes.append(f'{int(seen.sum()):,} spectra are peptides that were part of the RT calibration/training '
                             f'(median error {r.loc[seen, "rt_abs_error_min"].median():.2f} min); the others: '
                             f'{r.loc[~seen, "rt_abs_error_min"].median():.2f} min.' if (~seen).any() else
                             'All peptides were part of the RT calibration/training: the RT error is optimistic.')
            _figures_page(pdf, 'Retention time', items, footer, notes)
            if 'rt_outlier' in r and (r['rt_outlier'] == 1).any():
                cols = [c for c in ('Peptide', 'length', 'charge', 'rt_measured_min', 'rt_predicted_min', 'rt_error_min',
                                    metric, 'source_row') if c in r.columns]
                t = r[r['rt_outlier'] == 1].sort_values('rt_abs_error_min', ascending=False)
                _text_pages(pdf, f'RT outliers ({len(t):,})',
                            'Spectra whose RT error is far from the typical error (> 4 robust SD and > 0.5 min): '
                            'possible misidentifications, isomers or modifications.\n\n' + _table_text(t, cols, 200), footer,
                            wrap=None)

        # ---- 6 worst / best
        cols = [c for c in ('Peptide', 'length', 'charge', metric, 'bond_cosine', 'n_measured_ions', 'top_ion_measured',
                            'top_ion_predicted', 'rt_measured_min', 'rt_predicted_min', 'source_row') if c in d.columns]
        cols = list(dict.fromkeys(cols))
        k = 40
        _text_pages(pdf, f'Worst and best predicted spectra ({label})',
                    f'WORST {k}\n\n' + _table_text(d.nsmallest(k, metric), cols, k) +
                    f'\n\n\nBEST {k}\n\n' + _table_text(d.nlargest(k, metric), cols, k), footer, wrap=None)

    pdf_bytes = out.getvalue()

    # ---- 7 mirror plots: append examples.pdf
    ex = run_dir / 'comparison' / 'examples.pdf' if run_dir is not None else None
    if ex is not None and ex.is_file():
        try:
            from pypdf import PdfReader, PdfWriter
            w = PdfWriter()
            for src in (io.BytesIO(pdf_bytes), str(ex)):
                for page in PdfReader(src).pages:
                    w.add_page(page)
            merged = io.BytesIO()
            w.write(merged)
            pdf_bytes = merged.getvalue()
        except ImportError:
            pass                                   # report without the mirror plots (pip install pypdf to add them)
    return pdf_bytes
