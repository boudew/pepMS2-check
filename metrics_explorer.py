"""Prediction-quality explorer: which peptides does the model predict well, which badly?

Reads metrics.csv (written by compare_spectra.py). Filter by peptide length, precursor charge, number of measured
ions and terminal residues, and compare groups: dots are the median score, bars the 95% confidence interval of
that median, and the lower panel shows how many spectra each group has.

Start:  streamlit run metrics_explorer.py
Needs:  pip install streamlit pandas numpy matplotlib
"""
import io
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.colors import LinearSegmentedColormap              # noqa: E402
import numpy as np                                                 # noqa: E402
import pandas as pd                                                # noqa: E402
import streamlit as st                                             # noqa: E402

METRICS = {'cosine': 'cosine similarity', 'bond_cosine': 'bond-level cosine (b + complementary y)',
           'spectral_angle': 'spectral angle similarity', 'pearson': 'Pearson r'}
THEMES = {  # colours for the figures (transparent background, so they follow the Streamlit theme)
    'light': dict(ink='#0b0b0b', ink2='#52514e', grid='#e4e3df', series='#2a78d6', bar='#c3d3ea'),
    'dark': dict(ink='#ffffff', ink2='#c3c2b7', grid='#3a3a38', series='#3987e5', bar='#34445e')}
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10})
STYLE = '''<style>
.block-container {padding-top: 2.4rem; max-width: 1400px;}
h1 {font-size: 1.85rem !important; font-weight: 650 !important; letter-spacing: -0.01em;}
h5 {font-weight: 620 !important; margin: 0.9rem 0 0.2rem 0 !important;}
[data-testid="stMetricValue"] {font-size: 1.75rem; font-weight: 620;}
[data-testid="stMetricLabel"] p {opacity: 0.7; font-size: 0.85rem;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 12px;}
</style>'''
SEQ_CMAP = LinearSegmentedColormap.from_list('seq', ['#dbe7f8', '#2a78d6', '#0f3a7a'])   # one hue, light -> dark
GROUP_COLUMNS = {'group': 'group', 'n': 'spectra', 'median': 'median', 'ci_low': '95% CI low', 'ci_high': '95% CI high',
                 'q25': '25%', 'q75': '75%', 'mean': 'mean', 'vs_overall': 'vs overall median', 'differs': 'differs'}


# ---------------------------------------------------------------- style
def apply_style():
    st.markdown(STYLE, unsafe_allow_html=True)


def heading(text):
    st.markdown(f'##### {text}')


# ---------------------------------------------------------------- data
def prepare(df):
    missing = {'Peptide', 'length', 'cosine'} - set(df.columns)
    if missing:
        raise ValueError(f'This does not look like a metrics.csv from compare_spectra.py (missing columns: {sorted(missing)}).')
    df = df.dropna(subset=['Peptide']).copy()
    pep = df['Peptide'].astype(str)
    for d in (1, 2, 3):                       # residue d from the N-terminus (N1 = first) and from the C-terminus
        df[f'N{d}'] = pep.str[d - 1]
        df[f'C{d}'] = pep.str[-d]
    if 'n_measured_ions' not in df.columns:
        df['n_measured_ions'] = 0
    return df


@st.cache_resource(show_spinner='Reading metrics...')
def load_path(path, mtime):
    return prepare(pd.read_csv(path))


@st.cache_resource(show_spinner='Reading metrics...')
def load_bytes(raw):
    return prepare(pd.read_csv(io.BytesIO(raw)))


def apply_filters(df, f):
    m = df['length'].between(f['length'][0], f['length'][1])
    if f.get('charge') and 'charge' in df:
        m &= df['charge'].isin(f['charge'])
    if f.get('min_ions', 0) > 1:
        m &= df['n_measured_ions'] >= f['min_ions']
    if f.get('min_frac', 0) > 0 and 'ion_fraction' in df:
        m &= df['ion_fraction'] >= f['min_frac']
    if f.get('min_kept', 0) > 0 and 'intensity_kept_fraction' in df:
        m &= df['intensity_kept_fraction'] >= f['min_kept']
    for key, col in (('n1', 'N1'), ('c1', 'C1')):
        if f.get(key):
            m &= df[col].isin(f[key])
    if f.get('source') and 'source_file' in df:
        m &= df['source_file'].isin(f['source'])
    return df[m]


# ---------------------------------------------------------------- statistics
def median_ci(x):
    """Distribution-free 95% confidence interval of the median from the sorted sample x."""
    n = len(x)
    if n < 6:
        return np.nan, np.nan
    half = 1.96 * math.sqrt(n) / 2
    return x[max(int(math.floor(n / 2 - half)), 0)], x[min(int(math.ceil(n / 2 + half)), n - 1)]


def group_stats(df, key, value, overall, min_effect=0.01):
    """One row per group: n, median with 95% CI, quartiles, mean, difference to the overall median.
    'differs' is 'lower' / 'higher' when the whole CI of the group median lies below / above the overall median
    AND the difference is at least min_effect (with many spectra even tiny differences are statistically clear)."""
    rows = []
    for k, g in df.groupby(key, sort=True):
        x = np.sort(g[value].dropna().to_numpy(float))
        if len(x) == 0:
            continue
        lo, hi = median_ci(x)
        rows.append(dict(group=k, n=len(x), median=float(np.median(x)), ci_low=lo, ci_high=hi,
                         q25=float(np.percentile(x, 25)), q75=float(np.percentile(x, 75)), mean=float(x.mean())))
    out = pd.DataFrame(rows, columns=['group', 'n', 'median', 'ci_low', 'ci_high', 'q25', 'q75', 'mean'])
    out['vs_overall'] = out['median'] - overall
    lower = (out['ci_high'] < overall) & (out['vs_overall'] <= -min_effect)
    higher = (out['ci_low'] > overall) & (out['vs_overall'] >= min_effect)
    out['differs'] = np.where(lower, 'lower', np.where(higher, 'higher', ''))
    return out


def adjusted(df, metric):
    """Score minus the median of its (length, charge) cell, plus the overall median: shows what a residue does
    beyond what length and charge already explain."""
    keys = ['length'] + (['charge'] if 'charge' in df.columns and df['charge'].notna().any() else [])
    base = df.groupby(keys, dropna=False)[metric].transform('median')
    return df[metric] - base + df[metric].median()


def pair_table(df, row_key, col_key, value, min_n, row_order=None, col_order=None):
    g = df.groupby([row_key, col_key])[value].agg(['median', 'count']).reset_index()
    g = g[g['count'] >= min_n]
    if g.empty:
        return None
    pv = g.pivot(index=row_key, columns=col_key, values='median')
    if row_order is not None:
        pv = pv.reindex(index=[r for r in row_order if r in pv.index])
    if col_order is not None:
        pv = pv.reindex(columns=[c for c in col_order if c in pv.columns])
    return pv


def findings(stats, label, fmt=str, top=4):
    """Plain-language line: which groups are clearly lower / higher than the overall median."""
    low = stats[stats['differs'] == 'lower'].sort_values('vs_overall')
    high = stats[stats['differs'] == 'higher'].sort_values('vs_overall', ascending=False)

    def listing(rows):
        return ', '.join(f'**{fmt(r.group)}** ({r.vs_overall:+.3f}, n={r.n:,})' for r in rows.head(top).itertuples())
    parts = []
    if len(low):
        parts.append('clearly lower: ' + listing(low))
    if len(high):
        parts.append('clearly higher: ' + listing(high))
    return f'{label}: ' + ('; '.join(parts) if parts else 'no group differs clearly from the overall median.')


# ---------------------------------------------------------------- figures
def _style(ax, th):
    ax.set_facecolor('none')
    ax.grid(axis='y', color=th['grid'], lw=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(th['grid'])
    ax.tick_params(colors=th['ink2'])


def hist_fig(values, th, xlabel, overall, lower):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    fig.patch.set_alpha(0)
    _style(ax, th)
    ax.hist(values, bins=np.linspace(lower, 1, 41), color=th['series'], alpha=0.9)
    top = ax.get_ylim()[1] * 1.15
    ax.set_ylim(0, top)
    ax.axvline(overall, color=th['ink2'], lw=1, ls='--')
    ax.text(overall, top * 0.99, f' median {overall:.3f}', color=th['ink2'], fontsize=9, va='top')
    ax.set_xlabel(xlabel, color=th['ink2'])
    ax.set_ylabel('spectra', color=th['ink2'])
    fig.tight_layout()
    return fig


def balance_fig(d, th):
    """Share of intensity in b ions: measured (x) against predicted (y). On the diagonal = right b/y balance."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    fig.patch.set_alpha(0)
    _style(ax, th)
    ax.plot([0, 1], [0, 1], color=th['ink2'], lw=1, ls='--')
    ax.scatter(d['b_share_measured'], d['b_share_predicted'], s=14, color=th['series'], alpha=0.35, linewidths=0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('measured share of intensity in b ions', color=th['ink2'])
    ax.set_ylabel('predicted', color=th['ink2'])
    fig.tight_layout()
    return fig


def dot_fig(stats, th, xlabel, ylabel, overall, bounds, numeric=False, min_n=1, sort_desc=False, fmt=None):
    """Median (dot) with 95% CI (bar) per group, spectra per group below. None if no group is large enough."""
    s = stats[stats['n'] >= min_n]
    if sort_desc:
        s = s.sort_values('median', ascending=False)
    if s.empty:
        return None
    fmt = fmt or (lambda v: str(int(v)) if numeric else str(v))
    x = s['group'].to_numpy(float) if numeric else np.arange(len(s), dtype=float)
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(6.8, 4.4), sharex=True, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.12))
    fig.patch.set_alpha(0)
    _style(ax, th)
    _style(axn, th)
    lo = np.nanmin(np.r_[np.where(np.isnan(s['ci_low']), s['median'], s['ci_low']), overall])
    hi = np.nanmax(np.r_[np.where(np.isnan(s['ci_high']), s['median'], s['ci_high']), overall])
    pad = max((hi - lo) * 0.15, 0.02)
    ax.set_ylim(max(lo - pad, bounds[0]), min(hi + pad, bounds[1]))
    ax.axhline(overall, color=th['ink2'], lw=1, ls='--')
    ax.text(1.012, overall, f'overall\nmedian\n{overall:.3f}', transform=ax.get_yaxis_transform(), ha='left',
            va='center', fontsize=9, color=th['ink2'], linespacing=1.1)          # outside the plot: no collisions
    ax.vlines(x, s['ci_low'], s['ci_high'], color=th['series'], lw=1.4)
    ax.plot(x, s['median'], 'o', color=th['series'], ms=6.5, mew=0)
    ax.set_ylabel(ylabel, color=th['ink2'])
    ax.tick_params(labelbottom=False)
    use_log = s['n'].max() / max(s['n'].min(), 1) > 30
    axn.bar(x, s['n'], width=0.7 if numeric else 0.6, color=th['bar'], log=use_log)
    axn.set_ylabel('spectra', color=th['ink2'])
    axn.set_xlabel(xlabel, color=th['ink2'])
    if numeric:
        ticks = list(range(int(x.min()), int(x.max()) + 1))
        step = 1 if len(ticks) <= 18 else 2
        axn.set_xticks(ticks)
        axn.set_xticklabels([fmt(t) if (t - ticks[0]) % step == 0 else '' for t in ticks])
        axn.set_xlim(x.min() - 0.8, x.max() + 0.8)
    else:
        axn.set_xticks(x)
        axn.set_xticklabels([fmt(g) for g in s['group']])
        axn.set_xlim(-0.8, len(s) - 0.2)
    return fig


def heat_fig(pv, th, cbar_label, row_label, col_label, row_fmt=str, col_fmt=str):
    if pv is None or pv.empty or not np.isfinite(pv.to_numpy(float)).any():
        return None
    arr = pv.to_numpy(float)
    nr, nc = arr.shape
    fig, ax = plt.subplots(figsize=(min(max(4.8, 0.62 * nc + 2.4), 10), min(max(3.0, 0.4 * nr + 1.5), 9)))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    vmin, vmax = np.nanmin(arr), np.nanmax(arr)
    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-3
    im = ax.imshow(np.ma.masked_invalid(arr), cmap=SEQ_CMAP, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks(range(nc))
    ax.set_xticklabels([col_fmt(c) for c in pv.columns], color=th['ink2'])
    ax.set_yticks(range(nr))
    ax.set_yticklabels([row_fmt(r) for r in pv.index], color=th['ink2'])
    for i in range(nr):
        for j in range(nc):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8,
                        color='#ffffff' if (v - vmin) / (vmax - vmin) > 0.55 else '#0b0b0b')
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label(cbar_label, color=th['ink2'])
    cb.ax.tick_params(colors=th['ink2'])
    cb.outline.set_visible(False)
    ax.set_xlabel(col_label, color=th['ink2'])
    ax.set_ylabel(row_label, color=th['ink2'])
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.tight_layout()
    return fig


def show_fig(fig, empty_message):
    if fig is None:
        st.info(empty_message)
        return
    buf = io.BytesIO()                                             # own PNG: transparent, same on every Streamlit version
    fig.savefig(buf, format='png', dpi=200, transparent=True, bbox_inches='tight')
    plt.close(fig)
    st.image(buf.getvalue())


def current_theme():
    try:
        return 'dark' if st.context.theme.type == 'dark' else 'light'
    except Exception:
        return 'light'


def display_table(stats, label):
    t = stats.rename(columns=GROUP_COLUMNS).rename(columns={'group': label})
    return t.round(3)


# ---------------------------------------------------------------- app
def main(source_path=None):
    """source_path: show this metrics.csv directly (used by evaluate_app.py, which also sets the page config and title)."""
    if source_path is None:
        st.set_page_config(page_title='Prediction quality explorer', layout='wide')
        st.title('Prediction quality explorer')
        st.caption('Which peptides does the model predict well, and which not? Reads the metrics.csv written by '
                   'compare_spectra.py.')
    th = THEMES[current_theme()]
    apply_style()

    # ---------------- data
    with st.sidebar:
        st.header('Data')
        if source_path is None:
            path = st.text_input('metrics.csv', os.path.join('comparison', 'metrics.csv'), key='path')
            up = st.file_uploader('or upload it', type=['csv'], key='upload')
        else:
            path, up = str(source_path), None
            st.caption(f'Showing: {source_path}')
    try:
        if up is not None:
            df = load_bytes(up.getvalue())
        elif os.path.isfile(path):
            df = load_path(path, os.path.getmtime(path))
        else:
            st.info(f'File not found: {path}\n\nRun compare_spectra.py first, then give the path to its metrics.csv '
                    f'in the sidebar (or upload it).')
            st.stop()
    except ValueError as e:
        st.error(str(e))
        st.stop()
    has_charge = 'charge' in df.columns and df['charge'].notna().any()

    # ---------------- sidebar filters
    with st.sidebar:
        metric = st.selectbox('Score', [m for m in METRICS if m in df.columns], format_func=METRICS.get, key='metric')
        st.header('Filters')
        lo, hi = int(df['length'].min()), int(df['length'].max())
        length = st.slider('Peptide length (residues)', lo, hi, (lo, hi), key='length') if lo < hi else (lo, hi)
        charge = []
        if has_charge:
            options = sorted(df['charge'].dropna().unique())
            charge = st.multiselect('Precursor charge (empty = all)', options, format_func=lambda z: f'{z:.0f}+', key='charge')
        max_ions = int(max(df['n_measured_ions'].max(), 1))
        min_ions = st.slider('Min. measured y/b ions', 1, max(max_ions, 2), 1, key='min_ions')
        min_frac = 0.0
        if 'ion_fraction' in df.columns:
            min_frac = st.slider('Min. share of possible ions observed', 0.0, 1.0, 0.0, 0.05, key='min_frac')
        min_kept = 0.0
        if 'intensity_kept_fraction' in df.columns:
            min_kept = st.slider('Min. share of listed intensity that is plain y/b', 0.0, 1.0, 0.0, 0.05, key='min_kept')
        n1 = st.multiselect('N-terminal residue (empty = all)', sorted(df['N1'].dropna().unique()), key='n1f')
        c1 = st.multiselect('C-terminal residue (empty = all)', sorted(df['C1'].dropna().unique()), key='c1f')
        source = []
        if 'source_file' in df.columns and df['source_file'].nunique() > 1:
            source = st.multiselect('Source file (empty = all)', sorted(df['source_file'].dropna().unique()), key='src')
        st.header('Analysis')
        min_n = int(st.number_input('Hide groups with fewer spectra than', 1, 100000, 30, key='min_n'))
        min_effect = float(st.number_input('Smallest difference that counts', 0.0, 0.5, 0.01, 0.005, format='%.3f',
                                           key='min_effect',
                                           help="A group is called 'clearly lower/higher' only if its whole confidence "
                                                "interval is on one side of the overall median AND the difference is at "
                                                "least this large."))
        adjust = st.checkbox('Adjust residue comparisons for length and charge', value=False, key='adjust',
                             help='Subtracts the typical score of peptides with the same length and charge, so a '
                                  'residue is only credited with what length and charge do not already explain.')

    d = apply_filters(df, dict(length=length, charge=charge, min_ions=min_ions, min_frac=min_frac, min_kept=min_kept, n1=n1, c1=c1,
                               source=source))
    n_invalid = int(d[metric].isna().sum())
    d = d[d[metric].notna()]
    if d.empty:
        st.warning('No spectra match the current filters.')
        st.stop()

    label = METRICS[metric]
    bounds = (-1.0, 1.02) if metric == 'pearson' else (0.0, 1.02)
    overall = float(d[metric].median())
    kpis = [('Spectra after filtering', f'{len(d):,}', f'{100 * len(d) / len(df):.1f}% of all spectra'),
            (f'Median {label}', f'{overall:.3f}', 'half of the spectra score higher'),
            (f'Mean {label}', f'{d[metric].mean():.3f}', 'average over all spectra'),
            (f'Share with {label} >= 0.8', f'{100 * (d[metric] >= 0.8).mean():.1f}%', f'{int((d[metric] >= 0.8).sum()):,} spectra')]
    for col, (k_label, k_value, k_note) in zip(st.columns(4), kpis):
        with col.container(border=True):
            st.metric(k_label, k_value)
            st.caption(k_note or '\u00a0')
    if n_invalid:
        st.caption(f'{n_invalid:,} filtered spectra have no valid {label} (flat spectrum) and are left out.')

    tab_over, tab_term, tab_rank, tab_data = st.tabs(['Overview', 'Terminal residues', 'Worst and best', 'Table'])

    # ---------------- overview
    with tab_over:
        left, right = st.columns(2)
        with left:
            heading(f'Distribution of {label}')
            show_fig(hist_fig(d[metric], th, label, overall, bounds[0]), 'No data.')
        with right:
            if {'b_share_measured', 'b_share_predicted'} <= set(d.columns):
                heading('b/y balance')
                show_fig(balance_fig(d.sample(min(len(d), 5000), random_state=0), th), 'No data.')
                dd = d['b_share_predicted'] - d['b_share_measured']
                st.write(f'Median measured b-share {d["b_share_measured"].median():.2f}, predicted '
                         f'{d["b_share_predicted"].median():.2f}. Predicted minus measured: median {dd.median():+.2f}.')
                with st.expander('How to read this'):
                    st.write('Points above the dashed line: the model predicts more b-ion intensity than was measured; '
                             'below it, more y. A large shift means the model puts the charge on the wrong half of '
                             'the peptide. Compare "cosine" with "bond-level cosine" (Score, sidebar): if the bond-level '
                             'score is much higher, the model knows where the peptide breaks, but not which side keeps the charge.')
        heading('Length and charge together')
        if has_charge:
            pv = pair_table(d, 'length', 'charge', metric, min_n)
            show_fig(heat_fig(pv.T if pv is not None else None, th, f'median {label}', 'precursor charge', 'peptide length',
                              row_fmt=lambda v: f'{v:.0f}+', col_fmt=lambda v: str(int(v))),
                     f'No length/charge combination has {min_n} or more spectra.')
        else:
            st.info('No precursor charge in this file (run format_measured.py and compare_spectra.py with the '
                    'metadata file next to measured.csv).')
        left, right = st.columns(2)
        len_stats = group_stats(d, 'length', metric, overall, min_effect)
        with left:
            heading('By peptide length')
            show_fig(dot_fig(len_stats, th, 'peptide length (residues)', label, overall, bounds, numeric=True, min_n=min_n),
                     f'No length has {min_n} or more spectra.')
            st.markdown(findings(len_stats[len_stats['n'] >= min_n], 'Length', fmt=lambda v: str(int(v))))
        with right:
            heading('By precursor charge')
            if has_charge:
                ch_stats = group_stats(d, 'charge', metric, overall, min_effect)
                show_fig(dot_fig(ch_stats, th, 'precursor charge', label, overall, bounds, numeric=True, min_n=min_n,
                                 fmt=lambda v: f'{int(v)}+'), f'No charge state has {min_n} or more spectra.')
                st.markdown(findings(ch_stats[ch_stats['n'] >= min_n], 'Charge', fmt=lambda v: f'{int(v)}+'))
        st.caption('Dots: median score of the group. Bars: 95% confidence interval of that median. Dashed line: median '
                   'of all filtered spectra. Lower panel: spectra per group (log scale when sizes differ a lot). '
                   "'Clearly lower/higher' = the whole interval lies on one side of the overall median and the "
                   'difference is at least the minimum set in the sidebar.')
        with st.expander('Tables'):
            st.write('By length')
            st.dataframe(display_table(len_stats, 'length'), hide_index=True)
            st.download_button('Download length table', len_stats.to_csv(index=False).encode('utf-8'),
                               file_name='by_length.csv', mime='text/csv', key='dl_len')
            if has_charge:
                st.write('By charge')
                st.dataframe(display_table(ch_stats, 'charge'), hide_index=True)
                st.download_button('Download charge table', ch_stats.to_csv(index=False).encode('utf-8'),
                                   file_name='by_charge.csv', mime='text/csv', key='dl_ch')

    # ---------------- terminal residues
    with tab_term:
        pc1, pc2 = st.columns(2)
        pos_n = pc1.radio('N-terminal position', [1, 2, 3], horizontal=True, key='posn',
                          format_func=lambda i: f'N{i} (residue {i} from the N-terminus)')
        pos_c = pc2.radio('C-terminal position', [1, 2, 3], horizontal=True, key='posc',
                          format_func=lambda i: f'C{i} (residue {i} from the C-terminus)')
        dv = d.assign(_v=adjusted(d, metric) if adjust else d[metric])
        ov = float(dv['_v'].median())
        vlabel = label + (' (adjusted for length and charge)' if adjust else '')
        n_stats = group_stats(dv, f'N{pos_n}', '_v', ov, min_effect)
        c_stats = group_stats(dv, f'C{pos_c}', '_v', ov, min_effect)
        left, right = st.columns(2)
        with left:
            heading(f'By N{pos_n} residue')
            show_fig(dot_fig(n_stats, th, f'N{pos_n} residue', label, ov, bounds, min_n=min_n, sort_desc=True),
                     f'No residue has {min_n} or more spectra.')
        with right:
            heading(f'By C{pos_c} residue')
            show_fig(dot_fig(c_stats, th, f'C{pos_c} residue', label, ov, bounds, min_n=min_n, sort_desc=True),
                     f'No residue has {min_n} or more spectra.')
        st.markdown(findings(n_stats[n_stats['n'] >= min_n], f'N{pos_n}'))
        st.markdown(findings(c_stats[c_stats['n'] >= min_n], f'C{pos_c}'))
        heading(f'N{pos_n} and C{pos_c} together')
        n_ok, c_ok = n_stats[n_stats['n'] >= min_n], c_stats[c_stats['n'] >= min_n]
        pv = pair_table(dv, f'N{pos_n}', f'C{pos_c}', '_v', min_n,
                        row_order=list(n_ok.sort_values('median', ascending=False)['group']),
                        col_order=list(c_ok.sort_values('median', ascending=False)['group']))
        show_fig(heat_fig(pv, th, f'median {vlabel}', f'N{pos_n} residue', f'C{pos_c} residue'),
                 f'No residue pair has {min_n} or more spectra.')
        st.caption('Tryptic peptides almost always end in K or R, so the C-terminal comparison mostly separates those '
                   'two. Residue effects can also be an effect of peptide length or charge in disguise (for example, '
                   'residues that start long peptides): fix length and charge in the sidebar, or tick the adjustment '
                   'option, before reading much into a difference.')
        with st.expander('Tables'):
            st.write(f'N{pos_n} residue')
            st.dataframe(display_table(n_stats, f'N{pos_n}'), hide_index=True)
            st.download_button('Download N table', n_stats.to_csv(index=False).encode('utf-8'),
                               file_name=f'by_N{pos_n}.csv', mime='text/csv', key='dl_n')
            st.write(f'C{pos_c} residue')
            st.dataframe(display_table(c_stats, f'C{pos_c}'), hide_index=True)
            st.download_button('Download C table', c_stats.to_csv(index=False).encode('utf-8'),
                               file_name=f'by_C{pos_c}.csv', mime='text/csv', key='dl_c')

    # ---------------- worst / best
    cols = [c for c in ('Peptide', 'length', 'charge', metric, 'cosine', 'bond_cosine', 'n_measured_ions',
                        'b_share_measured', 'b_share_predicted', 'top_ion_measured', 'top_ion_predicted', 'source_file', 'source_row', 'row') if c in d.columns]
    cols = list(dict.fromkeys(cols))                              # metric may already be one of the listed columns
    with tab_rank:
        k = st.slider('Spectra to list', 5, 200, 25, key='k')
        left, right = st.columns(2)
        with left:
            heading('Worst predicted')
            st.dataframe(d.nsmallest(k, metric)[cols].round(3), hide_index=True)
        with right:
            heading('Best predicted')
            st.dataframe(d.nlargest(k, metric)[cols].round(3), hide_index=True)
        st.caption('For de novo sequences, a very low score can also mean the sequence itself is wrong.')

    # ---------------- table
    with tab_data:
        st.write(f'{len(d):,} spectra after filtering (first 1,000 shown).')
        st.dataframe(d.head(1000), hide_index=True)
        if st.button('Prepare CSV of all filtered spectra', key='prep'):
            st.download_button('Download filtered metrics.csv', d.to_csv(index=False).encode('utf-8'),
                               file_name='filtered_metrics.csv', mime='text/csv', key='dl_all')


if __name__ == '__main__':
    main()
