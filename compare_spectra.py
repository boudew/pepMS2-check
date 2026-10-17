"""Compare spectra predicted by the trained model with measured spectra.

Usage
    python compare_spectra.py measured.csv --model peptide_predictor.pt --out comparison
    python compare_spectra.py measured.csv --model peptide_predictor.pt --out comparison --split split_assignment.tsv --unseen-only

    measured.csv comes from format_measured.py; measured_meta.csv next to it is picked up automatically
    (precursor charge, collision energy, RT, source file ...).

    --split split_assignment.tsv (written by the training notebook) marks every measured peptide that was in the
    training / validation / test set of the model; --unseen-only drops the ones the model has seen. I and L are
    treated as the same residue, like in the training split. A fair test on a new library should use unseen peptides.

Writes into the --out folder
    predicted.csv    predicted spectra, same format as the training file (y and b ions sum to 1)
    metrics.csv      one row per spectrum: cosine, spectral angle, Pearson r, top ions (+ metadata columns)
    summary.txt      overall numbers, per peptide length, per precursor charge
    summary.png      distribution of the cosine similarity, cosine by peptide length
    examples.pdf     mirror plots (measured up, predicted down), spread from worst to best match

Needs: numpy, pandas, matplotlib, torch, and peptide_model.py in the same folder as this script.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages               # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator           # noqa: E402
import numpy as np                                                 # noqa: E402
import pandas as pd                                                # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch                                                       # noqa: E402
from peptide_model import load_checkpoint, predict_batch           # noqa: E402

PROTON, WATER = 1.007276, 18.010565
RESIDUE = {  # monoisotopic residue masses (cysteine unmodified, as in the notebook and format_measured.py)
    'G': 57.02146, 'A': 71.03711, 'S': 87.03203, 'P': 97.05276, 'V': 99.06841, 'T': 101.04768,
    'C': 103.00919, 'L': 113.08406, 'I': 113.08406, 'N': 114.04293, 'D': 115.02694, 'Q': 128.05858,
    'K': 128.09496, 'E': 129.04259, 'M': 131.04049, 'H': 137.05891, 'F': 147.06841, 'R': 156.10111,
    'Y': 163.06333, 'W': 186.07931}
C_B, C_Y, C_BAR = '#2a78d6', '#eb6834', '#2a78d6'                  # b ions blue, y ions orange (colour-blind-safe pair)
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


# ---------------------------------------------------------------- similarity measures (all rows at once)
def cosine_rows(p, m, eps=1e-12):
    return (p * m).sum(1) / (np.sqrt((p * p).sum(1)) * np.sqrt((m * m).sum(1)) + eps)


def bond_vectors(arr, lens, n):
    """Per cleavage site j (between residue j and j+1): b_j + the complementary y_(L-j).
    Both ions come from the same peptide bond, so this ignores which half keeps the charge.
    Only sites where both ions exist in the table (j <= n and L-j <= n) are used. Returns (n_rows, n)."""
    j = np.arange(1, n + 1)[None, :]
    k = lens[:, None] - j
    valid = (k >= 1) & (k <= n)
    y = np.take_along_axis(arr, np.clip(k, 1, n) - 1, axis=1)
    return (y + arr[:, n:2 * n]) * valid


def spectral_angle_rows(cos):
    """1 - 2*angle/pi: 1 = identical direction, 0 = orthogonal (the normalised spectral angle)."""
    return 1 - 2 * np.arccos(np.clip(cos, 0, 1)) / np.pi


def pearson_rows(p, m, valid):
    """Pearson r over the ion positions that exist for each peptide (NaN if either spectrum is flat)."""
    n = valid.sum(1)
    dp = (p - ((p * valid).sum(1) / n)[:, None]) * valid
    dm = (m - ((m * valid).sum(1) / n)[:, None]) * valid
    den = np.sqrt((dp * dp).sum(1) * (dm * dm).sum(1))
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (dp * dm).sum(1) / den
    return np.where(den > 0, r, np.nan)


# ---------------------------------------------------------------- plots
def fragment_mz(seq):
    r = np.array([RESIDUE[a] for a in seq])
    return np.cumsum(r)[:-1] + PROTON, np.cumsum(r[::-1])[:-1] + WATER + PROTON      # b1.., y1.. (1+)


def ion_sticks(seq, vec, n_frag):
    b_mz, y_mz = fragment_mz(seq)
    n = min(len(seq) - 1, n_frag)
    k = np.arange(1, n + 1)
    return pd.DataFrame({'ion': [f'y{i}' for i in k] + [f'b{i}' for i in k], 'type': ['y'] * n + ['b'] * n,
                         'mz': np.concatenate([y_mz[:n], b_mz[:n]]),
                         'intensity': np.concatenate([vec[:n], vec[n_frag:n_frag + n]])})


def draw_half(ax, tab, sign, n_label=6):
    top = tab['intensity'].max()
    if not top > 0:
        return
    rel = 100 * tab['intensity'] / top
    for t, colour in (('b', C_B), ('y', C_Y)):
        s = tab['type'] == t
        ax.vlines(tab.loc[s, 'mz'], 0, sign * rel[s], color=colour, lw=1.6)
    for j in rel.nlargest(n_label).index:
        if rel[j] < 3:
            continue
        ax.text(tab.at[j, 'mz'], sign * (rel[j] + 2), tab.at[j, 'ion'], rotation=90, ha='center',
                va='bottom' if sign > 0 else 'top', fontsize=8, color=INK2)


def mirror_figure(seq, meas, pred, n_frag, heading):
    tm, tp = ion_sticks(seq, meas, n_frag), ion_sticks(seq, pred, n_frag)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    draw_half(ax, tm, +1)
    draw_half(ax, tp, -1)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlim(tm['mz'].min() - 30, tm['mz'].max() + 30)
    ax.set_ylim(-135, 135)
    ax.set_yticks(range(-100, 101, 50))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{abs(v):.0f}'))
    ax.text(0.01, 0.97, 'measured', transform=ax.transAxes, va='top', fontsize=9, color=INK2)
    ax.text(0.01, 0.03, 'predicted', transform=ax.transAxes, va='bottom', fontsize=9, color=INK2)
    ax.set_xlabel('m/z (1+ fragments)', color=INK2)
    ax.set_ylabel('% of base peak', color=INK2)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.set_title(heading, loc='left', color=INK, fontsize=11)
    ax.legend(handles=[Line2D([0], [0], color=C_B, lw=2, label='b ions'), Line2D([0], [0], color=C_Y, lw=2, label='y ions')],
              loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, labelcolor=INK2)
    fig.tight_layout()
    return fig


def summary_figure(metrics, path):
    cos, ln = metrics['cosine'].to_numpy(), metrics['length'].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor(SURF)
    for ax in axes:
        ax.set_facecolor(SURF)
        ax.grid(axis='y', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK2)
    ax = axes[0]
    ax.hist(cos, bins=np.linspace(0, 1, 41), color=C_BAR, alpha=0.85)
    med = float(np.median(cos))
    top = ax.get_ylim()[1] * 1.15                                   # headroom so the label never touches a bar
    ax.set_ylim(0, top)
    ax.axvline(med, color=INK2, lw=1, ls='--')
    ax.text(med, top * 0.99, f' median {med:.3f}', color=INK2, fontsize=9, va='top')
    ax.set_xlabel('cosine similarity, measured vs predicted', color=INK2)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel('spectra', color=INK2)
    ax.set_title(f'{len(cos):,} spectra', loc='left', color=INK, fontsize=11)
    ax = axes[1]
    lengths = sorted(set(ln))
    bp = ax.boxplot([cos[ln == L] for L in lengths], positions=lengths, widths=0.6,
                    patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.5),
                    whiskerprops=dict(color=INK2, lw=0.8), capprops=dict(color=INK2, lw=0.8))
    for box in bp['boxes']:
        box.set(facecolor=C_BAR, alpha=0.55, edgecolor=C_BAR)
    ticks = list(range(min(lengths), max(lengths) + 1))
    ax.set_xticks(ticks)
    step = 1 if len(ticks) <= 18 else 2                            # keep long length ranges readable
    ax.set_xticklabels([str(t) if (t - ticks[0]) % step == 0 else '' for t in ticks])   # boxplot() installs its own labels
    ax.set_xlim(min(lengths) - 0.8, max(lengths) + 0.8)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('peptide length (residues)', color=INK2)
    ax.set_ylabel('cosine similarity', color=INK2)
    ax.set_title('by peptide length (box = quartiles)', loc='left', color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURF)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('measured', help='training-format CSV written by format_measured.py')
    ap.add_argument('--model', default='peptide_predictor.pt', help='file written by the notebook export cell')
    ap.add_argument('--out', default='comparison', help='output folder (default: comparison)')
    ap.add_argument('--meta', default=None, help='metadata CSV (default: <measured>_meta.csv if it exists)')
    ap.add_argument('--examples', type=int, default=12, help='mirror plots in examples.pdf (0 = none)')
    ap.add_argument('--batch-size', type=int, default=4096)
    ap.add_argument('--energy', type=float, default=None,
                    help='collision energy (NCE) to predict for ALL spectra (default: an energy/nce/collision_energy column of the metadata file)')
    ap.add_argument('--split', default=None,
                    help='split_assignment.tsv from the training notebook: adds a train_set column (train / val / test / new) to metrics.csv')
    ap.add_argument('--unseen-only', action='store_true',
                    help='with --split: compare only peptides that were not in the training or validation set')
    ap.add_argument('--charge', type=int, default=None,
                    help='precursor charge to predict for ALL spectra (default: the charge column of the metadata file)')
    a = ap.parse_args()

    if not os.path.isfile(a.measured):
        sys.exit(f'Measured file not found: {a.measured}')
    if not os.path.isfile(a.model):
        sys.exit(f'Model file not found: {a.model}\nRun the export cell at the end of the notebook first.')
    os.makedirs(a.out, exist_ok=True)

    # ---- measured spectra
    df = pd.read_csv(a.measured, index_col=0)
    if 'Peptide' not in df.columns:
        sys.exit("The measured file needs a 'Peptide' column (use the output of format_measured.py).")
    df = df.dropna(subset=['Peptide'])
    names = [c for c in df.columns if c != 'Peptide']
    n_frag = len(names) // 2
    if len(names) % 2 or n_frag == 0 or names[0] != 'y1' or names[n_frag] != 'b1':
        sys.exit('Expected the columns Peptide, y1..yN, b1..bN (as in the training file).')

    # ---- model
    model, mmeta = load_checkpoint(a.model)
    if mmeta['n_frag'] != n_frag:
        sys.exit(f'The model predicts {mmeta["n_frag"]} ions per type but the measured file has {n_frag}.')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'model: {a.model} | device: {device} | {len(df):,} measured spectra')

    meta_path = a.meta or os.path.splitext(a.measured)[0] + '_meta.csv'
    meta = pd.read_csv(meta_path, index_col=0).reindex(df.index) if os.path.isfile(meta_path) else None

    seqs_all = df['Peptide'].astype(str).str.strip().str.upper().tolist()
    vocab = set(mmeta['aa'])
    ok = np.array([len(s) >= 2 and all(c in vocab for c in s) for s in seqs_all])
    n_bad_seq = int((~ok).sum())
    train_set = None
    n_seen = 0
    if a.split:
        if not os.path.isfile(a.split):
            sys.exit(f'Split file not found: {a.split}')
        sp = pd.read_csv(a.split, sep='\t')
        norm = lambda s: str(s).strip().upper().replace('I', 'L')
        role = {}
        for pep, st in zip(sp['Peptide'], sp['set']):                # one role per peptide (the training split guarantees it)
            role[norm(pep)] = st
        train_set = np.array([role.get(norm(s), 'new') for s in seqs_all])
        if a.unseen_only:
            seen = np.isin(train_set, ['train', 'val'])
            n_seen = int((ok & seen).sum())
            ok &= ~seen
    zs = mmeta.get('charges') or []
    charges = None
    n_bad_z = 0
    if zs:                                                          # the model needs the precursor charge of every spectrum
        if a.charge is not None:
            zall = np.full(len(df), a.charge, dtype=float)
        elif meta is not None and 'charge' in meta:
            zall = pd.to_numeric(meta['charge'], errors='coerce').to_numpy(dtype=float)
        else:
            sys.exit(f'This model needs the precursor charge ({zs}). Provide the metadata file (measured_meta.csv with a '
                     f'charge column) or use --charge N.')
        z_ok = np.isin(zall, zs)
        n_bad_z = int((ok & ~z_ok).sum())
        ok &= z_ok
    es = mmeta.get('energies') or []
    energies = None
    n_bad_e = 0
    if es:                                                          # the model also needs the collision energy of every spectrum
        if a.energy is not None:
            eall = np.full(len(df), a.energy, dtype=float)
        else:
            ecol = None
            if meta is not None:
                low = {str(c).lower().strip(): c for c in meta.columns}
                ecol = next((low[n] for n in ('energy', 'nce', 'collision_energy', 'collision energy', 'hcd_energy', 'hcd') if n in low), None)
            if ecol is None:
                sys.exit(f'This model needs the collision energy (trained on NCE {es}). Add an energy column (energy / nce / '
                         f'collision_energy) to the metadata file or use --energy N.')
            eall = pd.to_numeric(meta[ecol].astype(str).str.extract(r'(\d+(?:\.\d+)?)')[0], errors='coerce').to_numpy(dtype=float)
        e_ok = np.isfinite(eall) & (eall >= min(es)) & (eall <= max(es))
        n_bad_e = int((ok & ~e_ok).sum())
        ok &= e_ok
    charges = zall[ok].astype(int) if zs else None
    energies = eall[ok] if es else None
    n_skipped = int((~ok).sum())
    if not ok.any():
        sys.exit('No spectrum can be compared (peptides too short or with residues the model has not seen).')
    idx = df.index[ok]
    seqs = [s for s, o in zip(seqs_all, ok) if o]
    train_set = train_set[ok] if train_set is not None else None
    lens = np.array([len(s) for s in seqs])
    k = np.arange(1, n_frag + 1)[None, :]
    v = (k <= lens[:, None] - 1).astype(np.float64)
    valid = np.concatenate([v, v], axis=1)                          # ion positions that exist for each peptide
    meas = df[names].fillna(0).to_numpy(dtype=np.float64)[ok] * valid

    # ---- predict (batched, GPU if available), normalise like the training data: y + b sum to 1
    pred = predict_batch(model, mmeta, seqs, charges=charges, energies=energies, device=device, batch_size=a.batch_size).astype(np.float64)
    tot = pred.sum(1, keepdims=True)
    pred = np.divide(pred, tot, out=np.zeros_like(pred), where=tot > 0)

    # ---- metrics
    cos = cosine_rows(pred, meas)
    sa = spectral_angle_rows(cos)
    pr = pearson_rows(pred, meas, valid)
    base = cosine_rows(meas.mean(0, keepdims=True) * valid, meas)   # same average spectrum for every peptide
    bond_cos = cosine_rows(bond_vectors(pred, lens, n_frag), bond_vectors(meas, lens, n_frag))   # where does it break?
    b_meas = meas[:, n_frag:].sum(1) / np.maximum(meas.sum(1), 1e-12)                             # share of intensity in b ions
    b_pred = pred[:, n_frag:].sum(1) / np.maximum(pred.sum(1), 1e-12)
    n_ions = (meas > 0).sum(1)
    ion_frac = n_ions / (2 * np.minimum(lens - 1, n_frag))                                         # share of possible ions observed
    name_arr = np.array(names)
    top_m, top_p = name_arr[meas.argmax(1)], name_arr[pred.argmax(1)]
    metrics = pd.DataFrame({'Peptide': seqs, 'length': lens, 'cosine': cos, 'spectral_angle': sa, 'pearson': pr,
                            'bond_cosine': bond_cos, 'b_share_measured': b_meas, 'b_share_predicted': b_pred,
                            'n_measured_ions': n_ions, 'ion_fraction': ion_frac, 'top_ion_measured': top_m,
                            'top_ion_predicted': top_p, 'top_ion_match': top_m == top_p}, index=idx)
    if es:
        metrics['energy'] = energies                                # the collision energy each spectrum was predicted at
    if train_set is not None:
        metrics['train_set'] = train_set
    if meta is not None:
        meta = meta.reindex(idx)
        metrics = metrics.join(meta[[c for c in meta.columns if c not in metrics.columns]])

    pred_out = pd.DataFrame(pred, columns=names, index=idx)
    pred_out.insert(0, 'Peptide', seqs)
    pred_out.to_csv(os.path.join(a.out, 'predicted.csv'), index_label='', float_format='%.10g')
    metrics.to_csv(os.path.join(a.out, 'metrics.csv'), index_label='row', float_format='%.6g')

    # ---- summary text
    L = []

    def say(s=''):
        print(s)
        L.append(s)

    q = np.percentile(cos, [25, 75])
    say(f'Spectra compared: {len(cos):,}' + (f'   (skipped {n_skipped:,}: ' + ', '.join(
        x for x in (f'{n_bad_seq:,} too short or with residues the model has not seen' if n_bad_seq else '',
                    f'{n_bad_z:,} with a precursor charge outside the model\'s {zs}' if n_bad_z else '',
                    f'{n_bad_e:,} with a collision energy outside {min(es):g}-{max(es):g}' if n_bad_e else '',
                    f'{n_seen:,} peptides already in the training or validation set' if n_seen else '') if x) + ')' if n_skipped else ''))
    say(f'Model: {os.path.basename(a.model)} | {n_frag} ions per type | trained on peptides up to {mmeta["max_len_trained"]} residues'
        + (f' | precursor charges {zs}' + (f' (all predicted as {a.charge}+)' if a.charge is not None else '') if zs else ' | no charge input')
        + (f' | NCE {min(es):g}-{max(es):g}' + (f' (all predicted at {a.energy:g})' if a.energy is not None else '') if es else ' | no energy input'))
    n_long = int((lens > mmeta['max_len_trained']).sum())
    if n_long:
        say(f'  {n_long:,} peptides are longer than anything the model was trained on')
    n_cut = int((lens > n_frag + 1).sum())
    if n_cut:
        say(f'  {n_cut:,} peptides have more than {n_frag + 1} residues: only y1-y{n_frag} and b1-b{n_frag} are compared')
    say('')
    say('Cosine similarity, measured vs predicted (all y/b positions)')
    say(f'  median {np.median(cos):.3f} | mean {cos.mean():.3f} | quartiles {q[0]:.3f} - {q[1]:.3f}')
    say('  share of spectra with cosine >= 0.9: %.1f%% | >= 0.8: %.1f%% | >= 0.7: %.1f%% | < 0.5: %.1f%%'
        % tuple(100 * x for x in ((cos >= .9).mean(), (cos >= .8).mean(), (cos >= .7).mean(), (cos < .5).mean())))
    say(f'  reference: the same average measured spectrum for every peptide gives a median of {np.median(base):.3f}')
    say(f'Spectral angle similarity: median {np.median(sa):.3f}')
    say(f'Pearson r over existing ion positions: median {np.nanmedian(pr):.3f}')
    say(f'Bond-level cosine (b_j + complementary y, ignores which half keeps the charge): median {np.median(bond_cos):.3f}')
    say(f'Share of intensity in b ions: measured median {np.median(b_meas):.2f} | predicted median {np.median(b_pred):.2f}')
    say(f'Most intense ion identical in measured and predicted: {100 * metrics["top_ion_match"].mean():.1f}%')
    say('')
    say('By minimum number of measured ions (a stricter filter keeps easier spectra)')
    say(f'  {"min ions":>8} {"spectra":>9} {"kept":>6} {"median cosine":>14}')
    for mi in range(1, int(min(n_ions.max(), 6)) + 1):
        sel = n_ions >= mi
        if sel.sum() == 0:
            break
        say(f'  {mi:>8} {int(sel.sum()):>9,} {100 * sel.mean():>5.0f}% {np.median(cos[sel]):>14.3f}')
    say('')
    say('By peptide length')
    say(f'  {"length":>6} {"spectra":>9} {"median cosine":>14}')
    for ln_, g in metrics.groupby('length')['cosine']:
        say(f'  {ln_:>6} {len(g):>9,} {g.median():>14.3f}')
    if 'charge' in metrics and metrics['charge'].notna().any():
        say('')
        say('By precursor charge')
        say(f'  {"charge":>6} {"spectra":>9} {"median cosine":>14}')
        for z, g in metrics.groupby('charge')['cosine']:
            say(f'  {z:>5.0f}+ {len(g):>9,} {g.median():>14.3f}')
    if train_set is not None:
        say('')
        say('By membership of the model\'s training data (train / val were seen, test and new were not)')
        say(f'  {"set":>6} {"spectra":>9} {"median cosine":>14}')
        for s_, g_ in metrics.groupby('train_set')['cosine']:
            say(f'  {s_:>6} {len(g_):>9,} {g_.median():>14.3f}')
    if es:
        say('')
        say('By collision energy (NCE)')
        say(f'  {"NCE":>6} {"spectra":>9} {"median cosine":>14}')
        for e_, g_ in metrics.groupby('energy')['cosine']:
            say(f'  {e_:>6g} {len(g_):>9,} {g_.median():>14.3f}')
    with open(os.path.join(a.out, 'summary.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')

    # ---- figures
    summary_figure(metrics, os.path.join(a.out, 'summary.png'))
    n_ex = min(a.examples, len(cos))
    if n_ex > 0:
        order = np.argsort(cos)
        picks = np.unique(np.round(np.linspace(0, len(order) - 1, n_ex)).astype(int))
        with PdfPages(os.path.join(a.out, 'examples.pdf')) as pdf:
            for j in order[picks]:
                z = metrics['charge'].iloc[j] if 'charge' in metrics else np.nan
                heading = (f'{seqs[j]}' + (f'  |  precursor {int(z)}+' if pd.notna(z) else '')
                           + f'  |  cosine {cos[j]:.3f}  |  spectral angle {sa[j]:.3f}')
                fig = mirror_figure(seqs[j], meas[j], pred[j], n_frag, heading)
                pdf.savefig(fig, facecolor=SURF)
                plt.close(fig)
    say('')
    say(f'results in {os.path.abspath(a.out)}: predicted.csv, metrics.csv, summary.txt, summary.png'
        + (', examples.pdf' if n_ex > 0 else ''))


if __name__ == '__main__':
    main()
