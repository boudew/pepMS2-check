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
    rt_standard.pt   RT model calibrated (or retrained: rt_standard_retrained.pt) on the standard
    rt_calibrated.pt RT calibration fitted on this library (only when the library was its own calibration)

Retention time (needs rt_model.py, calibrate_rt.py and rt_DIA.pt next to this script; switched off with --no-rt)
    The measured RT comes from measured_meta.csv (rt_seconds, written by format_measured.py).
    Preferred: --rt-standard standard.csv, a measured standard (trusted peptides + RT, same LC method). The general
    model is calibrated on it (--rt-retrain-standard: retrained on it, for standards of >= 200 peptides), and the
    whole library is a fair test. Library peptides that are also in the standard are marked (rt_model_seen).
    --rt-model my_model.pt: use a calibration or retrained model made earlier (calibrate_rt.py, finetune_rt.py, the
    pepMS2 app or an earlier run) as it is.
    Without either: the library is its own calibration, cross-validated: unique sequences (I/L merged) are split into
    5 parts and every spectrum's predicted RT comes from a calibration fitted without its peptide. Every source file
    (= LC run) gets its own calibration; files with fewer than 20 peptides are pooled. The library is never used to
    retrain the model, because it can contain wrong identifications.
    Adds rt_measured_min, rt_predicted_min, rt_error_min (predicted - measured), rt_abs_error_min, rt_relative
    (model output 0-1) and rt_outlier to metrics.csv. Peptides the RT model cannot handle (modified, > 19 residues)
    keep their spectrum scores and get an empty RT.

Needs: numpy, pandas, matplotlib, torch, and peptide_model.py in the same folder as this script
       (+ rt_model.py, calibrate_rt.py, rt_DIA.pt for retention time).
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
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


# ---------------------------------------------------------------- retention time
def measured_rt_minutes(metrics):
    """Measured RT in minutes from the metadata columns, or None."""
    for col, factor in (('rt_seconds', 1 / 60), ('rt_minutes', 1.0), ('rt_min', 1.0)):
        if col in metrics and pd.to_numeric(metrics[col], errors='coerce').notna().any():
            return pd.to_numeric(metrics[col], errors='coerce').to_numpy(float) * factor
    return None


def _run_name(out_dir):
    return os.path.basename(os.path.normpath(os.path.dirname(os.path.abspath(out_dir)))) or 'library'


def add_rt(metrics, a, out_dir):
    """Adds the RT columns to metrics (in place). Returns (lines for summary.txt, info dict or None).
    RT source, in order of preference: a measured standard (--rt-standard), a calibrated / retrained model
    (--rt-model), or the library itself (cross-validated calibration)."""
    lines = []
    if a.no_rt:
        return lines, None
    base_path = os.path.join(HERE, 'rt_DIA.pt')
    rt_path = a.rt_model or base_path
    if not os.path.isfile(rt_path):
        if a.rt_model:
            sys.exit(f'RT model not found: {a.rt_model}')
        return ['Retention time: not predicted (rt_DIA.pt, rt_model.py and calibrate_rt.py are not next to this script)'], None
    try:
        from rt_model import RTPredictor, check_peptide
        from calibrate_rt import (library_rt_predictions, rt_outliers, calibrate_table, CalibrationError,
                                  model_from_standard, read_table, sequence_key, unit_mismatch, early_elution_cutoff)
    except ImportError as e:
        return [f'Retention time: not predicted (rt_model.py / calibrate_rt.py missing or outdated: {e})'], None
    model = RTPredictor(rt_path)
    peptides = metrics['Peptide'].tolist()
    early_opt = parse_early(a.rt_early_cutoff)                      # 'auto', None (off) or a cutoff in minutes
    meas = measured_rt_minutes(metrics)
    info = dict(model=os.path.basename(rt_path))
    files = metrics['source_file'].astype(str).to_numpy() if 'source_file' in metrics else None

    if a.rt_standard:                                               # preferred: a measured standard
        if model.is_calibrated:
            sys.exit('--rt-standard needs the general model (rt_DIA.pt), not an already calibrated --rt-model')
        if not os.path.isfile(a.rt_standard):
            sys.exit(f'RT standard not found: {a.rt_standard}')
        name = a.rt_standard_name or os.path.splitext(os.path.basename(a.rt_standard))[0]
        try:
            std_early = early_opt * 60 if isinstance(early_opt, float) and a.rt_standard_unit == 's' else early_opt
            model, msgs = model_from_standard(read_table(a.rt_standard), model, a.rt_standard_pep_col,
                                              a.rt_standard_rt_col, a.rt_standard_unit, name,
                                              retrain=a.rt_retrain_standard, log=lambda m: print(m, flush=True),
                                              early_cutoff=std_early)
        except CalibrationError as e:
            sys.exit(f'RT standard {os.path.basename(a.rt_standard)}: {e}')
        lines += [f'  standard: {m}' for m in msgs]
        lo, hi = model.calibration_info['rt_range']
        factor = 1 / 60 if a.rt_standard_unit == 's' else 1.0
        if meas is not None:
            problem = unit_mismatch([lo * factor, hi * factor], [np.nanmin(meas), np.nanmax(meas)])
            if problem:
                sys.exit(f'RT standard {os.path.basename(a.rt_standard)}: {problem}. Set the unit of the standard '
                         f'(--rt-standard-unit / "RT unit" in the app) and run again.')
        model.calibration_info['source_table'] = os.path.basename(a.rt_standard)
        fname = 'rt_standard_retrained.pt' if a.rt_retrain_standard else 'rt_standard.pt'
        model.save(os.path.join(out_dir, fname))
        info.update(saved=fname, source='standard')

    raw = np.full(len(metrics), np.nan)
    if model.is_calibrated:                                         # standard, or a calibrated / retrained .pt
        cal = model.calibration_info
        unit = (cal.get('unit') or 'min').strip()
        ok = np.array([check_peptide(p) is None for p in peptides])
        pred = np.full(len(metrics), np.nan)
        if ok.any():
            sub = [p for p, o in zip(peptides, ok) if o]
            raw[ok] = model.predict_raw(sub)
            pred[ok] = model.predict(sub)
        metrics['rt_relative'] = raw
        if unit.lower() == 'irt':
            metrics['rt_predicted_irt'] = pred
            return (lines + [f'Retention time: predicted as iRT ({cal.get("name", "")}); not compared with the '
                             'measured RT (different scale)'], None)
        pred = pred / 60 if unit == 's' else pred
        source = info.get('source', 'uploaded model')
        if source != 'standard' and meas is not None and 'rt_range' in cal:
            f = 1 / 60 if unit == 's' else 1.0
            problem = unit_mismatch([cal['rt_range'][0] * f, cal['rt_range'][1] * f], [np.nanmin(meas), np.nanmax(meas)])
            if problem:
                lines.append(f'  WARNING: RT units look inconsistent: {problem.replace("standard", "uploaded model")}')
        groups = np.array([f'{source} ({"retrained" if cal.get("finetuned") else "calibrated"})'] * len(metrics), dtype=object)
        trained_on = set(cal.get('training_keys') or [])
        if trained_on:                                              # which library peptides were in the calibration?
            metrics['rt_model_seen'] = [sequence_key(p) in trained_on for p in peptides]
        verb = 'retrained' if cal.get('finetuned') else 'calibrated'
        what = f'standard "{cal.get("name")}"' if source == 'standard' else f'model "{cal.get("name") or info["model"]}"'
        info.update(mode=source, retrained=bool(cal.get('finetuned')),
                    detail=f'{what}: {verb} on {cal.get("n_used", "?")} peptides'
                           + (f', RT {cal["rt_range"][0]:.1f}-{cal["rt_range"][1]:.1f} {unit}' if 'rt_range' in cal else ''))
        if meas is not None and np.isfinite(meas).any():          # reference: the library calibrating itself
            ref = library_rt_predictions(peptides, meas, RTPredictor(base_path), groups=files, folds=a.rt_folds,
                                         early_cutoff=early_opt)
            info['_reference_error'] = ref['predicted'] - meas         # summarised later, outside the early zone
    else:                                                           # no standard: the library calibrates itself
        if meas is None:
            res = library_rt_predictions(peptides, np.full(len(peptides), np.nan), model)
            metrics['rt_relative'] = res['raw']
            return ['Retention time: no measured RT in the metadata, only the relative RT (0-1) is given'], None
        res = library_rt_predictions(peptides, meas, model, groups=files, folds=a.rt_folds, early_cutoff=early_opt)
        raw, pred, groups = res['raw'], res['predicted'], res['group']
        metrics['rt_relative'] = raw
        lines += [f'  note: {n}' for n in res['notes']]
        info.update(mode='self', groups=res['groups'],
                    detail=f'no standard: calibrated on this library, {a.rt_folds}-fold cross-validated by sequence '
                           '(I/L merged)' + (', one calibration per source file' if len(res['groups']) > 1 else ''))
        if len(res['groups']) == 1 and res['groups'][0]['n_peptides'] >= 20:
            sel = np.isfinite(meas)
            try:
                full = RTPredictor(rt_path)
                calibrate_table(pd.DataFrame({'Peptide': np.array(peptides)[sel], 'RT': meas[sel]}), full,
                                unit='min', name=_run_name(out_dir), force=True)
                full.save(os.path.join(out_dir, 'rt_calibrated.pt'))
                info['saved'] = 'rt_calibrated.pt'
            except CalibrationError as e:
                lines.append(f'  note: no rt_calibrated.pt written ({e})')

    if meas is None:
        metrics['rt_predicted_min'] = pred
        return lines + ['Retention time: no measured RT in the metadata; only predicted RT given'], info
    err = pred - meas
    # early-elution zone: peptides that come off around the void time (weak retention) cannot be predicted; they are
    # marked, not flagged as outliers, and left out of the RT statistics
    early = np.zeros(len(err), bool)
    cutoffs = {}
    for g in set(groups[np.isfinite(err)]):
        sel = (groups == g) & np.isfinite(err)
        cut = early_opt if early_opt != 'auto' else early_elution_cutoff(meas[sel], err[sel], floor=1.0)
        if cut:
            early |= sel & (meas < cut)
            cutoffs[g] = float(cut)
    info['early_cutoffs'] = cutoffs
    info['early_mode'] = 'auto' if early_opt == 'auto' else ('off' if early_opt is None else 'set')
    if '_reference_error' in info:
        ref_err = np.abs(info.pop('_reference_error'))
        if np.isfinite(ref_err[~early]).any():
            info['library_calibration_error'] = float(np.nanmedian(ref_err[~early]))
    metrics['rt_measured_min'] = meas
    metrics['rt_predicted_min'] = pred
    metrics['rt_error_min'] = err
    metrics['rt_abs_error_min'] = np.abs(err)
    metrics['rt_early'] = np.where(np.isfinite(err), early, np.nan)
    # outliers: the main zone against its own typical error; the early zone against the early zone's typical error,
    # so the usual early blob is never flagged, but an RT that is extreme even for that zone still is
    flag = rt_outliers(np.where(early, np.nan, err), groups) | rt_outliers(np.where(early, err, np.nan), groups)
    metrics['rt_outlier'] = np.where(np.isfinite(err), flag, np.nan)
    metrics['rt_calibration'] = np.where(np.isfinite(err), groups, '')
    info['n'] = int(np.isfinite(err).sum())
    return lines, info


def parse_early(value):
    """--rt-early-cutoff: 'auto' (default), 0 / 'off' (no early zone) or a cutoff in minutes."""
    v = str(value).strip().lower()
    if v in ('auto', ''):
        return 'auto'
    if v in ('off', 'none', '0', '0.0'):
        return None
    try:
        return float(v)
    except ValueError:
        sys.exit(f'--rt-early-cutoff must be auto, off or a number of minutes, not {value!r}')


def rt_summary(metrics, info, say):
    if info is None or 'rt_abs_error_min' not in metrics:
        return
    m_all = metrics[np.isfinite(metrics['rt_abs_error_min'])]
    if m_all.empty:
        say('Retention time: no spectrum could be predicted')
        return
    early = m_all['rt_early'].fillna(0).astype(bool) if 'rt_early' in m_all else pd.Series(False, index=m_all.index)
    m = m_all[~early]
    e, ae = m['rt_error_min'].to_numpy(), m['rt_abs_error_min'].to_numpy()
    meas = m['rt_measured_min'].to_numpy()
    say(f'Retention time ({info["model"]}, {info["detail"]})')
    say(f'  spectra with predicted RT: {len(m_all):,} of {len(metrics):,} (others: modified or longer than 19 residues, or no measured RT)')
    cuts = info.get('early_cutoffs') or {}
    if early.any():
        how = 'detected' if info.get('early_mode') == 'auto' else 'set'
        say(f'  early-elution zone ({how}): {int(early.sum()):,} spectra elute before '
            + ', '.join(f'{c:.2f} min' + (f' ({g})' if len(cuts) > 1 else '') for g, c in cuts.items())
            + f'; weak retention, so their RT cannot be predicted (median error there '
              f'{m_all.loc[early, "rt_abs_error_min"].median():.2f} min). Left out of the numbers below; flagged as '
              'RT outlier only when extreme even for that zone.')
    elif info.get('early_mode') == 'auto':
        say('  early-elution zone: none detected')
    if m.empty:
        say('  all predicted spectra are in the early-elution zone')
        return
    say(f'  absolute error{" (outside the early zone)" if early.any() else ""}: median {np.median(ae):.2f} min | '
        f'95% of spectra within {np.percentile(ae, 95):.2f} min | mean signed error {np.mean(e):+.2f} min')
    groups = m['rt_calibration'].to_numpy() if 'rt_calibration' in m else np.zeros(len(m))
    base = np.concatenate([np.abs(meas[groups == g] - np.median(meas[groups == g])) for g in set(groups)])
    say(f'  reference: predicting the median RT for every peptide gives a median error of {np.median(base):.2f} min')
    if info.get('library_calibration_error') is not None:
        say(f'  reference: calibrating on this library itself (cross-validated) gives a median error of '
            f'{info["library_calibration_error"]:.2f} min')
    if np.median(ae) >= np.median(base):
        say('  WARNING: the RT predictions are no better than this reference.'
            + (f' The {"standard" if info.get("mode") == "standard" else "uploaded model"} was probably measured with a '
               'different LC method or gradient than this library.' if info.get('mode') in ('standard', 'uploaded model')
               else ' The RT model does not capture the retention of this LC method well.'))
    elif info.get('mode') in ('standard', 'uploaded model') and abs(np.mean(e)) > 2 * np.median(base):
        say(f'  WARNING: predictions are shifted by {np.mean(e):+.2f} min on average: the '
            f'{"standard" if info.get("mode") == "standard" else "uploaded model"} may come from another LC method, '
            'or the retention times drifted')
    if info.get('library_calibration_error') is not None and np.median(ae) > 1.5 * info['library_calibration_error'] \
            and np.median(ae) < np.median(base):
        say('  note: the library calibrating itself fits clearly better than the standard; check that the standard was '
            'measured with the same method and gradient as this library')
    r = np.corrcoef(m['rt_predicted_min'], meas)[0, 1] if len(m) > 2 else np.nan
    say(f'  predicted vs measured: r = {r:.3f} | measured RT {meas.min():.2f}-{meas.max():.2f} min')
    for g in info.get('groups', []):
        say(f'  {"model" if g.get("epochs") else "calibration"} {g["group"]}: {g["n_peptides"]:,} peptides, '
            + (f'retrained {g["epochs"]} epochs per part' if g.get('epochs') else
               f'{"straight line" if g["degree"] == 1 else "curve (degree %d)" % g["degree"]}')
            + f', r (before {"retraining" if g.get("epochs") else "calibration"}) = {g["r"]:.3f}'
            + ('  -> elution order reversed vs the reversed-phase training data (e.g. HILIC)' if g['reversed'] else ''))
    if 'rt_model_seen' in m and m['rt_model_seen'].any():
        seen = m['rt_model_seen'].astype(bool)
        if info.get('retrained'):
            say(f'  WARNING: {seen.mean():.0%} of these spectra are peptides this RT model was retrained on, so their error '
                'is too optimistic.')
        else:
            say(f'  note: {seen.sum():,} spectra ({seen.mean():.1%}) are peptides that were part of the calibration (e.g. '
                'spiked standard peptides); a calibration has only 2-3 parameters, so this barely flatters their error.')
        lab_in, lab_out = (('trained on', 'new to the model') if info.get('retrained') else
                           ('in the calibration', 'not in the calibration'))
        say(f'    peptides {lab_in + ":":<24} {seen.sum():,} spectra, median error {m.loc[seen, "rt_abs_error_min"].median():.2f} min')
        if (~seen).any():
            say(f'    peptides {lab_out + ":":<24} {(~seen).sum():,} spectra, median error '
                f'{m.loc[~seen, "rt_abs_error_min"].median():.2f} min  <- the fair number')
        else:
            say('    none of the peptides is new to the model: for a fair error, calibrate or retrain on this library '
                'instead (cross-validated)')
    n_out = int(np.nansum(m['rt_outlier'].astype(float)))
    say(f'  RT outliers (far from the typical error, possible misidentifications): {n_out:,}'
        + (f', of which {int(((m["rt_outlier"] == 1) & (m["cosine"] >= 0.8)).sum()):,} with cosine >= 0.8' if n_out else ''))
    if info.get('saved'):
        say(f'  RT model of this library written to {info["saved"]} (usable in the pepMS2 app and with --rt-model)')
    say('')
    say('  RT error by peptide length')
    say(f'  {"length":>6} {"spectra":>9} {"median abs error (min)":>23}')
    for ln_, g in m.groupby('length')['rt_abs_error_min']:
        say(f'  {ln_:>6} {len(g):>9,} {g.median():>23.2f}')
    if 'charge' in m and m['charge'].notna().any():
        say('  RT error by precursor charge')
        for z, g in m.groupby('charge')['rt_abs_error_min']:
            say(f'  {z:>5.0f}+ {len(g):>9,} {g.median():>23.2f}')


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
    has_rt = 'rt_abs_error_min' in metrics and np.isfinite(metrics['rt_abs_error_min']).any()
    fig, axes = plt.subplots(1, 3 if has_rt else 2, figsize=(16 if has_rt else 11, 4))
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
    if has_rt:
        ax = axes[2]
        m = metrics[np.isfinite(metrics['rt_abs_error_min'])]
        out = m['rt_outlier'] == 1
        early = m['rt_early'] == 1 if 'rt_early' in m else pd.Series(False, index=m.index)
        lo, hi = m['rt_measured_min'].min(), m['rt_measured_min'].max()
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls='--')
        if early.any():
            ax.axvspan(lo, m.loc[early, 'rt_measured_min'].max(), color='#9a9a96', alpha=0.12, lw=0)
            ax.scatter(m.loc[early, 'rt_measured_min'], m.loc[early, 'rt_predicted_min'], s=10, color='#9a9a96', alpha=0.6,
                       linewidths=0, label='early-eluting (not scored)')
        ok = ~out & ~early
        ax.scatter(m.loc[ok, 'rt_measured_min'], m.loc[ok, 'rt_predicted_min'], s=10, color=C_BAR, alpha=0.5, linewidths=0)
        if out.any():
            ax.scatter(m.loc[out, 'rt_measured_min'], m.loc[out, 'rt_predicted_min'], s=22, color=C_Y, marker='x',
                       label='RT outlier')
        if out.any() or early.any():
            ax.legend(frameon=False, labelcolor=INK2, loc='upper left')
        ax.set_xlabel('measured RT (min)', color=INK2)
        ax.set_ylabel('predicted RT (min)', color=INK2)
        ax.set_title(f'retention time: median error {m.loc[~early, "rt_abs_error_min"].median():.2f} min', loc='left',
                     color=INK, fontsize=11)
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
    ap.add_argument('--rt-model', default=None,
                    help='RT model (default rt_DIA.pt next to this script). An uncalibrated model is calibrated on this '
                         'library (cross-validated); a calibrated one (calibrate_rt.py / pepMS2 app) is used as it is')
    ap.add_argument('--rt-folds', type=int, default=5, help='cross-validation parts for the RT self-calibration (default 5)')
    ap.add_argument('--rt-standard', default=None,
                    help='measured RT standard (CSV/TSV/Excel: peptide + RT, same LC method): the preferred calibration')
    ap.add_argument('--rt-standard-unit', default='min', choices=['min', 's'], help='RT unit of the standard (default min)')
    ap.add_argument('--rt-standard-pep-col', default=None, help='peptide column of the standard (default: detected)')
    ap.add_argument('--rt-standard-rt-col', default=None, help='RT column of the standard (default: detected)')
    ap.add_argument('--rt-standard-name', default=None, help='name of the standard / LC method (default: file name)')
    ap.add_argument('--rt-early-cutoff', default='auto',
                    help='early-elution zone (peptides that elute too early to predict, around the void time): auto '
                         '(default, detected from the errors), off, or a cutoff in minutes. These spectra are marked, '
                         'not flagged as RT outliers, and left out of the RT statistics and of every calibration fit')
    ap.add_argument('--rt-retrain-standard', action='store_true',
                    help='retrain (fine-tune) the RT model on the standard instead of only calibrating; needs >= 200 peptides')
    ap.add_argument('--no-rt', action='store_true', help='do not predict retention time')
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

    rt_lines, rt_info = add_rt(metrics, a, a.out)

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
    if rt_lines or rt_info:
        say('')
        for line in rt_lines:
            say(line)
        rt_summary(metrics, rt_info, say)
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
                if 'rt_predicted_min' in metrics and np.isfinite(metrics['rt_predicted_min'].iloc[j]) \
                        and 'rt_measured_min' in metrics:
                    heading += (f'  |  RT {metrics["rt_measured_min"].iloc[j]:.2f} min '
                                f'(predicted {metrics["rt_predicted_min"].iloc[j]:.2f})')
                fig = mirror_figure(seqs[j], meas[j], pred[j], n_frag, heading)
                pdf.savefig(fig, facecolor=SURF)
                plt.close(fig)
    say('')
    say(f'results in {os.path.abspath(a.out)}: predicted.csv, metrics.csv, summary.txt, summary.png'
        + (', examples.pdf' if n_ex > 0 else '') + (f', {rt_info["saved"]}' if rt_info and rt_info.get('saved') else ''))


if __name__ == '__main__':
    main()
