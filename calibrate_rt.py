"""
calibrate_rt.py — calibrate the RT model to your own LC method.

The RT model predicts retention on a relative 0-1 scale. This script fits a map from that
scale to your measured RT (minutes, seconds or iRT — whatever unit your file uses) and saves
a calibrated copy of the model that the app can load directly.

Input: a CSV / TSV / Excel file with one peptide column and one RT column, e.g. a PEAKS,
MaxQuant, DIA-NN or Skyline export of confidently identified peptides from one run/method.
Column names are detected automatically (Peptide / Sequence / seq ... and RT / tr / Retention time / iRT ...);
override with --pep-col / --rt-col.

Steps
  1. drop peptides the model cannot handle (modified, > 19 residues, non-standard residues)
  2. one RT per sequence (median over repeated identifications)
  3. robust fit: fit, drop outliers (> 4 x MAD from the fit, e.g. wrong IDs), refit
  4. honest error estimate: 5-fold cross-validation split by sequence
  5. final fit on all clean peptides -> saved into the model file

Curve: straight line by default for < 50 peptides, 2nd-order curve from 50 peptides on
(tested on the pea alcalase data: ~10% lower error with enough peptides). Force with --degree.

    python calibrate_rt.py my_ids.csv --model rt_DIA.pt --out rt_DIA_myLC.pt --name "C18 40 min" --unit min

The same calibration is used by the pepMS2 app when you upload a peptide/RT table there.
"""
import argparse
import io
import os
import re
import sys

import numpy as np
import pandas as pd

from rt_model import RTPredictor, check_peptide

PEP_COLS = ["peptide", "sequence", "sequence (backbone)", "stripped.sequence", "stripped sequence", "seq",
            "peptide sequence", "peptidesequence", "modified sequence"]
MOD_COLS = ["ptm", "modifications", "modification", "mod", "mods"]
RT_COLS = ["rt", "retention time", "retentiontime", "rt (min)", "rt [min]", "rt_min", "rt (minutes)",
           "retention time (min)", "rt (s)", "rt [s]", "rt (sec)", "rt (seconds)", "rt_seconds", "rt_s",
           "retention time (s)", "tr", "observed rt", "apex rt", "irt"]
MIN_PEPTIDES = 5
MIN_R = 0.5           # minimum |r| model output vs RT: catches wrong columns / nonsense; reversed order (HILIC) is fine
MIN_R_RETRAIN = 0.3   # before retraining a weaker start is fine (retraining fixes it), nonsense is still caught


class CalibrationError(ValueError):
    pass


def read_table(source, name=None):
    """source: path or bytes. name: file name (for the extension) when source is bytes."""
    name = name or (source if isinstance(source, str) else "")
    ext = os.path.splitext(name)[1].lower()
    buf = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    if ext in (".xlsx", ".xls"):
        try:
            return pd.read_excel(buf)
        except ImportError:
            raise CalibrationError("Reading Excel needs the 'openpyxl' package; save the table as CSV instead.")
    try:
        if ext in (".tsv", ".tab"):
            return pd.read_csv(buf, sep="\t", encoding="utf-8-sig")
        raw = buf.read() if hasattr(buf, "read") else open(buf, "rb").read()
        head = raw[:20000].decode("utf-8-sig", errors="replace").splitlines()[0] if raw else ""
        # separator = whichever of tab / comma / semicolon splits the header line most (no free sniffing:
        # peak lists contain ';' and ':')
        sep = max(["\t", ",", ";"], key=lambda d: head.count(d)) if head else ","
        return pd.read_csv(io.BytesIO(raw), sep=sep, encoding="utf-8-sig")
    except Exception as e:
        raise CalibrationError(f"Could not read the table: {e}")


def _pick(df, wanted, given, what, flag):
    if given:
        if given not in df.columns:
            raise CalibrationError(f"Column '{given}' not found. Columns are: {list(df.columns)}")
        return given
    low = {str(c).strip().lower(): c for c in df.columns}
    for w in wanted:
        if w in low:
            return low[w]
    for w in wanted:                       # prefix match, e.g. "Sequence (backbone)", "Peptide sequence (stripped)"
        if len(w) > 3:
            for k, c in low.items():
                if k.startswith(w) and not k.startswith(("peptide mass", "sequence coverage")):
                    return c
    raise CalibrationError(f"Could not find the {what} column (looked for e.g. {', '.join(wanted[:4])}). "
                           f"Columns are: {list(df.columns)}. Name it with {flag}.")


def guess_columns(df):
    """Auto-detected (peptide column, RT column); None where nothing matched."""
    out = []
    for wanted in (PEP_COLS, RT_COLS):
        try:
            out.append(_pick(df, wanted, None, "", ""))
        except CalibrationError:
            out.append(None)
    return tuple(out)


def guess_unit(column_name, values=None):
    """RT unit from the column name (RT [s], rt (sec), rt_seconds ... -> 's'; iRT -> 'iRT'; RT (min) -> 'min'),
    otherwise from the values: RTs above 300 are seconds (no LC gradient runs 300 minutes)."""
    n = str(column_name).strip().lower()
    if "irt" in n:
        return "iRT"
    if re.search(r"(\[s\]|\(s\)|\bsec|second|_s$)", n):
        return "s"
    if re.search(r"(min|\[m\]|\(m\))", n):
        return "min"
    if values is not None:
        v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
        if len(v) and v.max() > 300:
            return "s"
    return "min"


def unit_mismatch(standard_minutes, library_minutes):
    """Message if two sets of RTs (both already converted to minutes) differ by about a factor 60, else None."""
    a = np.nanmedian(np.asarray(standard_minutes, dtype=float))
    b = np.nanmedian(np.asarray(library_minutes, dtype=float))
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b <= 0:
        return None
    ratio = a / b
    if 15 < ratio < 240:
        return (f"the standard's RTs are about {ratio:.0f}x larger than the library's (median {a:.1f} vs {b:.2f} min): "
                "the standard is probably in seconds but set to minutes (or the library RT is in minutes, not seconds)")
    if 1 / 240 < ratio < 1 / 15:
        return (f"the standard's RTs are about {1 / ratio:.0f}x smaller than the library's (median {a:.2f} vs {b:.1f} min): "
                "the standard is probably in minutes but set to seconds (or the library RT is not in seconds)")
    return None


def clean_sequence(p):
    p = str(p).strip()
    m = re.fullmatch(r"[A-Z\-_]\.(.+)\.[A-Z\-_]", p)   # K.PEPTIDE.R  -> PEPTIDE
    if m:
        p = m.group(1)
    return p.strip("_")                               # _PEPTIDE_ (MaxQuant)  -> PEPTIDE


def robust_fit(x, y, degree, n_iter=3, k=4.0):
    keep = np.ones(len(x), bool)
    for _ in range(n_iter):
        coef = np.polyfit(x[keep], y[keep], degree)
        res = y - np.polyval(coef, x)
        mad = np.median(np.abs(res[keep] - np.median(res[keep]))) * 1.4826
        new_keep = np.abs(res) <= max(k * mad, 1e-9)
        if new_keep.sum() < degree + 2 or (new_keep == keep).all():
            break
        keep = new_keep
    return np.polyfit(x[keep], y[keep], degree), keep


def cross_validate(x, y, degree, folds=5, seed=0, early=None):
    """Out-of-fold error; folds split by unique sequence (one row per sequence here). early: peptides of the
    early-elution zone, left out of every training part (their error is still computed)."""
    n = len(x)
    if n < 2 * folds:
        return None
    early = np.zeros(n, bool) if early is None else np.asarray(early, bool)
    idx = np.random.default_rng(seed).permutation(n)
    err = np.empty(n)
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        train = train[~early[train]]
        coef, _ = robust_fit(x[train], y[train], degree)
        err[test] = np.polyval(coef, x[test]) - y[test]
    return err


def calibrate_table(raw_df, model, pep_col=None, rt_col=None, degree=None, keep_modified=False,
                    name="", unit="", force=False, min_r=MIN_R, early_cutoff="auto"):
    """Fit a calibration. Sets model.calibration / model.calibration_info and returns
    (data, messages): per-peptide DataFrame and a list of human-readable notes.
    Raises CalibrationError if the table cannot be used."""
    if model.is_calibrated:
        raise CalibrationError("This RT model is already calibrated; start from the uncalibrated model (rt_DIA.pt).")
    msgs = []
    pc = _pick(raw_df, PEP_COLS, pep_col, "peptide", "--pep-col")
    rc = _pick(raw_df, RT_COLS, rt_col, "RT", "--rt-col")
    msgs.append(f"Read {len(raw_df)} rows (peptide column '{pc}', RT column '{rc}')")

    low = {str(c).strip().lower(): c for c in raw_df.columns}
    mc = next((low[m] for m in MOD_COLS if m in low), None)
    df = pd.DataFrame({"peptide": raw_df[pc].map(clean_sequence),
                       "rt": pd.to_numeric(raw_df[rc], errors="coerce")})
    n_bad_rt = int(df.rt.isna().sum())
    if n_bad_rt == len(df):
        raise CalibrationError(f"Column '{rc}' has no numeric RT values; pick the column with the retention times.")
    reason = df["peptide"].map(check_peptide)
    df["skip_reason"] = reason.map(lambda r: None if not isinstance(r, str) else
                                   ("modified / non-standard residues" if "non-standard" in r else "longer than 19 residues"))
    if mc is not None and not keep_modified:
        has_mod = raw_df[mc].fillna("").astype(str).str.strip().replace({"Unmodified": "", "nan": ""}) != ""
        df.loc[has_mod & df.skip_reason.isna(), "skip_reason"] = f"modified (flagged in '{mc}' column)"
    df = df.dropna(subset=["rt"])
    if n_bad_rt:
        msgs.append(f"Skipped {n_bad_rt} rows without a numeric RT")
    skipped = df[df.skip_reason.notna()]
    if len(skipped):
        msgs.append(f"Skipped {len(skipped)} rows the model can't handle: " +
                    "; ".join(f"{n} {r}" for r, n in skipped.skip_reason.value_counts().items()))
    df = df[df.skip_reason.isna()]

    grp = df.groupby("peptide").rt
    data = pd.DataFrame({"measured_rt": grp.median(), "n_ids": grp.size(),
                         "rt_spread": grp.max() - grp.min()}).reset_index()
    n = len(data)
    msgs.append(f"{n} unique peptides usable for calibration")
    if n < MIN_PEPTIDES:
        raise CalibrationError(f"Only {n} usable peptides; at least {MIN_PEPTIDES} are needed.")
    if n < 20:
        msgs.append("Fewer than 20 peptides: the calibration will be rough")

    x = model.predict_raw(list(data.peptide)).astype(float)
    y = data.measured_rt.values.astype(float)
    degree = int(degree or (1 if n < 50 else 2))
    floor = early_floor(unit)
    if floor is None or early_cutoff is None:
        coef, keep = robust_fit(x, y, degree)
        early, cut = np.zeros(n, bool), None
    else:
        coef, keep, early, cut = fit_without_early(x, y, degree, floor, early_cutoff)
    if cut is not None:
        msgs.append(f"Early-elution zone: {int(early.sum())} peptides below {cut:.2f} {unit or 'RT units'} elute too early "
                    "to predict (weak retention); left out of the fit")
    cv = cross_validate(x, y, degree, early=early)
    r = float(np.corrcoef(x[keep], y[keep])[0, 1])
    if not np.isfinite(r) or (abs(r) < min_r and not force):
        raise CalibrationError(
            f"Model output and your RTs barely agree (r = {r:.2f}). Usually the wrong RT or peptide column was "
            f"picked, the file mixes several LC methods, or the identifications are unreliable.")
    if r < 0:
        msgs.append("Elution order is reversed compared with the reversed-phase training data (e.g. HILIC): "
                    "the calibration maps it the other way round")
    grid = np.linspace(x.min(), x.max(), 200)
    steps = np.diff(np.polyval(coef, grid))
    if degree > 1 and not (np.all(steps > 0) or np.all(steps < 0)):
        msgs.append("The fitted curve bends back inside your RT range: use a straight line (degree 1)")

    data["model_output"] = x
    data["predicted_rt"] = np.polyval(coef, x)
    data["residual"] = data.predicted_rt - data.measured_rt
    data["used_in_fit"] = keep
    data["early_elution"] = early
    if cv is not None:
        data["cv_residual"] = cv

    info = {"name": name, "unit": unit, "n_peptides": int(n), "n_used": int(keep.sum()), "degree": degree,
            "reversed": bool(r < 0), "training_keys": sorted({sequence_key(p) for p in data.peptide}),
            "early_cutoff": cut, "n_early": int(early.sum()),
            "rt_range": (float(y.min()), float(y.max())),
            "model_output_range": (float(x.min()), float(x.max())), "pearson_r": round(r, 4)}
    if cv is not None:                                  # error outside the early-elution zone
        cvl = np.abs(cv[~early])
        info["cv_median_abs_error"] = round(float(np.median(cvl)), 3)
        info["cv_abs_error_95pct"] = round(float(np.percentile(cvl, 95)), 3)
    model.calibration = coef
    model.calibration_info = info
    return data, msgs


def sequence_key(p):
    """Same peptide for splitting purposes: upper case, I and L merged (as in the training split)."""
    return str(p).strip().upper().replace("I", "L")


def library_rt_predictions(peptides, rt, model, groups=None, folds=5, min_peptides=20, seed=0, early_cutoff="auto",
                           floor=1.0):
    """Cross-validated RT predictions for a measured library that is also the calibration set.

    Every spectrum's predicted RT comes from a calibration fitted WITHOUT its peptide: the unique sequences
    (I/L merged) are split into `folds` parts; all spectra of one sequence fall in the same part.
    groups: e.g. the source file of each spectrum. Each group (= one LC run/method) is calibrated on its own;
    groups with fewer than `min_peptides` sequences are pooled together.
    rt: measured RT in any unit; predictions come back in the same unit.
    The early-elution zone (peptides that elute too early to predict) is left out of every fit (auto-detected
    per training part, or early_cutoff in RT units, None = off).
    Only a calibration is fitted here (no retraining): a library can contain wrong identifications, so the model
    itself is retrained only on a trusted standard (model_from_standard).

    Returns dict with
        raw        model output (relative 0-1), NaN where the model cannot handle the peptide
        predicted  cross-validated predicted RT, NaN where not possible
        group      calibration group of every spectrum ('' where not calibrated)
        groups     list of per-group summaries (n_peptides, degree, r, reversed)
        notes      human-readable notes
    """
    peptides = [str(p).strip().upper() for p in peptides]
    rt = np.asarray(rt, dtype=float)
    n = len(peptides)
    raw = np.full(n, np.nan)
    pred = np.full(n, np.nan)
    grp_out = np.array([""] * n, dtype=object)
    notes, summaries = [], []
    ok = np.array([check_peptide(p) is None for p in peptides])
    if ok.any():
        raw[ok] = model.predict_raw([p for p, o in zip(peptides, ok) if o])
    usable = ok & np.isfinite(rt)
    keys = np.array([sequence_key(p) for p in peptides], dtype=object)
    labels = np.array(["all"] * n if groups is None else [str(g) for g in groups], dtype=object)

    # one calibration per group; small groups are pooled
    sizes = {g: len(set(keys[usable & (labels == g)])) for g in set(labels[usable])}
    big = [g for g, k in sizes.items() if k >= min_peptides]
    small = [g for g, k in sizes.items() if k < min_peptides]
    plan = [(g, labels == g) for g in sorted(big)]
    if small:
        name = small[0] if len(small) == 1 and not big else "pooled (small files)"
        plan.append((name, np.isin(labels, small)))
    rng = np.random.default_rng(seed)
    for name, in_group in plan:
        sel = usable & in_group
        ukeys = np.array(sorted(set(keys[sel])), dtype=object)
        if len(ukeys) < min_peptides:
            notes.append(f"{name}: only {len(ukeys)} usable peptides (need {min_peptides}); no RT calibration")
            continue
        fold_of = dict(zip(rng.permutation(ukeys), np.arange(len(ukeys)) % folds))
        fold = np.array([fold_of.get(k, -1) for k in keys])
        per_key = pd.DataFrame({"key": keys[sel], "raw": raw[sel], "rt": rt[sel], "fold": fold[sel]}) \
            .groupby("key").agg(raw=("raw", "median"), rt=("rt", "median"), fold=("fold", "first"))
        degree = 1 if len(per_key) < 50 else 2
        for f in range(folds):
            train = per_key[per_key.fold != f]
            coef, _, _, _ = fit_without_early(train.raw.to_numpy(), train.rt.to_numpy(), degree, floor, early_cutoff)
            test = in_group & ok & (fold == f)
            pred[test] = np.polyval(coef, raw[test])
        grp_out[in_group & ok & (fold >= 0)] = name
        # spectra of this group whose peptide had no RT (so no fold): predict with the full fit
        coef_all, keep, _, _ = fit_without_early(per_key.raw.to_numpy(), per_key.rt.to_numpy(), degree, floor,
                                                 early_cutoff)
        rest = in_group & ok & (fold < 0)
        pred[rest] = np.polyval(coef_all, raw[rest])
        grp_out[rest] = name
        r = float(np.corrcoef(per_key.raw[keep], per_key.rt[keep])[0, 1])
        summaries.append(dict(group=name, n_peptides=len(per_key), n_spectra=int(sel.sum()), degree=degree,
                              r=r, reversed=bool(r < 0)))
    return dict(raw=raw, predicted=pred, group=grp_out, groups=summaries, notes=notes)


MIN_RETRAIN = 200


def model_from_standard(table, base, pep_col=None, rt_col=None, unit="min", name="", retrain=False, epochs=None,
                        log=print, early_cutoff="auto"):
    """RT model for a measured standard (a CSV/TSV/Excel table of trusted peptides with their RT, measured with the
    same LC method as the library). Default: calibrate the general model on the standard. retrain=True: fine-tune
    the model on the standard (TransferL recipe), only for standards of at least MIN_RETRAIN peptides; peptides the
    robust calibration marks as gross outliers are left out of the training.
    Returns (model, messages). Raises CalibrationError when the standard cannot be used."""
    import copy
    model = copy.deepcopy(base)
    data, msgs = calibrate_table(table, model, pep_col=pep_col, rt_col=rt_col, unit=unit, name=name,
                                 min_r=MIN_R_RETRAIN if retrain else MIN_R, early_cutoff=early_cutoff)
    if not retrain:
        return model, msgs
    if len(data) < MIN_RETRAIN:
        raise CalibrationError(f"The standard has {len(data)} usable peptides; retraining needs at least {MIN_RETRAIN} "
                               f"(a calibration works with fewer: switch retraining off).")
    from finetune_rt import finetune, auto_epochs
    use = data[data.used_in_fit]
    ep = epochs or auto_epochs(len(use))
    log(f"Retraining the RT model on {len(use):,} standard peptides ({ep} epochs) ...")
    tuned = finetune(base, list(use.peptide), use.measured_rt.to_numpy(), epochs=ep)
    cal_info = model.calibration_info
    tuned.calibration_info.update(name=name, unit=unit, n_used=len(use), source_table=cal_info.get("source_table", ""),
                                  early_cutoff=cal_info.get("early_cutoff"), n_early=cal_info.get("n_early", 0),
                                  training_keys=sorted({sequence_key(p) for p in data.peptide}))
    msgs.append(f"Retrained on {len(use):,} standard peptides ({len(data) - len(use)} outliers left out), {ep} epochs")
    return tuned, msgs


def early_elution_cutoff(rt, error, floor=1.0, max_share=0.25, min_bin=20):
    """RT below which peptides elute too early to be predicted (weak or no interaction with the column: they come
    off around the void time whatever their sequence, and the model predicts them later than measured).

    Walks from the earliest RT in bins of equal size and keeps going while a bin's median error (predicted -
    measured) is above the mid-gradient median error by at least `floor` (RT units) and clearly more than the
    bin's own uncertainty. The zone must start at the earliest RT and hold at most `max_share` of the spectra.
    Returns the cutoff (RT units) or None when there is no such zone."""
    rt, error = np.asarray(rt, dtype=float), np.asarray(error, dtype=float)
    ok = np.isfinite(rt) & np.isfinite(error)
    rt, error = rt[ok], error[ok]
    n = len(rt)
    if n < 3 * min_bin:
        return None
    order = np.argsort(rt, kind="stable")
    rt, error = rt[order], error[order]
    mid = (rt > np.percentile(rt, 25)) & (rt < np.percentile(rt, 90))
    ref = np.median(error[mid])
    sd = 1.4826 * np.median(np.abs(error[mid] - ref))
    size = max(min_bin, n // 50)
    last = None
    for start in range(0, int(n * max_share), size):
        b = error[start:start + size]
        shift = np.median(b) - ref
        if shift < floor or shift < 3 * 1.25 * sd / np.sqrt(len(b)):
            break
        last = start + len(b) - 1
    if last is None:
        return None
    nxt = rt[last + 1] if last + 1 < n else rt[last]
    return float((rt[last] + nxt) / 2)


def fit_without_early(x, y, degree, floor, cutoff="auto"):
    """Robust fit that leaves out the early-elution zone: fit, find the zone from the errors (or use the given
    cutoff; None = no zone), fit again without it. One detection only: where retention builds up gradually, a second
    round would keep pushing the cutoff up. Returns (coef, keep, early_mask, cutoff)."""
    coef, keep = robust_fit(x, y, degree)
    cut = early_elution_cutoff(y, np.polyval(coef, x) - y, floor=floor) if cutoff == "auto" else cutoff
    if not cut:
        return coef, keep, np.zeros(len(y), bool), None
    early = y < cut
    if (~early).sum() < degree + 3:
        return coef, keep, np.zeros(len(y), bool), None
    coef, keep_late = robust_fit(x[~early], y[~early], degree)
    keep = np.zeros(len(y), bool)
    keep[~early] = keep_late
    return coef, keep, early, float(cut)


def early_floor(unit):
    """Minimum error shift that counts as early elution: 1 min (60 s); no detection on the iRT scale."""
    return {"s": 60.0, "irt": None}.get(str(unit).lower(), 1.0)


def rt_outliers(error, groups, k=4.0, floor=0.5):
    """Flag RT errors far from the typical error of their calibration group: more than k robust standard
    deviations (1.4826 x MAD) from the group's median error, and at least `floor` (RT units) away."""
    error = np.asarray(error, dtype=float)
    flag = np.zeros(len(error), dtype=bool)
    for g in set(groups):
        sel = (np.asarray(groups) == g) & np.isfinite(error)
        if sel.sum() < 5:
            continue
        e = error[sel]
        center = np.median(e)
        scale = 1.4826 * np.median(np.abs(e - center))
        flag[sel] = np.abs(e - center) > max(k * scale, floor)
    return flag


def report_text(model, model_path, out):
    info, coef = model.calibration_info, model.calibration
    u = f" {info['unit']}" if info.get("unit") else ""
    lines = [
        f"RT calibration — {info['name']}",
        f"model            : {model_path}  ->  {out}",
        f"peptides         : {info['n_peptides']} unique, {info['n_used']} used, "
        f"{info['n_peptides'] - info['n_used']} outliers removed",
        f"fit              : {'straight line' if info['degree'] == 1 else 'polynomial degree %d' % info['degree']}"
        f"   coefficients {np.array2string(np.asarray(coef), precision=4)}",
        f"RT range covered : {info['rt_range'][0]:.2f} - {info['rt_range'][1]:.2f}{u}",
        f"correlation (r)  : {info['pearson_r']:.3f}",
    ]
    if "cv_median_abs_error" in info:
        lines.append(f"expected error   : median {info['cv_median_abs_error']:.2f}{u}, 95% of peptides within "
                     f"{info['cv_abs_error_95pct']:.2f}{u} (5-fold cross-validation)")
    lines.append("Predictions outside the RT range above are extrapolations; treat them as rough.")
    return "\n".join(lines)


def save_plot(data, model, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, y, keep = data.model_output.values, data.measured_rt.values, data.used_in_fit.values
    grid = np.linspace(x.min(), x.max(), 200)
    u = model.calibration_info.get("unit") or ""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].scatter(x[keep], y[keep], s=6, alpha=.5, color="#1f77b4", label="used")
    if (~keep).any():
        ax[0].scatter(x[~keep], y[~keep], s=10, color="#d62728", marker="x", label="outlier (removed)")
    ax[0].plot(grid, np.polyval(model.calibration, grid), color="black", lw=1.5, label="calibration")
    ax[0].set_xlabel("model output (relative 0-1)"), ax[0].set_ylabel(f"measured RT {u}".strip())
    ax[0].legend(frameon=False), ax[0].set_title("Calibration fit")
    cv = "cv_residual" in data
    ax[1].scatter(y, data.cv_residual if cv else data.residual, s=6, alpha=.5, color="#1f77b4")
    ax[1].axhline(0, color="black", lw=1)
    ax[1].set_xlabel(f"measured RT {u}".strip()), ax[1].set_ylabel("predicted − measured")
    ax[1].set_title("Error " + ("(cross-validated)" if cv else "(in-sample)"))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", help="CSV/TSV/XLSX with peptide sequences and measured RT")
    ap.add_argument("--model", default="rt_DIA.pt", help="uncalibrated model (default rt_DIA.pt)")
    ap.add_argument("--out", help="calibrated model to write (default <model>_calibrated.pt)")
    ap.add_argument("--pep-col"), ap.add_argument("--rt-col")
    ap.add_argument("--degree", type=int, choices=[1, 2, 3], help="1 = line, 2/3 = curve (default: auto)")
    ap.add_argument("--keep-modified", action="store_true",
                    help="also use rows flagged in a PTM/Modifications column (they are predicted as unmodified)")
    ap.add_argument("--force", action="store_true", help=f"save even if the fit looks wrong (|r| < {MIN_R})")
    ap.add_argument("--name", default="", help="label for this LC method, shown in the app")
    ap.add_argument("--unit", default="min", help="RT unit of your file: min, s, iRT ... (default min)")
    a = ap.parse_args()

    out = a.out or os.path.splitext(a.model)[0] + "_calibrated.pt"
    stem = os.path.splitext(out)[0]
    model = RTPredictor(a.model)
    try:
        data, msgs = calibrate_table(read_table(a.table), model, a.pep_col, a.rt_col, a.degree, a.keep_modified,
                                     a.name or os.path.basename(a.table), a.unit, a.force)
    except CalibrationError as e:
        for m in getattr(e, "msgs", []):
            print(m)
        sys.exit(f"Stopped: {e}")
    for m in msgs:
        print(m)
    model.calibration_info["source_table"] = os.path.basename(a.table)
    model.save(out)
    data.sort_values("measured_rt").to_csv(stem + "_calibration.csv", index=False)
    report = report_text(model, a.model, out)
    open(stem + "_calibration.txt", "w", encoding="utf-8").write(report + "\n")
    print("\n" + report)
    try:
        save_plot(data, model, stem + "_calibration.png")
        print(f"\nSaved {out}, {stem}_calibration.txt / .csv / .png")
    except ImportError:
        print(f"\nSaved {out}, {stem}_calibration.txt / .csv  (install matplotlib for the plot)")


if __name__ == "__main__":
    main()
