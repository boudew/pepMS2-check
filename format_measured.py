"""Convert measured MS/MS peak lists into the training format of the fragment-intensity model.

One row per spectrum:   "", Peptide, y1..yN, b1..bN     (N = 30, like the training file; --n-frag must match the model)
Intensities are normalised so that all y and b ions of a spectrum sum to 1 (as in the training data).
Only plain, singly charged y and b ions are used; neutral losses (b5-NH3 ...) and other ion types are ignored.
Peptides up to 30 residues are accepted (--max-len). The table holds y1..yN / b1..bN (N = --n-frag), so for peptides
longer than N+1 residues the ions above number N are left out (they are counted in the report).

Collision energy: the model needs the HCD energy (NCE) of every spectrum. Use --energy 28 when the whole library was
measured at one energy, or --energy-col NAME when the input file has a column with it. It is stored as the 'energy'
column of measured_meta.csv, which compare_spectra.py picks up. The precursor charge comes from the z / charge column.

Usage
    python format_measured.py sample1.tsv sample2.tsv -o measured.csv
    python format_measured.py "data/*.tsv" -o measured.csv         (wildcards also work on Windows)
    python format_measured.py data_folder -o measured.csv          (every .tsv / .txt / .tab file in the folder)
    python format_measured.py lib.tsv -o measured.csv --energy 28  (library measured at NCE 28)

Writes
    measured.csv          training-format table, the input for compare_spectra.py
    measured_meta.csv     same row numbers: source file, precursor m/z, charge, RT and QC numbers
    measured_report.txt   what was kept, what was skipped and why

Needs: numpy, pandas
"""
import argparse
import glob
import os
import re
import sys
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd

PROTON, WATER = 1.007276, 18.010565
RESIDUE = {  # monoisotopic residue masses (cysteine unmodified, like the notebook default)
    'G': 57.02146, 'A': 71.03711, 'S': 87.03203, 'P': 97.05276, 'V': 99.06841, 'T': 101.04768,
    'C': 103.00919, 'L': 113.08406, 'I': 113.08406, 'N': 114.04293, 'D': 115.02694, 'Q': 128.05858,
    'K': 128.09496, 'E': 129.04259, 'M': 131.04049, 'H': 137.05891, 'F': 147.06841, 'R': 156.10111,
    'Y': 163.06333, 'W': 186.07931}
CYS_CARBAMIDOMETHYL = 160.03065
ION_RE = re.compile(r'([yb])(\d+)')          # plain y3 / b5 only; "b5-NH3", "y3++" etc. do not match


# ---------------------------------------------------------------- helpers
def find_col(columns, *names, required=True):
    """Case-insensitive column lookup: exact name first, then prefix match for names longer than 3 characters."""
    low = {str(c).strip().lower(): c for c in columns}
    for n in names:
        if n in low:
            return low[n]
    for n in names:
        if len(n) > 3:
            for k, c in low.items():
                if k.startswith(n):
                    return c
    if required:
        sys.exit(f'Could not find a column like {names[0]!r}. Columns in the file: {list(columns)}')
    return None


def column(chunk, name):
    return chunk[name].tolist() if name else [''] * len(chunk)


def expand_inputs(items):
    files = []
    for item in items:
        if os.path.isdir(item):
            for ext in ('*.tsv', '*.txt', '*.tab'):
                files += sorted(glob.glob(os.path.join(item, ext)))
        elif any(ch in item for ch in '*?['):
            files += sorted(glob.glob(item))
        else:
            files.append(item)
    out, seen = [], set()
    for f in files:
        key = os.path.abspath(f)
        if key not in seen:
            seen.add(key)
            out.append(f)
    missing = [f for f in out if not os.path.isfile(f)]
    if missing:
        sys.exit('File not found: ' + ', '.join(missing))
    if not out:
        sys.exit('No input files found.')
    return out


@lru_cache(maxsize=200_000)
def ion_masses(seq):
    r = np.array([RESIDUE[a] for a in seq])
    b = np.cumsum(r)[:-1] + PROTON                       # b1..b(L-1), singly charged
    y = np.cumsum(r[::-1])[:-1] + WATER + PROTON         # y1..y(L-1), singly charged
    return b, y


# ---------------------------------------------------------------- one spectrum
def convert_spectrum(seq_raw, mods_raw, peaks_raw, a, other_ann, peak_stats):
    """Returns (normalised vector, info dict) or (None, reason)."""
    seq = seq_raw.strip().upper()
    if not seq:
        return None, 'empty sequence'
    if not all(c in RESIDUE for c in seq):
        return None, 'sequence has unsupported characters'
    if a.mods == 'skip' and mods_raw.strip():
        return None, 'has modifications'
    L = len(seq)
    if L < 2:
        return None, 'peptide shorter than 2 residues'
    if L > a.max_len:
        return None, f'peptide longer than {a.max_len} residues'

    b_mz, y_mz = ion_masses(seq)
    vec = np.zeros(2 * a.n_frag)
    listed = 0.0
    n_other = n_mis = n_beyond = 0
    for item in peaks_raw.split(';'):
        parts = item.strip().split(':')
        if len(parts) < 3:
            continue
        try:
            mz, inten = float(parts[0]), float(parts[1])
        except ValueError:
            peak_stats['malformed peaks'] += 1
            continue
        listed += inten
        ann = parts[2].strip()
        m = ION_RE.fullmatch(ann)
        if not m:
            other_ann[re.sub(r'^([A-Za-z]+)\d+', r'\1n', ann)] += 1
            n_other += 1
            continue
        t, k = m.group(1), int(m.group(2))
        if k < 1 or k > L - 1:
            peak_stats['peaks with impossible ion number'] += 1
            continue
        if k > a.n_frag:                       # long peptide: this ion has no column in the table
            n_beyond += 1
            continue
        if a.mass_tol > 0 and abs(mz - (y_mz if t == 'y' else b_mz)[k - 1]) > a.mass_tol:
            n_mis += 1
            continue
        idx = k - 1 if t == 'y' else a.n_frag + k - 1
        vec[idx] = max(vec[idx], inten)

    n_ions = int((vec > 0).sum())
    total = vec.sum()
    if n_ions == 0 or n_ions < a.min_peaks or total <= 0:
        return None, f'fewer than {max(a.min_peaks, 1)} usable y/b peaks'
    info = dict(n_ions=n_ions, intensity_kept_fraction=total / listed if listed > 0 else np.nan,
                n_other_peaks=n_other, n_mass_mismatch=n_mis, n_beyond_limit=n_beyond)
    return vec / total, info


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+', help='TSV files, wildcards or folders')
    ap.add_argument('-o', '--output', default='measured.csv', help='output table (default measured.csv)')
    ap.add_argument('--n-frag', type=int, default=30, help='ions per type, must match the model (default 30)')
    ap.add_argument('--max-len', type=int, default=30,
                    help='longest peptide to keep (default 30). The table holds y1..yN / b1..bN (N = --n-frag), so for '
                         'peptides longer than N+1 residues the ions above number N are left out')
    ap.add_argument('--mods', choices=['skip', 'ignore'], default='skip',
                    help='skip: drop peptides with a non-empty Modifications column (default); '
                         'ignore: keep them and treat the sequence as unmodified')
    ap.add_argument('--mass-tol', type=float, default=0.05,
                    help='drop annotated peaks whose m/z differs from the theoretical singly charged y/b m/z '
                         'by more than this many Da (default 0.05, 0 = no check)')
    ap.add_argument('--min-peaks', type=int, default=2, help='minimum number of usable y/b peaks per spectrum (default 2)')
    ap.add_argument('--energy', type=float, default=None,
                    help='collision energy (NCE) of ALL spectra in the input, e.g. 28; stored as the energy column of the metadata')
    ap.add_argument('--energy-col', default=None,
                    help='name of a column in the input file that holds the collision energy ("28", "HCD28" and "NCE=28" all work)')
    ap.add_argument('--carbamidomethyl', action='store_true',
                    help='cysteines are carbamidomethylated (+57.02146) for the m/z check; default: unmodified')
    ap.add_argument('--activation', default=None,
                    help='only keep spectra whose Activation Mode contains this text, e.g. CID (case-insensitive)')
    ap.add_argument('--chunk', type=int, default=100_000, help=argparse.SUPPRESS)
    a = ap.parse_args()

    files = expand_inputs(a.inputs)
    if a.carbamidomethyl:
        RESIDUE['C'] = CYS_CARBAMIDOMETHYL

    peptides, blocks, meta = [], [], []
    skipped, other_ann, peak_stats, modes = Counter(), Counter(), Counter(), Counter()
    n_read = 0
    for path in files:
        print(f'reading {path}')
        reader = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, encoding='utf-8-sig',
                             chunksize=a.chunk, on_bad_lines='warn')
        cols, row_no = None, 0
        for chunk in reader:
            if cols is None:
                cols = dict(
                    seq=find_col(chunk.columns, 'sequence (backbone)', 'sequence', 'peptide'),
                    peaks=find_col(chunk.columns, 'peaks list', 'peaks'),
                    mods=find_col(chunk.columns, 'modifications', 'modification', required=False),
                    mz=find_col(chunk.columns, 'm/z', 'precursor m/z', required=False),
                    z=find_col(chunk.columns, 'z', 'charge', required=False),
                    rt=find_col(chunk.columns, 'rt (seconds)', 'rt', required=False),
                    act=find_col(chunk.columns, 'activation mode', 'activation', required=False),
                    engine=find_col(chunk.columns, 'engine', required=False),
                    energy=(find_col(chunk.columns, a.energy_col.strip().lower(), required=True) if a.energy_col else None))
            seqs, peaks, mods, mzs, zs, rts, acts, engines, ens = (
                column(chunk, cols[k]) for k in ('seq', 'peaks', 'mods', 'mz', 'z', 'rt', 'act', 'engine', 'energy'))
            block = []
            for i in range(len(chunk)):
                row_no += 1
                n_read += 1
                modes[acts[i].strip() or '(none)'] += 1
                if a.activation and a.activation.lower() not in acts[i].lower():
                    skipped['activation mode filtered out'] += 1
                    continue
                vec, info = convert_spectrum(seqs[i], mods[i], peaks[i], a, other_ann, peak_stats)
                if vec is None:
                    skipped[info] += 1
                    continue
                peptides.append(seqs[i].strip().upper())
                block.append(vec)
                meta.append(dict(source_file=os.path.basename(path), source_row=row_no, precursor_mz=mzs[i],
                                 charge=zs[i], rt_seconds=rts[i], activation_mode=acts[i], engine=engines[i],
                                 energy=(a.energy if a.energy is not None else ens[i]), **info))
            if block:
                blocks.append(np.vstack(block))

    lines = []

    def say(s=''):
        print(s)
        lines.append(s)

    stem = os.path.splitext(a.output)[0]
    n_ok = len(peptides)
    say('')
    say(f'Spectra read:  {n_read:,} from {len(files)} file(s)')
    if n_ok == 0:
        say('Nothing usable was found.')
    else:
        say(f'Written:       {n_ok:,} ({100 * n_ok / n_read:.1f}%)  ->  {a.output}')
    if skipped:
        say('Skipped:')
        for reason, c in skipped.most_common():
            say(f'  {c:>10,}  {reason}')
    if not n_ok:
        with open(stem + '_report.txt', 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(lines) + '\n')
        sys.exit(1)

    names = [f'y{i}' for i in range(1, a.n_frag + 1)] + [f'b{i}' for i in range(1, a.n_frag + 1)]
    out = pd.DataFrame(np.vstack(blocks), columns=names)
    out.insert(0, 'Peptide', peptides)
    out.index = np.arange(1, n_ok + 1)
    meta = pd.DataFrame(meta, index=out.index)
    for c in ('precursor_mz', 'charge', 'rt_seconds'):
        meta[c] = pd.to_numeric(meta[c], errors='coerce')
    if a.energy is not None or a.energy_col:
        meta['energy'] = pd.to_numeric(meta['energy'].astype(str).str.extract(r'(\d+(?:\.\d+)?)')[0], errors='coerce')
    else:
        meta = meta.drop(columns='energy')

    out_dir = os.path.dirname(os.path.abspath(a.output))
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(a.output, index_label='', float_format='%.10g')
    meta.to_csv(stem + '_meta.csv', index_label='', float_format='%.6g')

    n_used = int(meta['n_ions'].sum())
    n_mis = int(meta['n_mass_mismatch'].sum())
    say('')
    say('Peak annotations')
    say(f'  plain y/b peaks used:                {n_used:,}')
    say(f'  other annotations ignored:           {sum(other_ann.values()):,}'
        + ('   (' + ', '.join(f'{k}: {v:,}' for k, v in other_ann.most_common(6)) + ')' if other_ann else ''))
    if a.mass_tol > 0:
        say(f'  dropped, m/z not a 1+ y/b of the sequence (> {a.mass_tol} Da): {n_mis:,}')
    n_beyond = int(meta['n_beyond_limit'].sum())
    if n_beyond:
        say(f'  ignored, ion number above {a.n_frag} (peptides longer than {a.n_frag + 1}): {n_beyond:,}')
    for k, v in peak_stats.items():
        say(f'  {k}: {v:,}')
    if a.mass_tol > 0 and n_mis > 0.05 * (n_used + n_mis):
        say('  NOTE: many annotated peaks do not match singly charged y/b masses. Check cysteine alkylation '
            '(--carbamidomethyl), higher fragment charge states, or raise --mass-tol.')
    say('')
    say('Written spectra')
    say(f'  y/b peaks per spectrum (median):        {meta["n_ions"].median():.0f}')
    say(f'  share of listed intensity that is plain y/b (median): {100 * meta["intensity_kept_fraction"].median():.0f}%')
    ln = out['Peptide'].str.len()
    say(f'  peptide length min / median / max:      {ln.min()} / {ln.median():.0f} / {ln.max()}')
    n_long = int((ln > a.n_frag + 1).sum())
    if n_long:
        say(f'  peptides longer than {a.n_frag + 1} residues: {n_long:,} (only y1-y{a.n_frag} / b1-b{a.n_frag} stored)')
    if 'energy' in meta:
        say('  collision energy (NCE): ' + ', '.join(f'{e:g}: {c:,}' for e, c in meta['energy'].value_counts().sort_index().items())
            + (f'   ({int(meta["energy"].isna().sum()):,} without a readable energy)' if meta['energy'].isna().any() else ''))
    else:
        say('  NOTE: no collision energy stored. A model with an energy input needs --energy N or --energy-col NAME here, '
            'or --energy N when running compare_spectra.py.')
    if meta['charge'].notna().any():
        say('  precursor charge: ' + ', '.join(f'{int(z)}+: {c:,}' for z, c in meta['charge'].value_counts().sort_index().items()))
    say('  activation modes (all spectra read): ' + ', '.join(f'{k}: {v:,}' for k, v in modes.most_common(5)))
    say('')
    say(f'training-format table: {a.output}')
    say(f'metadata:              {stem}_meta.csv')
    with open(stem + '_report.txt', 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
