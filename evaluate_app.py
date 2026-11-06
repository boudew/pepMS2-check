"""Evaluate a measured-spectra file with the trained model, then explore the result - all in one page.

Replaces the three manual steps
    python format_measured.py data/*.tsv -o measured.csv --energy 30
    python compare_spectra.py measured.csv --model peptide_predictor.pt --out comparison --split split_assignment.tsv --unseen-only
    streamlit run metrics_explorer.py

Start:  streamlit run evaluate_app.py
Needs:  streamlit pandas numpy matplotlib torch, and in the SAME folder: format_measured.py, compare_spectra.py,
        metrics_explorer.py, peptide_model.py, peptide_predictor.pt (and split_assignment.tsv for the unseen-only test).
        Retention time (optional): rt_model.py, calibrate_rt.py and rt_DIA.pt in the same folder (+ finetune_rt.py to
        retrain on a standard). Preferred: upload a measured RT standard (CSV: peptide + RT, same LC method). Or a
        calibrated / retrained .pt, or let the library calibrate itself (cross-validated by sequence).

Every run gets its own folder under runs/ (input copy, measured.csv, comparison/ ...), so earlier results stay available.
Large files: Streamlit accepts uploads up to 200 MB by default; for bigger files start with
    streamlit run evaluate_app.py --server.maxUploadSize 2000
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'
SCRIPTS = ('format_measured.py', 'compare_spectra.py', 'metrics_explorer.py', 'peptide_model.py')
RT_FILES = ('rt_model.py', 'calibrate_rt.py', 'rt_DIA.pt')
RT_MODES = ('calibrate with a measured standard (CSV)', 'use a calibrated or retrained RT model (.pt)',
            'no standard: calibrate on this library (cross-validated)', 'off')
STD, PT, LIB, OFF = RT_MODES

st.set_page_config(page_title='Evaluate measured spectra', layout='wide')
sys.path.insert(0, str(HERE))
import importlib                                                       # noqa: E402
import inspect                                                         # noqa: E402
import metrics_explorer as mx                                          # noqa: E402
mx = importlib.reload(mx)                                              # pick up a replaced metrics_explorer.py without restarting
if not {'source_path', 'run_dir'} <= set(inspect.signature(mx.main).parameters) or not hasattr(mx, 'apply_style'):
    st.error('metrics_explorer.py in this folder is an old version. Replace it with the newest metrics_explorer.py '
             f'(folder: {HERE}) and reload this page.')
    st.stop()
mx.apply_style()


# ---------------------------------------------------------------- helpers
@st.cache_resource(show_spinner='Reading the model...')
def model_info(path, mtime):
    from peptide_model import load_checkpoint
    _, meta = load_checkpoint(path)
    return meta


def safe_name(text):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', text).strip('._') or 'run'


def run_step(title, cmd, box):
    """Run one command, stream its output into `box`, return (ok, text)."""
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    lines = []
    box.code(f'{title} ...', language=None)
    proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding='utf-8', errors='replace')
    for line in proc.stdout:
        lines.append(line.rstrip('\n'))
        box.code('\n'.join(lines[-25:]), language=None)
    proc.wait()
    return proc.returncode == 0, '\n'.join(lines)


@st.cache_data(show_spinner='Reading the standard...')
def read_standard(data, name):
    from calibrate_rt import read_table
    return read_table(data, name)


def standard_options():
    """Upload + column / unit choices for the RT standard. Returns a dict, or None while nothing usable is uploaded."""
    up = st.file_uploader('RT standard (CSV / TSV / Excel: peptide + RT)', type=['csv', 'tsv', 'txt', 'xlsx'],
                          key='ev_rt_std')
    if up is None:
        return None
    data = up.getvalue()
    try:
        from calibrate_rt import guess_columns, guess_unit, MIN_RETRAIN
        from rt_model import check_peptide
        df = read_standard(data, up.name)
    except Exception as e:
        st.error(f'Could not read {up.name}: {e}')
        return None
    cols = [str(c) for c in df.columns]
    g_pep, g_rt = guess_columns(df)
    tag = f'{up.name}_{len(data)}'
    c1, c2, c3 = st.columns(3)
    pep_col = c1.selectbox('Peptide column', cols, index=cols.index(str(g_pep)) if g_pep is not None else 0,
                           key=f'ev_std_pep_{tag}')
    rt_col = c2.selectbox('RT column', cols, index=cols.index(str(g_rt)) if g_rt is not None else min(1, len(cols) - 1),
                          key=f'ev_std_rt_{tag}')
    units = ['min', 's']
    g_unit = guess_unit(rt_col, df[df.columns[cols.index(rt_col)]])
    unit = c3.selectbox('RT unit', units, index=units.index(g_unit) if g_unit in units else 0,
                        key=f'ev_std_unit_{tag}_{rt_col}')
    pep = df[df.columns[cols.index(pep_col)]].astype(str).str.strip().str.upper()
    rt = pd.to_numeric(df[df.columns[cols.index(rt_col)]], errors='coerce')
    n = int(pep[rt.notna() & pep.map(lambda p: check_peptide(p) is None)].nunique())
    st.caption(f'{n:,} usable standard peptides (unmodified, up to 19 residues, with a numeric RT).')
    retrain = False
    if (HERE / 'finetune_rt.py').is_file():
        retrain = st.checkbox('Retrain the RT model on the standard (instead of only calibrating)', value=False,
                              key=f'ev_std_retrain_{tag}', disabled=n < MIN_RETRAIN,
                              help=f'Fine-tunes the model on the standard peptides (TransferL recipe). Clearly more '
                                   f'accurate when the standard is large (needs at least {MIN_RETRAIN} peptides; an iRT '
                                   f'kit is only good for a calibration). Takes a minute or more; faster with a GPU.')
    return dict(data=data, name=up.name, pep_col=str(df.columns[cols.index(pep_col)]),
                rt_col=str(df.columns[cols.index(rt_col)]), unit=unit, retrain=retrain)


def previous_runs():
    if not RUNS.is_dir():
        return []
    found = [p for p in RUNS.iterdir() if (p / 'comparison' / 'metrics.csv').is_file()]
    return sorted(found, key=lambda p: (p / 'comparison' / 'metrics.csv').stat().st_mtime, reverse=True)


# ---------------------------------------------------------------- page
st.title('Evaluate measured spectra')
st.caption('Upload a measured spectra file, choose the HCD energy and compare it with the model\'s predictions. '
           'The result opens below.')

missing = [f for f in SCRIPTS if not (HERE / f).is_file()]
if missing:
    st.error('These files must be in the same folder as evaluate_app.py: ' + ', '.join(missing))
    st.stop()

has_result = bool(st.session_state.get('ev_result'))
with st.expander('1. Run an evaluation', expanded=not has_result):
    left, right = st.columns([3, 2])
    with left:
        ups = st.file_uploader('Measured spectra (.tsv / .txt / .tab, several files allowed)', type=['tsv', 'txt', 'tab'],
                               accept_multiple_files=True, key='ev_files')
    with right:
        model_path = st.session_state.get('ev_model', str(HERE / 'peptide_predictor.pt'))      # editable under 'More options'
        meta = None
        if os.path.isfile(model_path):
            try:
                meta = model_info(model_path, os.path.getmtime(model_path))
            except Exception as e:                                     # incompatible checkpoint / missing torch
                st.error(f'Could not load the model: {e}')
        else:
            st.error(f'Model file not found: {model_path}. Tick "More options" to change the path.')
        es = (meta or {}).get('energies') or []
        energy, energy_col = None, ''
        if es:
            how = st.radio('Collision energy (HCD, NCE)', ['one value for the whole file', 'from a column in the file'],
                           key='ev_how')
            if how.startswith('one'):
                default = 30.0 if min(es) <= 30 <= max(es) else float(es[len(es) // 2])
                energy = st.number_input('NCE', float(min(es)), float(max(es)), default, 1.0, format='%g', key='ev_energy')
                st.caption('Model trained at NCE ' + ', '.join(f'{e:g}' for e in es) + '. Values in between are interpolated.')
            else:
                energy_col = st.text_input('Name of the energy column', 'NCE', key='ev_ecol')
        elif meta is not None:
            st.caption('This model has no collision-energy input.')
        split_path = HERE / 'split_assignment.tsv'
        use_split = False
        if split_path.is_file():                                       # optional: without the file this option is simply not shown
            use_split = st.checkbox('Compare only peptides the model has not seen', value=True, key='ev_unseen',
                                    help='Uses split_assignment.tsv from the training notebook to leave out peptides that were in the training data.')
        rt_mode, rt_upload, rt_std, rt_early_arg = OFF, None, None, 'auto'
        rt_missing = [f for f in RT_FILES if not (HERE / f).is_file()]
        if rt_missing:
            st.caption('Retention time is off: ' + ', '.join(rt_missing) + ' not found next to this app.')
        else:
            rt_mode = st.radio('Retention time', RT_MODES, key='ev_rt_mode',
                               help='Measured standard (preferred): a table of trusted peptides with their RT, measured '
                                    'with the same LC method (e.g. an iRT kit or your own standard mix). The RT model is '
                                    'calibrated on it, so every library peptide is a fair test. '
                                    'Calibrated or retrained model: a .pt from the pepMS2 app, calibrate_rt.py, '
                                    'finetune_rt.py or an earlier run, used as it is. '
                                    'No standard: every spectrum\'s RT comes from a calibration fitted on the other '
                                    'peptides of the library (5-fold, split by sequence with I/L merged).')
            if rt_mode == STD:
                rt_std = standard_options()
            elif rt_mode == PT:
                rt_upload = st.file_uploader('Calibrated or retrained RT model (.pt)', type=['pt'], key='ev_rt_pt')
                if rt_upload is not None:
                    try:
                        from rt_model import RTPredictor
                        info = RTPredictor(rt_upload.getvalue())
                        if info.is_calibrated:
                            ci = info.calibration_info
                            st.caption(f"{'Retrained' if ci.get('finetuned') else 'Calibrated'}: "
                                       f"{ci.get('name') or rt_upload.name}"
                                       + (f", {ci['n_used']} peptides" if 'n_used' in ci else '')
                                       + (f", RT {ci['rt_range'][0]:.1f}-{ci['rt_range'][1]:.1f} {ci.get('unit') or ''}"
                                          if 'rt_range' in ci else ''))
                        else:
                            st.caption('This RT model has no calibration; it will be calibrated on this library instead.')
                    except Exception as e:
                        st.error(f'{rt_upload.name}: {e}')
                        rt_upload = None
            if rt_mode != OFF:
                e1, e2 = st.columns([3, 2])
                early_how = e1.selectbox('Early-eluting peptides', ['detect automatically', 'set a cutoff', 'no early zone'],
                                         key='ev_early',
                                         help='Peptides that hardly interact with the column elute around the void time '
                                              'whatever their sequence, so their RT cannot be predicted. They are marked '
                                              'as early-eluting instead of RT outliers, and left out of the RT statistics '
                                              'and the calibration. Automatic: the zone is found from the start of the '
                                              'gradient while the typical error stays clearly shifted.')
                if early_how == 'set a cutoff':
                    early_cut = e2.number_input('Cutoff (min)', 0.0, 300.0, 2.0, 0.25, key='ev_early_cut')
                    rt_early_arg = f'{early_cut:g}'
                else:
                    rt_early_arg = 'auto' if early_how.startswith('detect') else 'off'

    force_z, min_peaks, carb, mods, tol, examples = 'from the file', 3, False, 'skip', 0.0, 12   # defaults when the options are closed
    if st.checkbox('More options', key='ev_more'):                    # (expanders cannot be nested)
        c1, c2 = st.columns(2)
        zs = (meta or {}).get('charges') or []
        force_z = c1.selectbox('Precursor charge', ['from the file'] + [f'{z}+ for every spectrum' for z in zs], key='ev_z',
                               disabled=not zs)
        min_peaks = c2.number_input('Min. y/b peaks per spectrum', 1, 20, 3, key='ev_minpeaks')
        c4, c5, c6 = st.columns(3)
        carb = c4.checkbox('Cysteines carbamidomethylated', False, key='ev_carb')
        mods = c5.selectbox('Modified peptides', ['skip', 'ignore'], key='ev_mods',
                            help='skip: drop peptides with modifications; ignore: keep them and treat the sequence as unmodified')
        tol = c6.number_input('Mass tolerance for y/b peaks (Da, 0 = no check)', 0.0, 1.0, 0.0, 0.01, format='%.2f', key='ev_tol')
        examples = st.number_input('Mirror plots to draw (examples.pdf, 0 = none)', 0, 100, 12, key='ev_examples')
        st.text_input('Model file', str(HERE / 'peptide_predictor.pt'), key='ev_model')

    name_default = safe_name(Path(ups[0].name).stem) if ups else ''
    run_label = st.text_input('Name of this run', name_default, key='ev_name')
    rt_waiting = (rt_mode == PT and rt_upload is None) or (rt_mode == STD and rt_std is None)
    ready = bool(meta) and bool(ups) and not rt_waiting
    if rt_waiting and ups:
        st.caption('Upload the RT standard (or the RT model), or choose another retention-time option.')
    go = st.button('Run evaluation', type='primary', disabled=not ready, key='ev_go')

    if go:
        run_dir = RUNS / f'{time.strftime("%Y%m%d-%H%M%S")}_{safe_name(run_label or "run")}'
        in_dir = run_dir / 'input'
        in_dir.mkdir(parents=True)
        inputs = []
        for u in ups:                                                  # keep a copy of what was evaluated
            p = in_dir / safe_name(u.name)
            p.write_bytes(u.getbuffer())
            inputs.append(str(p))
        n_frag = str(meta['n_frag'])                                   # always matches the model
        fm = [sys.executable, 'format_measured.py', *inputs, '-o', str(run_dir / 'measured.csv'), '--n-frag', n_frag,
              '--min-peaks', str(int(min_peaks)), '--mods', mods, '--mass-tol', str(tol)]
        if energy is not None:
            fm += ['--energy', f'{energy:g}']
        elif energy_col:
            fm += ['--energy-col', energy_col]
        if carb:
            fm += ['--carbamidomethyl']
        cs = [sys.executable, 'compare_spectra.py', str(run_dir / 'measured.csv'), '--model', model_path,
              '--out', str(run_dir / 'comparison'), '--examples', str(int(examples))]
        if use_split and split_path.is_file():
            cs += ['--split', str(split_path), '--unseen-only']
        if force_z != 'from the file':
            cs += ['--charge', force_z.split('+')[0]]
        if rt_mode != OFF:
            cs += ['--rt-early-cutoff', rt_early_arg]
        if rt_mode == OFF:
            cs += ['--no-rt']
        elif rt_mode == STD and rt_std is not None:
            std_file = in_dir / ('rt_standard_' + safe_name(rt_std['name']))
            std_file.write_bytes(rt_std['data'])
            cs += ['--rt-standard', str(std_file), '--rt-standard-unit', rt_std['unit'],
                   '--rt-standard-pep-col', rt_std['pep_col'], '--rt-standard-rt-col', rt_std['rt_col'],
                   '--rt-standard-name', Path(rt_std['name']).stem]
            if rt_std['retrain']:
                cs += ['--rt-retrain-standard']
        elif rt_mode == PT and rt_upload is not None:
            rt_file = in_dir / safe_name(rt_upload.name)
            rt_file.write_bytes(rt_upload.getvalue())
            cs += ['--rt-model', str(rt_file)]
        if carb:
            st.caption('Note: compare_spectra.py draws mirror plots with unmodified cysteine masses.')

        with st.status('Running...', expanded=True) as status:
            box1 = st.empty()
            ok, out1 = run_step('Step 1/2: converting the measured spectra', fm, box1)
            ok2, out2 = False, ''
            if ok:
                box2 = st.empty()
                ok2, out2 = run_step('Step 2/2: predicting and comparing', cs, box2)
            status.update(label='Done' if ok and ok2 else 'Failed', state='complete' if ok and ok2 else 'error', expanded=not (ok and ok2))
        if ok and ok2:
            st.session_state['ev_result'] = str(run_dir)
            st.rerun()
        else:
            st.error('Step 1 (format_measured.py) failed. The output above says why.' if not ok else
                     'Step 2 (compare_spectra.py) failed. The output above says why.')

    runs = previous_runs()
    if runs:
        st.divider()
        names = [p.name for p in runs]
        cur = Path(st.session_state['ev_result']).name if has_result and Path(st.session_state['ev_result']).name in names else names[0]
        pick = st.selectbox('or open an earlier run', names, index=names.index(cur), key='ev_prev')
        if st.button('Open this run', key='ev_open'):
            st.session_state['ev_result'] = str(RUNS / pick)
            st.rerun()

# ---------------------------------------------------------------- result
result = st.session_state.get('ev_result')
if result and (Path(result) / 'comparison' / 'metrics.csv').is_file():
    run_dir = Path(result)
    comp = run_dir / 'comparison'
    mx.heading(f'Result: {run_dir.name}')
    with st.expander('Run reports (conversion and comparison)'):
        for label, path in (('Conversion', run_dir / 'measured_report.txt'), ('Comparison', comp / 'summary.txt')):
            if path.is_file():
                st.markdown(f'**{label}**')
                st.code(path.read_text(encoding='utf-8', errors='replace'), language=None)
        if (comp / 'summary.png').is_file():
            st.image(str(comp / 'summary.png'))
        files = [('metrics.csv', 'text/csv'), ('predicted.csv', 'text/csv'), ('examples.pdf', 'application/pdf'),
                 ('rt_standard.pt', 'application/octet-stream'), ('rt_standard_retrained.pt', 'application/octet-stream'),
                 ('rt_calibrated.pt', 'application/octet-stream')]
        files = [f for f in files if (comp / f[0]).is_file()]
        for col, (fname, mime) in zip(st.columns(max(len(files), 1)), files):
            if (comp / fname).is_file():
                col.download_button(f'Download {fname}', (comp / fname).read_bytes(), file_name=f'{run_dir.name}_{fname}',
                                    mime=mime, key=f'dl_{fname}')
        for fname, what in (('rt_standard.pt', 'the RT model calibrated on your standard'),
                            ('rt_standard_retrained.pt', 'the RT model retrained on your standard'),
                            ('rt_calibrated.pt', 'the RT calibration of this library (fitted on all its peptides)')):
            if (comp / fname).is_file():
                st.caption(f'{fname}: {what}. Upload it in the pepMS2 app, or use it for another run measured with the '
                           'same LC method.')
    mx.main(source_path=comp / 'metrics.csv', run_dir=run_dir)
elif result:
    st.warning(f'The run folder {result} has no comparison/metrics.csv any more.')
