"""Evaluate a measured-spectra file with the trained model, then explore the result - all in one page.

Replaces the three manual steps
    python format_measured.py data/*.tsv -o measured.csv --energy 30
    python compare_spectra.py measured.csv --model peptide_predictor.pt --out comparison --split split_assignment.tsv --unseen-only
    streamlit run metrics_explorer.py

Start:  streamlit run evaluate_app.py
Needs:  streamlit pandas numpy matplotlib torch, and in the SAME folder: format_measured.py, compare_spectra.py,
        metrics_explorer.py, peptide_model.py, peptide_predictor.pt (and split_assignment.tsv for the unseen-only test).

Every run gets its own folder under runs/ (input copy, measured.csv, comparison/ ...), so earlier results stay available.
Large files: Streamlit accepts uploads up to 200 MB by default. For bigger files give the path of the file or folder
on this computer instead, or start with:  streamlit run evaluate_app.py --server.maxUploadSize 2000
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'
SCRIPTS = ('format_measured.py', 'compare_spectra.py', 'metrics_explorer.py', 'peptide_model.py')

st.set_page_config(page_title='Evaluate measured spectra', layout='wide')
sys.path.insert(0, str(HERE))
import importlib                                                       # noqa: E402
import inspect                                                         # noqa: E402
import metrics_explorer as mx                                          # noqa: E402
mx = importlib.reload(mx)                                              # pick up a replaced metrics_explorer.py without restarting
mx.apply_style()
if 'source_path' not in inspect.signature(mx.main).parameters:
    st.error('metrics_explorer.py next to this app is the old version. Copy the new metrics_explorer.py into the same '
             f'folder ({HERE}) and reload this page.')
    st.stop()


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
        local = st.text_input('or a file / folder / wildcard on this computer', '', key='ev_local',
                              help='For very large files. Example: data/*.tsv or C:\\data\\library.tsv')
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
                default = 28.0 if min(es) <= 28 <= max(es) else float(es[len(es) // 2])
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

    force_z, activation, min_peaks, carb, mods, tol, examples = 'from the file', '', 2, False, 'skip', 0.05, 12   # defaults when the options are closed
    if st.checkbox('More options', key='ev_more'):                    # (expanders cannot be nested)
        c1, c2, c3 = st.columns(3)
        zs = (meta or {}).get('charges') or []
        force_z = c1.selectbox('Precursor charge', ['from the file'] + [f'{z}+ for every spectrum' for z in zs], key='ev_z',
                               disabled=not zs)
        activation = c2.text_input('Only this activation mode (e.g. HCD)', '', key='ev_act')
        min_peaks = c3.number_input('Min. y/b peaks per spectrum', 1, 20, 2, key='ev_minpeaks')
        c4, c5, c6 = st.columns(3)
        carb = c4.checkbox('Cysteines carbamidomethylated', False, key='ev_carb')
        mods = c5.selectbox('Modified peptides', ['skip', 'ignore'], key='ev_mods',
                            help='skip: drop peptides with modifications; ignore: keep them and treat the sequence as unmodified')
        tol = c6.number_input('Mass tolerance for y/b peaks (Da, 0 = no check)', 0.0, 1.0, 0.05, 0.01, format='%.2f', key='ev_tol')
        examples = st.number_input('Mirror plots to draw (examples.pdf, 0 = none)', 0, 100, 12, key='ev_examples')
        st.text_input('Model file', str(HERE / 'peptide_predictor.pt'), key='ev_model')

    name_default = safe_name(Path(ups[0].name).stem) if ups else (safe_name(Path(local).stem) if local.strip() else '')
    run_label = st.text_input('Name of this run', name_default, key='ev_name')
    ready = bool(meta) and bool(ups or local.strip())
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
        if local.strip():
            inputs.append(local.strip())
        n_frag = str(meta['n_frag'])                                   # always matches the model
        fm = [sys.executable, 'format_measured.py', *inputs, '-o', str(run_dir / 'measured.csv'), '--n-frag', n_frag,
              '--min-peaks', str(int(min_peaks)), '--mods', mods, '--mass-tol', str(tol)]
        if energy is not None:
            fm += ['--energy', f'{energy:g}']
        elif energy_col:
            fm += ['--energy-col', energy_col]
        if activation.strip():
            fm += ['--activation', activation.strip()]
        if carb:
            fm += ['--carbamidomethyl']
        cs = [sys.executable, 'compare_spectra.py', str(run_dir / 'measured.csv'), '--model', model_path,
              '--out', str(run_dir / 'comparison'), '--examples', str(int(examples))]
        if use_split and split_path.is_file():
            cs += ['--split', str(split_path), '--unseen-only']
        if force_z != 'from the file':
            cs += ['--charge', force_z.split('+')[0]]
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
        d1, d2, d3 = st.columns(3)
        for col, fname, mime in ((d1, 'metrics.csv', 'text/csv'), (d2, 'predicted.csv', 'text/csv'),
                                 (d3, 'examples.pdf', 'application/pdf')):
            if (comp / fname).is_file():
                col.download_button(f'Download {fname}', (comp / fname).read_bytes(), file_name=f'{run_dir.name}_{fname}',
                                    mime=mime, key=f'dl_{fname}')
    mx.main(source_path=comp / 'metrics.csv')
elif result:
    st.warning(f'The run folder {result} has no comparison/metrics.csv any more.')
