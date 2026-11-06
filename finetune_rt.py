"""
finetune_rt.py — retrain (fine-tune) the RT model on your own peptides, as in TransferL's transfer step.

TransferL recipe, ported to PyTorch:
  * start from a general model (rt_DIA.pt)
  * freeze the first BiLSTM layer, train the rest (second BiLSTM + dense layers)
  * dropout 0.2 before the output layer, Adam, learning rate 2e-4, MSE loss
  * target = your RT, min-max scaled to 0-1 on the training peptides
One target per peptide sequence (median RT over repeated identifications), so peptides measured many times do
not dominate. If your LC elutes in the reverse order of reversed-phase (e.g. HILIC), the target is flipped so the
network keeps its learned orientation; the saved mapping turns it back.

The result is a normal RT model file (same format as rt_DIA.pt) whose calibration maps the output straight to your
RT units, so it can be used in the pepMS2 app and in compare_spectra.py --rt-model.

    python finetune_rt.py my_ids.csv --model rt_DIA.pt --out rt_myLC.pt --name "C18 40 min" --unit min

For an honest error estimate use compare_spectra.py (RT mode 'retrain'), which retrains once per
cross-validation part so no peptide is trained on itself.
"""
import argparse
import copy
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from rt_model import RTPredictor, check_peptide, encode

EPOCHS, LR, BATCH, DROPOUT = 60, 2e-4, 16, 0.2


def auto_epochs(n_peptides):
    """60 epochs for small sets, fewer for large ones (about constant training time). Tested with 5-fold CV:
    HILIC library (375 peptides) 20 ep 0.25 / 60 ep 0.21 / 150 ep 0.21 min; pea C18 (3,499) 20 ep 0.44 / 60 ep 0.41 min."""
    return int(max(20, min(60, 120_000 // max(n_peptides, 1))))


def _forward_train(net, x, dropout):
    """RTNet forward with dropout before the output layer (only while training)."""
    h, _ = net.lstm1(x)
    _, (h, _) = net.lstm2(h)
    z = net.d2(net.d1(torch.cat([h[0], h[1]], dim=1)))
    return net.out(F.dropout(z, dropout, training=True)).squeeze(-1)


def finetune(base, peptides, rt, epochs=EPOCHS, lr=LR, batch_size=BATCH, dropout=DROPOUT, seed=0,
             freeze_first=True, threads=None, progress=None, device=None):
    """Returns a new RTPredictor fine-tuned on (peptides, rt), with its calibration mapping output -> RT units.
    peptides/rt: one entry per unique sequence (aggregate repeats before calling). base is not changed."""
    peptides = [str(p) for p in peptides]
    y = np.asarray(rt, dtype=float)
    if len(peptides) < 10:
        raise ValueError("Need at least 10 peptides to retrain the RT model.")
    if threads:
        torch.set_num_threads(threads)
    raw0 = base.predict_raw(peptides)
    reverse = bool(np.corrcoef(raw0, y)[0, 1] < 0)          # HILIC-like: flip the target, keep the orientation
    lo, hi = float(y.min()), float(y.max())
    span = max(hi - lo, 1e-9)
    t = (hi - y) / span if reverse else (y - lo) / span

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = copy.deepcopy(base)
    net = model.model.to(device)
    for p in net.lstm1.parameters():
        p.requires_grad = not freeze_first
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)
    X = encode(peptides).to(device)
    T = torch.tensor(t, dtype=torch.float32, device=device)
    net.train()
    for ep in range(epochs):
        order = rng.permutation(len(peptides))
        for i in range(0, len(order), batch_size):
            b = torch.from_numpy(order[i:i + batch_size]).to(device)
            loss = F.mse_loss(_forward_train(net, X[b], dropout), T[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if progress:
            progress(ep + 1, epochs)
    net.eval()
    model.model = net.to("cpu")
    for p in net.parameters():
        p.requires_grad = True
    # output 0-1 -> RT units (np.polyval order: slope, intercept)
    model.calibration = np.array([-span, hi]) if reverse else np.array([span, lo])
    keys = sorted({p.strip().upper().replace("I", "L") for p in peptides})   # to detect evaluation on training peptides
    model.calibration_info = {"finetuned": True, "reversed": reverse, "n_peptides": len(peptides),
                              "training_keys": keys,
                              "rt_range": (lo, hi), "epochs": epochs, "learning_rate": lr,
                              "model_output_range": (0.0, 1.0)}
    return model


def main():
    from calibrate_rt import read_table, guess_columns, clean_sequence
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", help="CSV/TSV/XLSX with peptide sequences and measured RT")
    ap.add_argument("--model", default="rt_DIA.pt", help="general model to start from (default rt_DIA.pt)")
    ap.add_argument("--out", required=True, help="fine-tuned model to write, e.g. rt_myLC.pt")
    ap.add_argument("--pep-col"), ap.add_argument("--rt-col")
    ap.add_argument("--epochs", type=int, default=None, help="default: 60 for small sets, fewer for large ones")
    ap.add_argument("--name", default="", help="label for this LC method, shown in the apps")
    ap.add_argument("--unit", default="min", help="RT unit of your file: min, s, iRT ... (default min)")
    a = ap.parse_args()

    df = read_table(a.table)
    g_pep, g_rt = guess_columns(df)
    pc, rc = a.pep_col or g_pep, a.rt_col or g_rt
    if pc is None or rc is None:
        sys.exit(f"Name the peptide / RT columns with --pep-col / --rt-col. Columns: {list(df.columns)}")
    d = pd.DataFrame({"peptide": df[pc].map(clean_sequence), "rt": pd.to_numeric(df[rc], errors="coerce")}).dropna()
    d = d[d.peptide.map(check_peptide).isna()]
    per = d.groupby("peptide").rt.median()
    epochs = a.epochs or auto_epochs(len(per))
    print(f"{len(per)} unique peptides from {a.table}; retraining {epochs} epochs ...")
    t0 = time.time()
    model = finetune(RTPredictor(a.model), list(per.index), per.values, epochs=epochs,
                     progress=lambda e, n: print(f"  epoch {e}/{n}", end="\r"))
    model.calibration_info.update(name=a.name or os.path.basename(a.table), unit=a.unit, n_used=len(per),
                                  source_table=os.path.basename(a.table))
    model.save(a.out)
    fit = model.predict(list(per.index))
    print(f"\nSaved {a.out} ({time.time() - t0:.0f} s). Error on the training peptides themselves: median "
          f"{np.median(np.abs(fit - per.values)):.2f} {a.unit} (optimistic; use compare_spectra.py for a fair estimate)")


if __name__ == "__main__":
    main()
