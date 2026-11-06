"""
rt_model.py — PyTorch port of the TransferL retention-time model
(https://github.com/BoudewijnHollebrands/TransferL).

Architecture identical to the original Keras model:
  one-hot (20 positions x 20 AA) -> BiLSTM(50) -> BiLSTM(50) -> Dense 512 -> Dense 256 -> Dense 1
(the Dense layers have no activation, as in the original).

Output is RT on a 0-1 scale (the general models were trained on min-max normalized RT).
Use `RTPredictor.calibrate()` with a few peptides of known RT to map that to your own
minutes / iRT scale.

Usage
-----
    from rt_model import RTPredictor
    rt = RTPredictor("rt_DIA.pt")
    rt.predict(["AAFN", "PEPTIDEK"])            # relative 0-1 scale
    rt.calibrate(known_seqs, known_rts)         # optional: quick fit (or use calibrate_rt.py)
    rt.predict(["AAFN", "PEPTIDEK"])            # now in your units
"""
import io

import numpy as np
import torch
import torch.nn as nn

AA = "ARNDCQEGHILKMFPSTWYV"
MAX_LEN = 20          # input width of the model
MAX_PEPTIDE_LEN = 19  # original training filter was Length < 20


def encode(peptides, max_len=MAX_LEN):
    """One-hot encode exactly like the original peptide_to_vector (positions x AA)."""
    X = np.zeros((len(peptides), max_len, len(AA)), dtype=np.float32)
    for n, p in enumerate(peptides):
        for i, c in enumerate(str(p)[:max_len]):
            j = AA.find(c)
            if j >= 0:
                X[n, i, j] = 1.0
    return torch.from_numpy(X)


def check_peptide(p):
    """Return None if the peptide is inside the model's domain, else a reason string."""
    p = str(p)
    if len(p) > MAX_PEPTIDE_LEN:
        return f"longer than {MAX_PEPTIDE_LEN} residues"
    bad = sorted(set(c for c in p if c not in AA))
    if bad:
        return f"non-standard characters {''.join(bad)} (modifications are not supported)"
    return None


class RTNet(nn.Module):
    def __init__(self, units=50):
        super().__init__()
        self.lstm1 = nn.LSTM(len(AA), units, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(2 * units, units, batch_first=True, bidirectional=True)
        self.d1 = nn.Linear(2 * units, 512)
        self.d2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        _, (h, _) = self.lstm2(x)               # h: (2, B, units) = last fwd, last bwd
        x = torch.cat([h[0], h[1]], dim=1)      # same as Keras Bidirectional(return_sequences=False)
        return self.out(self.d2(self.d1(x))).squeeze(-1)


class RTPredictor:
    def __init__(self, path, device="cpu"):
        """path: file path, file-like object, or the raw bytes of an RT model file."""
        src = io.BytesIO(path) if isinstance(path, (bytes, bytearray)) else path
        try:
            ckpt = torch.load(src, map_location=device, weights_only=False)
        except Exception:
            raise ValueError("This is not an RT model file (made by convert_keras_to_torch.py / calibrate_rt.py).")
        if not isinstance(ckpt, dict) or "state_dict" not in ckpt or \
                not any(k.startswith("lstm1.") for k in ckpt["state_dict"]):
            raise ValueError("This is not an RT model file (made by convert_keras_to_torch.py / calibrate_rt.py).")
        self.model = RTNet(ckpt.get("units", 50)).to(device).eval()
        self.model.load_state_dict(ckpt["state_dict"])
        self.device = device
        self.source = ckpt.get("source", "")
        # calibration = polynomial coefficients, highest power first (np.polyval order)
        self.calibration = np.asarray(ckpt.get("calibration", (1.0, 0.0)), dtype=float)
        self.calibration_info = ckpt.get("calibration_info", {})

    @torch.no_grad()
    def predict_raw(self, peptides, batch_size=4096):
        out = []
        for i in range(0, len(peptides), batch_size):
            out.append(self.model(encode(peptides[i:i + batch_size]).to(self.device)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def predict(self, peptides):
        """Predicted RT (0-1 scale, or your units after calibrate())."""
        return np.polyval(self.calibration, self.predict_raw(list(peptides)))

    @property
    def is_calibrated(self):
        c = self.calibration
        return not (c.shape == (2,) and np.allclose(c, (1.0, 0.0)))

    def calibrate(self, peptides, rts, degree=1):
        """Least-squares polynomial map from model output to your RT scale.
        For a full robust workflow (outlier removal, hold-out check, report) use calibrate_rt.py."""
        raw = self.predict_raw(list(peptides))
        self.calibration = np.polyfit(raw, np.asarray(rts, dtype=float), degree)
        return self.calibration

    def save(self, path):
        """Save weights plus current calibration (so the app loads it pre-calibrated).
        path may also be a file-like object (e.g. io.BytesIO for a download button)."""
        torch.save({"state_dict": self.model.state_dict(), "units": 50, "source": self.source,
                    "calibration": tuple(float(c) for c in self.calibration),
                    "calibration_info": self.calibration_info}, path)

    def to_bytes(self):
        buf = io.BytesIO()
        self.save(buf)
        return buf.getvalue()
