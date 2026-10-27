#!/usr/bin/env python3
"""phep_thu_R.py -- the partition test R for a phase-field code (A4/C3, 21/10/2026).

Reads runs that are already finished and reports:
  * R_angle = spread of the crack kink angle over rank counts / |change of the angle under a physics change|
  * R_peak  = the same ratio for the peak load
  * s_H     = relative spread over rank counts of the sum of the history field H at steps 1 and 40
  * B_angle, B_peak = deviation from a reference run (H on DG0 or at quadrature points),
                      divided by the same physics change.

Why the reference matters: R compares rank counts with each other. A code whose rank counts are
all biased alike passes R (SENT, nx = 200, P1 history field: 1, 3, 4 ranks at -31.13, -30.89,
-29.13 deg, while the reference crack is straight). So R is a screening test: R > 1 shows a
defect, R < 0.1 does not show that the code is right. Give a reference run whenever possible.

Thresholds (same as the gates of the campaign, scripts/run_C2_A3.sh and the CI workflow):
  any R > 1 or any B > 1            -> DEFECT
  every R < 0.1, s_H(step 1) <= 1e-6 and (no reference or every B < 0.1) -> PASS
  otherwise                         -> UNDECIDED

Input: a JSON file
  {
    "ell": 0.05, "x0": 0.60, "dmin": 0.2, "lam_ell": 6,
    "rank_runs":    [ {"label": "1 rank", "log": "...", "field": "....xdmf", "loaddisp": "..."}, ... ],
    "physics_pair": [ {run with physics A at N ranks}, {run with physics B at N ranks} ],
    "reference":    {run of the physics of rank_runs with H on DG0}            (optional)
  }
Each run needs at least one of: field (.xdmf with its .h5, for the angle), loaddisp (for the peak
load), log (for the H sums and, if no loaddisp, the PEAK LOAD line). Paths are relative to the
JSON file. rank_runs and physics_pair[0] should share the physics; physics_pair[0] may be one of
the rank runs.

Run:   python phep_thu_R.py case.json [--out report.md]
Exit:  0 PASS, 1 DEFECT, 2 UNDECIDED, 3 input error.
"""
import argparse, json, math, os, re, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def kink_angle(path, ell, x0, dmin, lam_ell):
    """Angle (deg) of the crack centre line, mean over windows of length lam_ell*ell (measure_kink.py)."""
    from measure_kink import doc_xdmf, duong_tam, goc_theo_doan
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        xy, d = doc_xdmf(path)
    g, _ = goc_theo_doan(duong_tam(xy, d, x0, ell / 5.0, dmin=dmin), lam_ell * ell)
    return float(np.mean(g)) if len(g) else None


def peak_load(run):
    if run.get("loaddisp") and os.path.exists(run["loaddisp"]):
        a = np.loadtxt(run["loaddisp"], comments="#", ndmin=2)
        return float(a[:, 1].max())
    if run.get("log") and os.path.exists(run["log"]):
        m = re.search(r"PEAK LOAD\s*=\s*([-\d.eE+]+)", open(run["log"], errors="ignore").read())
        return float(m.group(1)) if m else None
    return None


def h_sum(run, step):
    if not (run.get("log") and os.path.exists(run["log"])):
        return None
    m = re.search(r"H-INVARIANT step%d: sum=([-\d.eE+]+)" % step, open(run["log"], errors="ignore").read())
    return float(m.group(1)) if m else None


def spread(v):
    v = [x for x in v if x is not None]
    return (max(v) - min(v)) if len(v) >= 2 else None


def rel_spread(v):
    v = [x for x in v if x is not None]
    return (max(v) - min(v)) / max(abs(x) for x in v) if len(v) >= 2 else None


def fmt(x, f="{:.4g}"):
    return "n/a" if x is None else f.format(x)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("case")
    p.add_argument("--out")
    a = p.parse_args()
    try:
        C = json.load(open(a.case))
        base = os.path.dirname(os.path.abspath(a.case))
        for r in C["rank_runs"] + C["physics_pair"] + ([C["reference"]] if C.get("reference") else []):
            for k in ("log", "field", "loaddisp"):
                if r.get(k):
                    r[k] = os.path.join(base, r[k])
        assert len(C["rank_runs"]) >= 2 and len(C["physics_pair"]) == 2
    except Exception as e:
        print("input error:", e)
        sys.exit(3)
    ell, x0, dmin, lam = C.get("ell", 0.05), C.get("x0", 0.60), C.get("dmin", 0.2), C.get("lam_ell", 6)

    def measure(r):
        r["angle"] = kink_angle(r["field"], ell, x0, dmin, lam) if r.get("field") else None
        r["peak"] = peak_load(r)
        r["H1"], r["H40"] = h_sum(r, 1), h_sum(r, 40)
        return r

    runs = [measure(r) for r in C["rank_runs"]]
    pair = [measure(r) for r in C["physics_pair"]]
    ref = measure(C["reference"]) if C.get("reference") else None

    d_ang = abs(pair[1]["angle"] - pair[0]["angle"]) if None not in (pair[0]["angle"], pair[1]["angle"]) else None
    d_pk = abs(pair[1]["peak"] - pair[0]["peak"]) if None not in (pair[0]["peak"], pair[1]["peak"]) else None
    sp_ang, sp_pk = spread([r["angle"] for r in runs]), spread([r["peak"] for r in runs])
    R_ang = sp_ang / d_ang if None not in (sp_ang, d_ang) and d_ang > 0 else None
    R_pk = sp_pk / d_pk if None not in (sp_pk, d_pk) and d_pk > 0 else None
    sH1, sH40 = rel_spread([r["H1"] for r in runs]), rel_spread([r["H40"] for r in runs])
    B_ang = B_pk = None
    if ref:
        if ref["angle"] is not None and d_ang:
            dev = [abs(r["angle"] - ref["angle"]) for r in runs if r["angle"] is not None]
            B_ang = max(dev) / d_ang if dev else None
        if ref["peak"] is not None and d_pk:
            dev = [abs(r["peak"] - ref["peak"]) for r in runs if r["peak"] is not None]
            B_pk = max(dev) / d_pk if dev else None

    Rs = [x for x in (R_ang, R_pk) if x is not None]
    Bs = [x for x in (B_ang, B_pk) if x is not None]
    if any(x > 1 for x in Rs + Bs):
        verdict, code = "DEFECT", 1
    elif Rs and all(x < 0.1 for x in Rs) and sH1 is not None and sH1 <= 1e-6 and all(x < 0.1 for x in Bs):
        verdict, code = "PASS", 0
    else:
        verdict, code = "UNDECIDED", 2

    L = [f"# Partition test R: {os.path.basename(a.case)}", "",
         f"Angle: centre line from x0 = {x0}, d >= {dmin}, windows of {lam} ell (ell = {ell}).", "",
         "| run | angle (deg) | peak load | H sum step 1 | H sum step 40 |", "|---|---|---|---|---|"]
    for r in runs + pair + ([ref] if ref else []):
        L.append(f"| {r['label']} | {fmt(r['angle'], '{:.4f}')} | {fmt(r['peak'], '{:.6f}')} | {fmt(r['H1'], '{:.10e}')} | {fmt(r['H40'], '{:.10e}')} |")
    L += ["", "| quantity | value |", "|---|---|",
          f"| angle spread over rank counts (deg) | {fmt(sp_ang)} |",
          f"| angle change under the physics change (deg) | {fmt(d_ang)} |",
          f"| **R_angle** | {fmt(R_ang, '{:.3g}')} |",
          f"| peak spread over rank counts | {fmt(sp_pk)} |",
          f"| peak change under the physics change | {fmt(d_pk)} |",
          f"| **R_peak** | {fmt(R_pk, '{:.3g}')} |",
          f"| s_H step 1 / step 40 | {fmt(sH1, '{:.2e}')} / {fmt(sH40, '{:.2e}')} |",
          f"| **B_angle** (max deviation from reference / physics change) | {fmt(B_ang, '{:.3g}')} |",
          f"| **B_peak** | {fmt(B_pk, '{:.3g}')} |",
          "", f"**Verdict: {verdict}**", ""]
    nang = sum(r["angle"] is None for r in runs if r.get("field"))
    if nang:
        L.append(f"- {nang} rank run(s) gave no measurable angle (crack did not reach x0 + one window); R_angle uses the others.")
    if not ref:
        L.append("- No reference run: a PASS here does not exclude a bias shared by all rank counts.")
    if verdict == "PASS" and ref:
        L.append("- PASS with a reference run: rank counts agree with each other and with the reference.")
    txt = "\n".join(L) + "\n"
    print(txt)
    if a.out:
        open(a.out, "w").write(txt)
    sys.exit(code)


if __name__ == "__main__":
    main()
