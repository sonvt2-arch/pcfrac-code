"""Do goc nut tu DIEM KIEM (ck_<tag>/r*of*_s*.npz) khi khong con tep .xdmf/.h5.

    python goc_tu_diem_kiem.py ck_C2_dg0_sp_200_n4 --ell 0.05 --x0 0.60 --dmin 0.2 --ve

Moi nhan luu mang CUC BO cua d (ca ghost) va toa do bac tu do (xyz = [V_u, V_d,
V_H] noi duoi).  Ghep cac nhan theo toa do nut, bo trung ghost, roi dua (xy, d)
vao dung ham do cua measure_kink.py (thay doc_xdmf), nen luat do khong doi.
Chon khe co buoc LON NHAT chung cho moi nhan.
"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure_kink as mk  # noqa: E402


def doc_diem_kiem(ck):
    tep = glob.glob(os.path.join(ck, "r*of*_s*.npz"))
    if not tep:
        sys.exit(f"khong co diem kiem trong {ck}")
    buoc = {}
    for f in tep:
        r, n, s = map(int, re.search(r"r(\d+)of(\d+)_s(\d+)\.npz$", f).groups())
        with np.load(f) as z:
            buoc.setdefault(int(z["step"]), {})[r] = f
        nhan = n
    chung = [s for s, m in buoc.items() if len(m) == nhan]
    if not chung:
        sys.exit("khong co buoc chung cho moi nhan")
    s = max(chung)
    xy, d = [], []
    for r in range(nhan):
        with np.load(buoc[s][r]) as z:
            dd = z["d"]
            nu = len(z["u"]) // 2          # V_u bac 2 khoi: toa do theo nut
            x = z["xyz"].reshape(-1, 3)[nu:nu + len(dd), :2]
            xy.append(x); d.append(dd)
    xy = np.vstack(xy); d = np.concatenate(d)
    _, idx = np.unique(np.round(xy, 10), axis=0, return_index=True)
    print(f"  DIEM KIEM {ck}: buoc {s}, {nhan} nhan, {len(idx)} nut (sau khi bo ghost)")
    return xy[idx], d[idx]


if __name__ == "__main__":
    ck = sys.argv.pop(1)
    xy, d = doc_diem_kiem(ck)
    mk.doc_xdmf = lambda _p: (xy, d)
    sys.argv.insert(1, "khong-dung.xdmf")
    mk.main()
