#!/usr/bin/env python3
"""
do_sens_pca.py -- thuoc do vong 9-R cho SENS cat (ghi truoc: Pervious-Concrete
doc/CA-VONG9-R-2026-11-13.md).  KHONG phu thuoc huong truc: thay cho do_sens_hang.py
(vong 8-M) bi lat dau +-90 do voi vet gan thang dung.  Chi can numpy (h5py neu co;
neu khong, doc truc tiep tap du lieu lien tuc trong tep .h5 cua dolfinx).

  (a) Duong tam theo VONG TRON quanh dinh khia (0.5, 0.5) (nhu do_O.py): ban kinh r tu
      2 ell, buoc h/2; tren moi vong lay mau d noi suy song tuyen (cach ~h/4), ngoai mien
      = 0.  Doan lien tiep d >= 0.5 la ung vien; diem tam = trong tam (trong so d) cua doan.
      Vong dau: doan co tong d lon nhat, bo doan co tam goc trong 180 +- 30 do (chinh vet
      khia).  Sau do: doan gan diem truoc nhat, chi nhan neu cach <= 3 ell.  Dung khi het
      doan, hoac diem tam cach bien mien < 2 ell.
  (b) Goc: cua so dai Lam = 2/4/6 ell theo do dai cung cua duong tam, buoc Lam/2 (< 3 diem
      hoac phu < 0.8 Lam: bo); huong chinh PCA cua cua so, DINH HUONG theo chieu di tu
      khia ra (v . (p_cuoi - p_dau) > 0); goc atan2 trong (-180, 180].  theta = trung
      binh VONG (goc cua tong vecto don vi).  Vet di xuong: theta trong (-180, 0].
      Chinh: 6 ell (lui 4 ell, 2 ell).
  (c) Hausdorff hai duong tam (mm, /ell): toan bo va tren khoang r chung.

Dung:
  python3 do_sens_pca.py A.xdmf B.xdmf --nhan Q1 DG0 [--ell 0.01] [--json kq.json] [--png h.png] [--ve]
  python3 do_sens_pca.py --thu [--ve]      # thu tren truong tong hop (vet thang goc biet truoc)
"""
import argparse, json, math, os, re, struct, zlib
import numpy as np

XT = (0.5, 0.5)
DMIN = 0.5


# ---------------------------------------------------------------- doc truong
def _doc_h5_tho(h5, ten, n_byte):
    """Khong co h5py: tim thong diep Data Layout (v3, lien tuc) co kich thuoc dung n_byte.
    Chi dung khi kich thuoc la duy nhat trong tep (kiem tra)."""
    b = open(h5, "rb").read()
    gap, i = [], 0
    while True:
        i = b.find(b"\x03\x01", i)
        if i < 0:
            break
        a, s = struct.unpack("<QQ", b[i + 2:i + 18])
        if s == n_byte and a + s <= len(b):
            gap.append(a)
        i += 1
    gap = sorted(set(gap))
    if len(gap) != 1:
        raise SystemExit(f"{h5}:{ten}: tim thay {len(gap)} khoi du lieu {n_byte} byte -- can h5py")
    return b[gap[0]:gap[0] + n_byte]


def doc(path):
    x = open(path).read()
    hp = os.path.join(os.path.dirname(path) or ".", os.path.splitext(os.path.basename(path))[0] + ".h5")
    m = re.search(r'Dimensions="(\d+) (\d+)" Format="HDF">[^<]*:/Mesh/mesh/geometry', x)
    n, c = int(m.group(1)), int(m.group(2))
    ks = re.findall(r':/Function/damage/([^<]+)<', x)
    k = sorted(ks, key=float)[-1]
    try:
        import h5py
        with h5py.File(hp, "r") as f:
            xy = f["Mesh/mesh/geometry"][()][:, :2]
            d = np.ravel(f["Function/damage"][k][()])
    except ImportError:
        if len(ks) != 1:
            raise SystemExit("nhieu buoc thoi gian -- can h5py")
        xy = np.frombuffer(_doc_h5_tho(hp, "geometry", n * c * 8), "<f8").reshape(n, c)[:, :2]
        d = np.frombuffer(_doc_h5_tho(hp, "damage", n * 8), "<f8").copy()
    if len(d) != len(xy):
        raise SystemExit(f"{path}: d co {len(d)} gia tri, luoi {len(xy)} nut")
    if not (xy.min() > -1e-9 and xy.max() < 1 + 1e-9 and d.min() > -0.01 and d.max() < 1.01):
        raise SystemExit(f"{path}: gia tri ngoai khoang -- doc sai")
    return xy, d, k


def luoi(xy, d):
    u = np.unique(np.round(xy[:, 0], 12)); h = float(np.min(np.diff(u)))
    nn = int(round(1.0 / h))
    G = np.full((nn + 1, nn + 1), np.nan)
    G[np.rint(xy[:, 1] / h).astype(int), np.rint(xy[:, 0] / h).astype(int)] = d
    if np.isnan(G).any():
        raise SystemExit("luoi khong deu / thieu nut")
    return G, h


def noi_suy(G, h, X, Y):
    n = G.shape[0] - 1
    fx, fy = X / h, Y / h
    i0 = np.clip(np.floor(fx).astype(int), 0, n - 1); j0 = np.clip(np.floor(fy).astype(int), 0, n - 1)
    tx, ty = fx - i0, fy - j0
    v = ((1 - tx) * (1 - ty) * G[j0, i0] + tx * (1 - ty) * G[j0, i0 + 1]
         + (1 - tx) * ty * G[j0 + 1, i0] + tx * ty * G[j0 + 1, i0 + 1])
    return np.where((X >= 0) & (X <= 1) & (Y >= 0) & (Y <= 1), v, 0.0)


# ---------------------------------------------------------------- (a) duong tam
def duong_tam(G, h, ell):
    pts, rr, prev = [], [], None
    r = 2.0 * ell
    while r < 1.5:
        nth = max(64, int(math.ceil(2 * math.pi * r / (h / 4))))
        th = np.linspace(-math.pi, math.pi, nth, endpoint=False)
        X, Y = XT[0] + r * np.cos(th), XT[1] + r * np.sin(th)
        v = noi_suy(G, h, X, Y)
        m = v >= DMIN
        if not m.any():
            break
        # doan lien tiep tren vong (noi dau-cuoi)
        if m.all():
            break
        k0 = int(np.argmin(m))             # mot chi so khong thuoc doan nao
        idx = np.roll(np.arange(nth), -k0); mm = m[idx]
        doan, cur = [], []
        for q, t in zip(idx, mm):
            if t:
                cur.append(q)
            elif cur:
                doan.append(cur); cur = []
        if cur:
            doan.append(cur)
        ung = []
        for s in doan:
            s = np.asarray(s); w = v[s]
            ung.append((float(np.average(X[s], weights=w)), float(np.average(Y[s], weights=w)), float(w.sum())))
        if prev is None:
            ok = [u for u in ung if abs(abs(math.degrees(math.atan2(u[1] - XT[1], u[0] - XT[0]))) - 180) > 30]
            if not ok:
                break
            c = max(ok, key=lambda u: u[2])
        else:
            c = min(ung, key=lambda u: math.hypot(u[0] - prev[0], u[1] - prev[1]))
            if math.hypot(c[0] - prev[0], c[1] - prev[1]) > 3 * ell:
                break
        pts.append(c[:2]); rr.append(r); prev = c
        if min(c[0], c[1], 1 - c[0], 1 - c[1]) < 2 * ell:
            break
        r += 0.5 * h
    return np.asarray(pts).reshape(-1, 2), np.asarray(rr)


# ---------------------------------------------------------------- (b) goc
def goc_cung(tam, Lam):
    if len(tam) < 3:
        return []
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(tam, axis=0).T))])
    gs, sb = [], 0.0
    while sb + Lam <= s[-1] + 1e-12:
        i, j = np.searchsorted(s, sb), np.searchsorted(s, sb + Lam, side="right")
        sb += 0.5 * Lam
        if j - i < 3 or s[j - 1] - s[i] < 0.8 * Lam:
            continue
        P = tam[i:j] - tam[i:j].mean(0)
        v = np.linalg.svd(P, full_matrices=False)[2][0]
        if np.dot(v, tam[j - 1] - tam[i]) < 0:
            v = -v
        gs.append(math.degrees(math.atan2(v[1], v[0])))
    return gs


def tb_vong(gs):
    a = np.radians(gs)
    return math.degrees(math.atan2(np.sin(a).sum(), np.cos(a).sum()))


def hausdorff(A, B):
    if not len(A) or not len(B):
        return None
    D = np.hypot(A[:, None, 0] - B[None, :, 0], A[:, None, 1] - B[None, :, 1])
    return float(max(D.min(1).max(), D.min(0).max()))


def do_mot(G, h, ell):
    tam, rr = duong_tam(G, h, ell)
    r = {"so_diem_tam": int(len(tam)), "h_luoi": h}
    if len(tam):
        r["dau"] = [round(float(v), 4) for v in tam[0]]; r["cuoi"] = [round(float(v), 4) for v in tam[-1]]
        r["r_max"] = float(rr[-1])
    for k in (2, 4, 6):
        g = goc_cung(tam, k * ell)
        r[f"goc_{k}ell"] = tb_vong(g) if g else None
        r[f"cuaso_{k}ell"] = len(g)
        r[f"cuaso_huong_len_{k}ell"] = int(sum(x > 0 for x in g))
        r[f"cuaso_min_max_{k}ell"] = [round(min(g), 2), round(max(g), 2)] if g else None
    r["muc_dung"] = next((f"{k}ell" for k in (6, 4, 2) if r[f"goc_{k}ell"] is not None), None)
    r["theta"] = r[f"goc_{r['muc_dung'][0]}ell"] if r["muc_dung"] else None
    return r, tam, rr


# ---------------------------------------------------------------- ban do
def ve_ascii(G, tam, nc=100, nr=50):
    n = G.shape[0]
    iy = (np.arange(n) * nr // n); ix = (np.arange(n) * nc // n)
    M = np.zeros((nr, nc))
    np.maximum.at(M, (iy[:, None].repeat(n, 1), ix[None, :].repeat(n, 0)), G)
    C = np.where(M >= 0.9, "#", np.where(M >= 0.5, "+", np.where(M >= 0.2, ".", " ")))
    for x, y in tam:
        C[min(int(y * nr), nr - 1), min(int(x * nc), nc - 1)] = "o"
    return "\n".join("|" + "".join(r) + "|" for r in C[::-1])


def ve_png(path, panels, s=2):
    """panels: [(G, [(tam, mau)...]), ...] -> PNG xam + duong tam mau (khong can matplotlib)."""
    imgs = []
    for G, ds in panels:
        I = np.repeat(np.repeat((255 * (1 - np.clip(G[::-1], 0, 1))).astype(np.uint8), s, 0), s, 1)
        A = np.stack([I, I, I], -1)
        n = A.shape[0]
        for tam, col in ds:
            for x, y in tam:
                i, j = int(round((1 - y) * (n - 1))), int(round(x * (n - 1)))
                A[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2] = col
        imgs.append(A)
    sep = np.full((imgs[0].shape[0], 8, 3), 255, np.uint8)
    A = imgs[0]
    for b in imgs[1:]:
        A = np.concatenate([A, sep, b], 1)
    hgt, wid = A.shape[:2]
    raw = b"".join(b"\x00" + A[i].tobytes() for i in range(hgt))
    def ch(t, dat):
        return struct.pack(">I", len(dat)) + t + dat + struct.pack(">I", zlib.crc32(t + dat) & 0xffffffff)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", wid, hgt, 8, 2, 0, 0, 0))
                + ch(b"IDAT", zlib.compress(raw, 9)) + ch(b"IEND", b""))


# ---------------------------------------------------------------- truong tong hop
def tong_hop(goc_do, ell=0.01, nx=200):
    """d = exp(-dist/ell) toi hop: vet khia y=0.5, x<=0.5 va tia tu (0.5,0.5) goc goc_do toi bien."""
    h = 1.0 / nx
    g = np.arange(nx + 1) * h
    X, Y = np.meshgrid(g, g)
    d1 = np.hypot(np.maximum(X - 0.5, 0), Y - 0.5)
    t = math.radians(goc_do); e = np.array([math.cos(t), math.sin(t)])
    L = min([(b - 0.5) / c for b, c in ((0, e[0]), (1, e[0])) if c * (b - 0.5) > 0]
            + [(b - 0.5) / c for b, c in ((0, e[1]), (1, e[1])) if c * (b - 0.5) > 0])
    s = np.clip((X - 0.5) * e[0] + (Y - 0.5) * e[1], 0, L)
    d2 = np.hypot(X - 0.5 - s * e[0], Y - 0.5 - s * e[1])
    return np.exp(-np.minimum(d1, d2) / ell), h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("xdmf", nargs="*")
    p.add_argument("--nhan", nargs=2, default=["Q1", "DG0"])
    p.add_argument("--ell", type=float, default=0.01)
    p.add_argument("--json"); p.add_argument("--png"); p.add_argument("--ve", action="store_true")
    p.add_argument("--thu", action="store_true")
    a = p.parse_args(); ell = a.ell
    if a.thu:
        kq = {}
        print("THU tren truong tong hop (luoi 200x200, d = exp(-dist/ell)); sai so = theta_do - theta_that")
        for gt in (-30, -45, -53.6, -60, -75, -77, -85, -90, -95, -105):
            G, h = tong_hop(gt, ell)
            r, tam, _ = do_mot(G, h, ell)
            sai = {k: (None if r[f"goc_{k}ell"] is None else r[f"goc_{k}ell"] - gt) for k in (2, 4, 6)}
            kq[str(gt)] = {"do": r, "sai": sai}
            print(f"  that {gt:7.1f}: {r['so_diem_tam']} diem; 2/4/6ell = " + " / ".join(
                "--" if r[f"goc_{k}ell"] is None else f"{r[f'goc_{k}ell']:.2f} ({r[f'cuaso_{k}ell']})" for k in (2, 4, 6))
                + f"; sai 6ell = {sai[6]:+.2f}; cua so huong len 6ell: {r['cuaso_huong_len_6ell']}")
            if a.ve and gt in (-75, -90):
                print(ve_ascii(G, tam))
        mx = max(abs(v["sai"][6]) for v in kq.values())
        print(f"sai so lon nhat 6ell = {mx:.2f} do -> {'DAT' if mx <= 1.0 else 'KHONG DAT'} (nguong 1 do)")
        if a.json:
            json.dump(kq, open(a.json, "w"), indent=1, ensure_ascii=False)
        return
    kq, du = {}, {}
    for nh, f in zip(a.nhan, a.xdmf):
        xy, d, buoc = doc(f); G, h = luoi(xy, d)
        r, tam, rr = do_mot(G, h, ell)
        r["tep"] = os.path.basename(f); r["khoa_buoc"] = buoc
        kq[nh] = r; du[nh] = (G, tam, rr)
        print(f"{nh}: {r['tep']} (buoc {buoc}, h={h:.4g}), {r['so_diem_tam']} diem tam"
              + (f", dau ({r['dau'][0]:.3f};{r['dau'][1]:.3f}) cuoi ({r['cuoi'][0]:.3f};{r['cuoi'][1]:.3f}), r_max {r['r_max']:.3f}" if len(tam) else ""))
        print("   goc 2/4/6ell = " + " / ".join(
            "--" if r[f"goc_{k}ell"] is None else
            f"{r[f'goc_{k}ell']:.2f} ({r[f'cuaso_{k}ell']} cua so, {r[f'cuaso_huong_len_{k}ell']} huong len, min/max {r[f'cuaso_min_max_{k}ell']})"
            for k in (2, 4, 6)))
        if a.ve:
            print(ve_ascii(G, tam))
    A, B = (du[n] for n in a.nhan)
    dh = hausdorff(A[1], B[1])
    rc = min(A[2][-1], B[2][-1]) if len(A[2]) and len(B[2]) else None
    dc = hausdorff(A[1][A[2] <= rc + 1e-12], B[1][B[2] <= rc + 1e-12]) if rc else None
    kq["hausdorff_mm"] = dh; kq["hausdorff_ell"] = dh / ell if dh is not None else None
    kq["hausdorff_r_chung_mm"] = dc; kq["r_chung"] = rc
    if dh is not None:
        print(f"Hausdorff = {dh:.4f} mm = {dh/ell:.2f} ell; tren r <= {rc:.3f} chung: {dc:.4f} mm = {dc/ell:.2f} ell")
    t1, t2 = (kq[n]["theta"] for n in a.nhan)
    if t1 is not None and t2 is not None:
        dl = abs((t1 - t2 + 180) % 360 - 180)
        kq["delta"] = dl
        pq = ("DG0 khong khop nguon; khong ket luan" if not t2 < -5 else
              "co do lech" if dl > 10 else "khong" if dl < 3 else "lan")
        kq["phan_quyet_theo_chu"] = pq
        print(f"|theta_{a.nhan[0]} - theta_{a.nhan[1]}| ({kq[a.nhan[0]]['muc_dung']}/{kq[a.nhan[1]]['muc_dung']}) = {dl:.2f} do -> {pq}")
    if a.json:
        json.dump(kq, open(a.json, "w"), indent=1, ensure_ascii=False)
    if a.png:
        ve_png(a.png, [(du[n][0], [(du[n][1], (255, 0, 0)), (du[o][1], (0, 160, 255))])
                       for n, o in (a.nhan, a.nhan[::-1])])


if __name__ == "__main__":
    main()
