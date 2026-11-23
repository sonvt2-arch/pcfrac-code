"""
ly_thuyet_kii.py -- Ly thuyet huong 1 (thay duyet 23/11/2026): "K_II ao" do cach luu H.

Doc kem: doc/LY-THUYET-1-KII-AO.md (repo du an).  Chi can numpy (+ scipy cho brentq neu co;
khong co thi dung chia doi tu viet).  Khong doc truong FEM; dau vao la cac A_H da do o vong 11-B.

Quy uoc: dinh ranh o goc toa do, ranh nam tren truc x am, vet nut thang di theo +x.
phi = goc cuc (-pi, pi]; theta > 0: nut di LEN (y > 0.5 trong bai toan).  Bien dang phang,
nu = 0.3, tach Amor 2D: psi+ = K2/2 <tr e>+^2 + mu |dev e|^2, K2 = lam + mu (nhu benchmark_A.py).

Phan:
 0. Truong Williams (so hang dau) mode I, II dua tren chuyen vi; KIEM bang so: can bang (sai phan
    huu han), mat nut tu do, chuan hoa K (sigma_yy(r,0) sqrt(2 pi r) = K_I, sigma_xy = K_II).
    Cong thuc chuyen vi lay tu sach giao khoa nhung DUOC KIEM bang 3 phep tren, khong tin vao tri nho.
 1. Ung suat vong/cat theo huong phi -> MTS (Erdogan & Sih 1963): so voi cong thuc
    theta = 2 arctan[(K_I - sqrt(K_I^2 + 8 K_II^2)) / (4 K_II)] (kiem bang so).
 2. Xap xi bac nhat cho nut gay khuc (Cotterell & Rice 1980): k_I(theta), k_II(theta) bang
    sqrt(2 pi r) sigma_tt, sqrt(2 pi r) sigma_rt cua truong goc theo huong theta.
    -> G(theta) ~ k_I^2 + k_II^2 (G_max) ; PLS (Gol'dstein & Salganik 1974) k_II = 0.
 3. He so alpha: A(beta) = (int_T psi+ - int_B psi+)/int psi+ tren vung R (x in [0,4l], |y| <= 4l)
    cho truong K_I + beta K_I (mode II).  psi ~ 1/r nen alpha khong phu thuoc l.
 4. Mo hinh A (ghi truoc): beta_eff = A_H / alpha  (A_H do o buoc 1), roi
    MTS(beta), G_max(beta), PLS-lech: k_II,that(theta) + beta k_I,that(theta) = 0.
 5. Mo hinh B (tinh tien): ghi de = dao ham theo y lay o y + s h/2 -> du doan A_H(buoc 1) tu truong
    Williams roi rac (luoi Q1 deu, nguoi ghi = o tren-phai) de kiem "A_H do = dich h/2";
    goc trang thai dung: tan(theta) = -s W (h/2) / r*, r* = l (B0, khong tham so) hoac
    r* hieu chinh tren 2D nx200 (B1).
"""
import math
import sys
import numpy as np

NU = 0.3
E = 210000.0
MU = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))      # bien dang phang
K2 = LAM + MU
KAPPA = 3 - 4 * NU


# ---------------------------------------------------------------- 0. Williams
def u_williams(x, y, KI, KII):
    r = np.hypot(x, y)
    p = np.arctan2(y, x)
    c = np.sqrt(r / (2 * np.pi)) / (2 * MU)
    s2, c2 = np.sin(p / 2), np.cos(p / 2)
    ux = KI * c * c2 * (KAPPA - 1 + 2 * s2 ** 2) + KII * c * s2 * (KAPPA + 1 + 2 * c2 ** 2)
    uy = KI * c * s2 * (KAPPA + 1 - 2 * c2 ** 2) - KII * c * c2 * (KAPPA - 1 - 2 * s2 ** 2)
    return ux, uy


def grad_fd(x, y, KI, KII, eps=1e-7):
    """grad u bang sai phan trung tam (do dai eps tuong doi voi r)."""
    r = np.hypot(x, y)
    e = eps * np.maximum(r, 1e-300)
    uxp, uyp = u_williams(x + e, y, KI, KII)
    uxm, uym = u_williams(x - e, y, KI, KII)
    vxp, vyp = u_williams(x, y + e, KI, KII)
    vxm, vym = u_williams(x, y - e, KI, KII)
    return ((uxp - uxm) / (2 * e), (vxp - vxm) / (2 * e),     # dux/dx, dux/dy
            (uyp - uym) / (2 * e), (vyp - vym) / (2 * e))     # duy/dx, duy/dy


def strain(x, y, KI, KII):
    a, b, c, d = grad_fd(x, y, KI, KII)
    return a, d, 0.5 * (b + c)        # exx, eyy, exy


def stress(x, y, KI, KII):
    exx, eyy, exy = strain(x, y, KI, KII)
    tr = exx + eyy
    return 2 * MU * exx + LAM * tr, 2 * MU * eyy + LAM * tr, 2 * MU * exy


def psi_amor(exx, eyy, exy):
    tr = exx + eyy
    trp = np.maximum(tr, 0.0)
    dxx, dyy = exx - tr / 2, eyy - tr / 2
    return 0.5 * K2 * trp ** 2 + MU * (dxx ** 2 + dyy ** 2 + 2 * exy ** 2)


def kiem_williams():
    out = []
    r = 1e-3
    for KI, KII, ten in ((1.0, 0.0, "I"), (0.0, 1.0, "II")):
        sxx, syy, sxy = stress(np.array([r]), np.array([0.0]), KI, KII)
        out.append(f"  mode {ten}: sigma_yy(r,0)*sqrt(2 pi r) = {syy[0]*math.sqrt(2*math.pi*r):.6f} "
                   f"(phai {KI}), sigma_xy*sqrt(2 pi r) = {sxy[0]*math.sqrt(2*math.pi*r):.6f} (phai {KII})")
        # mat nut tu do: phi -> +-pi
        for ph in (math.pi - 1e-6, -math.pi + 1e-6):
            xx, yy = r * math.cos(ph), r * math.sin(ph)
            s = stress(np.array([xx]), np.array([yy]), KI, KII)
            scale = 1 / math.sqrt(2 * math.pi * r)
            out.append(f"     mat phi={ph:+.4f}: |sigma_yy|/s = {abs(s[1][0])/scale:.2e}, |sigma_xy|/s = {abs(s[2][0])/scale:.2e}")
        # can bang: div sigma tai vai diem, sai phan cua ung suat
        worst = 0.0
        rng = np.random.default_rng(1)
        for _ in range(20):
            rr = 10 ** rng.uniform(-3, -1)
            ph = rng.uniform(-3.0, 3.0)
            xx, yy = rr * math.cos(ph), rr * math.sin(ph)
            h = 1e-4 * rr
            sp = stress(np.array([xx + h]), np.array([yy]), KI, KII)
            sm = stress(np.array([xx - h]), np.array([yy]), KI, KII)
            tp = stress(np.array([xx]), np.array([yy + h]), KI, KII)
            tm = stress(np.array([xx]), np.array([yy - h]), KI, KII)
            fx = (sp[0] - sm[0]) / (2 * h) + (tp[2] - tm[2]) / (2 * h)
            fy = (sp[2] - sm[2]) / (2 * h) + (tp[1] - tm[1]) / (2 * h)
            s0 = 1 / math.sqrt(2 * math.pi * rr) / rr
            worst = max(worst, float(abs(fx[0]) / s0), float(abs(fy[0]) / s0))
        out.append(f"     can bang: max |div sigma| / (K r^-3/2) = {worst:.2e}")
    return out


# ---------------------------------------------------------------- 1-2. tieu chi re nhanh
def k_local(theta, KI, KII, r=1e-4):
    """sqrt(2 pi r) (sigma_tt, sigma_rt) cua truong goc theo huong theta (xap xi bac nhat Cotterell-Rice)."""
    x, y = r * np.cos(theta), r * np.sin(theta)
    sxx, syy, sxy = stress(x, y, KI, KII)
    c, s = np.cos(theta), np.sin(theta)
    stt = sxx * s * s + syy * c * c - 2 * sxy * s * c
    srt = (syy - sxx) * s * c + sxy * (c * c - s * s)
    f = np.sqrt(2 * np.pi * r)
    return stt * f, srt * f


def _brent(f, a, b, tol=1e-12):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        return float("nan")
    for _ in range(200):
        m = 0.5 * (a + b)
        fm = f(m)
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
        if b - a < tol:
            break
    return 0.5 * (a + b)


def theta_mts_cong_thuc(beta):
    if abs(beta) < 1e-15:
        return 0.0
    return 2 * math.atan((1 - math.sqrt(1 + 8 * beta ** 2)) / (4 * beta))


def theta_mts_so(beta):
    """max sigma_tt: sigma_rt = 0 voi sigma_tt > 0, tim trong (-pi/2.., pi/2..)."""
    g = lambda t: float(k_local(np.array([t]), 1.0, beta)[1][0])
    ts = np.linspace(-1.5, 1.5, 3001)
    best, bt = -1e300, 0.0
    vals = k_local(ts, 1.0, beta)[0]
    i = int(np.argmax(vals))
    lo, hi = ts[max(i - 1, 0)], ts[min(i + 1, len(ts) - 1)]
    t = _brent(g, lo, hi)
    return t if not math.isnan(t) else float(ts[i])


def theta_gmax(beta):
    ts = np.linspace(-1.5, 1.5, 30001)
    kI, kII = k_local(ts, 1.0, beta)
    G = np.where(kI > 0, kI ** 2, 0.0) + kII ** 2
    return float(ts[int(np.argmax(G))])


def theta_pls_lech(beta):
    """Huong nut ma k_II,that(theta) + beta k_I,that(theta) = 0 (truong ngoai mode I thuan,
    do lech luu H coi nhu K_II ao ti le k_I tai dinh moi)."""
    f = lambda t: float(k_local(np.array([t]), 1.0, 0.0)[1][0] + beta * k_local(np.array([t]), 1.0, 0.0)[0][0])
    return _brent(f, -1.4, 1.4)


# ---------------------------------------------------------------- 3. alpha
def A_vung(beta, a=4.0, b=4.0, n=20001):
    """A cua psi+ tren hinh chu nhat x in (0,a], |y| <= b (don vi l), truong K_I = 1, K_II = beta.
    psi = f(phi)/r  ->  int psi dA = int f(phi) R(phi) dphi, R = khoang cach toi bien."""
    ph = np.linspace(-math.pi / 2 + 1e-9, math.pi / 2 - 1e-9, n)
    r0 = 1e-4
    exx, eyy, exy = strain(r0 * np.cos(ph), r0 * np.sin(ph), 1.0, beta)
    f = psi_amor(exx, eyy, exy) * r0              # = r psi, doc lap r
    R = np.minimum(a / np.cos(ph), b / np.abs(np.sin(ph) + 1e-300))
    w = f * R
    T = np.trapezoid(np.where(ph > 0, w, 0.0), ph)
    B = np.trapezoid(np.where(ph < 0, w, 0.0), ph)
    return (T - B) / (T + B)


def alpha():
    db = 1e-4
    return (A_vung(db) - A_vung(-db)) / (2 * db)


# ---------------------------------------------------------------- 5. mo hinh B: A_H roi rac do ghi de
def A_ghi_de_williams(h_l, a=4.0, b=4.0, s=+1, dg0=False):
    """Luoi Q1 deu buoc h (don vi l), dinh ranh tai nut (0,0); u = Williams mode I tai nut.
    Ghi de 2D 1 nhan: nut (i,j) nhan psi+ cua o tren-phai (s=+1) hoac duoi-trai (s=-1, ghi nguoc)
    tinh tai chinh nut do.  H CG1 noi suy song tuyen; tich phan o = TB 4 nut * h^2 (dung).
    Vung R: o co tam x in [0,a], |y| <= b.  Tra ve A_H."""
    h = h_l
    nx, ny = int(round(a / h)), int(round(b / h))
    ii = np.arange(-1, nx + 2)
    jj = np.arange(-ny - 2, ny + 3)
    X, Y = np.meshgrid(ii * h, jj * h, indexing="ij")
    # mat nut (x<0, y=0) khong dung toi trong R; tranh nhanh cat: nut x<0,y=0 lay phi=+pi (khong vao R)
    ux, uy = u_williams(X, np.where((Y == 0) & (X < 0), 1e-300, Y), 1.0, 0.0)

    def psi_o_tai_goc(i0, j0, gx, gy):
        # o [i0,i0+1]x[j0,j0+1] (chi so mang), dao ham song tuyen tai goc (gx,gy) in {0,1}
        u00, u10, u01, u11 = ux[i0, j0], ux[i0 + 1, j0], ux[i0, j0 + 1], ux[i0 + 1, j0 + 1]
        v00, v10, v01, v11 = uy[i0, j0], uy[i0 + 1, j0], uy[i0, j0 + 1], uy[i0 + 1, j0 + 1]
        dux_dx = ((u10 - u00) * (1 - gy) + (u11 - u01) * gy) / h
        dux_dy = ((u01 - u00) * (1 - gx) + (u11 - u10) * gx) / h
        duy_dx = ((v10 - v00) * (1 - gy) + (v11 - v01) * gy) / h
        duy_dy = ((v01 - v00) * (1 - gx) + (v11 - v10) * gx) / h
        return psi_amor(dux_dx, duy_dy, 0.5 * (dux_dy + duy_dx))

    NI, NJ = X.shape
    Hn = np.zeros_like(X)
    for i in range(1, NI - 1):
        for j in range(1, NJ - 1):
            if dg0:
                continue
            if s > 0:
                Hn[i, j] = psi_o_tai_goc(i, j, 0, 0)
            else:
                Hn[i, j] = psi_o_tai_goc(i - 1, j - 1, 1, 1)
    T = Bt = 0.0
    for i in range(1, NI - 2):
        for j in range(1, NJ - 2):
            xc, yc = X[i, j] + h / 2, Y[i, j] + h / 2
            if not (0 <= xc <= a and abs(yc) <= b):
                continue
            if dg0:
                val = psi_o_tai_goc(i, j, 0.5, 0.5)
            else:
                val = 0.25 * (Hn[i, j] + Hn[i + 1, j] + Hn[i, j + 1] + Hn[i + 1, j + 1])
            if yc > 0:
                T += val
            else:
                Bt += val
    return (T - Bt) / (T + Bt)


# ---------------------------------------------------------------- bang du doan
CA = [  # ten, A_H buoc 1 (vong 11-B), h/l, W (11-B)
    ("2D Q1 nx200 1 nhan, ghi de", -0.0996, 0.5, +1.000),
    ("2D Q1 nx200 1 nhan, ghi nguoc", +0.1000, 0.5, -1.000),
    ("2D Q1 nx400 4 nhan, ghi de", -0.0291, 0.25, +0.963),
    ("3D hex nx100 nz2 4 nhan, ghi de", +0.0933, 1.0, -0.550),
    ("3D hex ghi nguoc", -0.1037, 1.0, +0.783),
]


def deg(t):
    return math.degrees(t)


def main():
    print("== 0. Kiem truong Williams (bien dang phang, nu = 0.3)")
    for s in kiem_williams():
        print(s)
    print("== 1. MTS: cong thuc so voi tim so (max sigma_tt)")
    for b in (0.05, 0.1, 0.25, 0.5, 1.0, -0.3):
        print(f"  beta {b:+.2f}: cong thuc {deg(theta_mts_cong_thuc(b)):+8.3f} do, so {deg(theta_mts_so(b)):+8.3f} do,"
              f" G_max {deg(theta_gmax(b)):+8.3f}, PLS-lech {deg(theta_pls_lech(b)):+8.3f}")
    print("== 2. Kiem k_local voi C_ij cua Cotterell & Rice (1980, pt. 32)")
    w = 0.0
    for a in np.linspace(-1.2, 1.2, 25):
        C11 = (3*math.cos(a/2) + math.cos(3*a/2))/4; C12 = -3*(math.sin(a/2) + math.sin(3*a/2))/4
        C21 = (math.sin(a/2) + math.sin(3*a/2))/4;   C22 = (math.cos(a/2) + 3*math.cos(3*a/2))/4
        for KI, KII in ((1.0, 0.0), (0.0, 1.0), (1.0, 0.4)):
            kI, kII = k_local(np.array([a]), KI, KII)
            w = max(w, abs(kI[0] - (C11*KI + C12*KII)), abs(kII[0] - (C21*KI + C22*KII)))
    print(f"  max |k_local - C k| = {w:.2e}")
    al = alpha()
    print(f"== 3. alpha = dA/dbeta (Amor, R = 4l x 8l) = {al:+.5f}")
    for b in (0.1, 0.3, 0.5):
        print(f"  A(beta={b}) = {A_vung(b):+.5f}  (tuyen tinh {al*b:+.5f})")
    for (aa, bb) in ((2.0, 4.0), (4.0, 2.0), (8.0, 8.0)):
        print(f"  do nhay hinh vung: a={aa}, b={bb}: alpha = {(A_vung(1e-4,aa,bb)-A_vung(-1e-4,aa,bb))/2e-4:+.5f}")
    print("== 4. Mo hinh A: beta = A_H/alpha -> goc (do)")
    print(f"  {'ca':38s} {'A_H':>8s} {'beta':>8s} {'MTS':>8s} {'G_max':>8s} {'PLS-lech':>8s}")
    for ten, A, hl, W in CA:
        b = A / al
        print(f"  {ten:38s} {A:+8.4f} {b:+8.4f} {deg(theta_mts_cong_thuc(b)):+8.2f} {deg(theta_gmax(b)):+8.2f}"
              f" {deg(theta_pls_lech(b)):+8.2f}")
    print("== 5. Mo hinh B: A_H buoc 1 du doan tu Williams roi rac + ghi de (s=+1: o tren-phai thang)")
    for hl in (1.0, 0.5, 0.25, 0.125):
        Ap = A_ghi_de_williams(hl, s=+1)
        An = A_ghi_de_williams(hl, s=-1)
        Ad = A_ghi_de_williams(hl, dg0=True)
        print(f"  h/l = {hl:5.3f}: A_H ghi de {Ap:+.4f}, ghi nguoc {An:+.4f}, DG0 {Ad:+.1e}")
    print("  goc trang thai dung tan(theta) = -s W (h/2)/r*:")
    t200 = math.tan(math.radians(37.90))     # chi dung cho B1 (hieu chinh), xem doc
    rstar_B1 = 0.25 / t200
    for ten, A, hl, W in CA:
        s = 1.0
        tB0 = -math.degrees(math.atan(s * W * hl / 2 / 1.0))
        tB1 = -math.degrees(math.atan(s * W * hl / 2 / rstar_B1))
        print(f"  {ten:38s} B0 (r*=l) {tB0:+8.2f}   B1 (r*={rstar_B1:.3f} l) {tB1:+8.2f}")


# ---------------------------------------------------------------- 6. so voi do (SAU khi ghi truoc; doc muc "So sanh")
DO = {  # ten ca -> theta do (do): theta 6l bang 4 ban v9; ghi nguoc 2D: theta_tip 10l (khong co 6l, ca dung som)
    "2D Q1 nx200 1 nhan, ghi de": -37.90,
    "2D Q1 nx200 1 nhan, ghi nguoc": +49.1,
    "2D Q1 nx400 4 nhan, ghi de": -27.60,
    "3D hex nx100 nz2 4 nhan, ghi de": +44.62,
}


def khop(p, m, tuyet_doi=5.0):
    """cung dau VA (|p - m| <= tuyet_doi HOAC |p - m|/|m| <= 20 %); voi A_H chi dung 20 % (tuyet_doi = 0)."""
    if p * m <= 0:
        return "SAI DAU"
    return "khop" if (abs(p - m) <= tuyet_doi or abs(p - m) / abs(m) <= 0.2) else "lech"


def so_sanh():
    al = alpha()
    rB1 = 0.25 / math.tan(math.radians(37.90))
    print("== 6. So sanh (sau ghi truoc)")
    for ten, A, hl, W in CA:
        if ten not in DO:
            continue
        m = DO[ten]
        b = A / al
        pr = {"A-MTS": deg(theta_mts_cong_thuc(b)), "A-Gmax": deg(theta_gmax(b)), "A-PLS": deg(theta_pls_lech(b)),
              "B0": -math.degrees(math.atan(W * hl / 2)), "B1": -math.degrees(math.atan(W * hl / 2 / rB1))}
        print(f"  {ten:34s} do {m:+7.2f} | " + " | ".join(f"{k} {v:+7.2f} {khop(v, m)}" for k, v in pr.items()))
    print("  ti le theta/A_H:")
    r = []
    for ten, A, hl, W in CA:
        if ten in DO and "nguoc" not in ten:
            r.append(DO[ten] / A)
            print(f"    {ten:34s} theta/A_H = {DO[ten]/A:8.1f} do ; tan(theta)/(W h/l) = "
                  f"{math.tan(math.radians(DO[ten]))/(-W*hl):6.3f} ; r* hieu dung = {abs(W*hl/2/math.tan(math.radians(DO[ten]))):.3f} l")
    print(f"    max/min theta/A_H = {max(r)/min(r):.2f}")
    print("  A_H buoc 1 mo hinh B so do: nx200 xuoi -0.0397/-0.0996, nguoc +0.0312/+0.1000, nx400 "
          f"{-0.0208*0.963:+.4f}/-0.0291 ->", khop(-0.0397, -0.0996, 0), khop(0.0312, 0.1, 0), khop(-0.0208 * 0.963, -0.0291, 0))


if __name__ == "__main__":
    main()
    so_sanh()
