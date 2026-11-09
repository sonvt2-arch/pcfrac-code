"""
Benchmark O -- tam chu L (vet nut CONG), phase-field, DOLFINx 0.11.

Cau hoi (vong 8, 09/11/2026): do lech cua cach luu H nut Q1 (--hspace cg1, ghi
de "phan tu ghi sau thang") so voi DG0, thay tren Miehe SENT (vet thang,
35-38 do), co xay ra tren bai toan vet nut CONG khong.

Hinh hoc, dieu kien bien, vat lieu: tam chu L cua Winkler (2001) theo mo ta
trong nguon mo arXiv:2603.12210v3 (Bromley & Lipton), muc 8.4, Hinh 15, Bang 3:
  canh ngoai 500 mm, khoet 250 x 250 mm, day 100 mm, diem dat chuyen vi cach
  goc lom 220 mm (cach mep phai 30 mm), mep duoi ngam hai phuong;
  gia tri thi nghiem Winkler: E = 25.85 GPa, nu = 0.18; f_t = 2.7 MPa;
  G_f = 90 N/m = 0.09 N/mm.
Huong dat: mien [0,500]^2 bo [250,500]x[0,250]; mep y = 0 (0 <= x <= 250)
ngam; u_y tang dan dat tren doan mat duoi canh tay ngang y = 250,
|x - 470| <= --lpatch/2.  Goc lom tai (250, 250).

Mo hinh: AT2 (mac dinh, nhu benchmark_A Miehe) hoac WU; --split amor; ung suat
phang (day 100 mm).  Luc F tinh tren 1 mm day (N/mm); nhan 100 ra N cua mau.

Ma nay CHEP cac khoi cua code/benchmark_A.py (psi_pos, g/w, bo giai hu hong
vinewtonrsls, vong so le, diem kiem --ckpt/--resume, cach luu dg0/cg1) vi
benchmark_A.py la tap lenh chay tu tren xuong (khong import duoc ham).
benchmark_A.py KHONG doi.  Chi --hspace dg0 | cg1.

Xuat: loaddisp_<tag>.txt, dong "PEAK LOAD", damage_<model>_ell<ell>.xdmf (d),
dfield_<tag>.npz (toa do nut + d, gom ve nhan 0) cho code/do_O.py.
Don vi: mm, N, MPa.
"""
import argparse
import os as _os
import inspect as _inspect
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import fem, mesh, io
from dolfinx.fem import functionspace, Function, Constant, dirichletbc, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem

_SNES_STYLE = "petsc_options_prefix" in _inspect.signature(NonlinearProblem.__init__).parameters
if not _SNES_STYLE:
    from dolfinx.nls.petsc import NewtonSolver
try:
    from dolfinx.fem import set_bc as _set_bc
except ImportError:
    from dolfinx.fem.petsc import set_bc as _set_bc

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["AT2", "WU"], default="AT2")
p.add_argument("--ell", type=float, default=10.0, help="mm")
p.add_argument("--nx", type=int, default=250, help="so o tren canh 500 mm (h = 500/nx)")
p.add_argument("--steps", type=int, default=300)
p.add_argument("--n1", type=int, default=100)
p.add_argument("--u1", type=float, default=0.5, help="mm, het chang min")
p.add_argument("--umax", type=float, default=1.5, help="mm")
p.add_argument("--maxstagg", type=int, default=200)
p.add_argument("--split", choices=["amor", "spectral", "full"], default="amor")
p.add_argument("--hspace", choices=["dg0", "cg1"], default="dg0")
p.add_argument("--plane", choices=["stress", "strain"], default="stress")
p.add_argument("--lpatch", type=float, default=20.0, help="mm, be rong doan dat chuyen vi")
p.add_argument("--E", type=float, default=25850.0)
p.add_argument("--nu", type=float, default=0.18)
p.add_argument("--Gc", type=float, default=0.09)
p.add_argument("--ft", type=float, default=2.7)
p.add_argument("--ckpt", default=None)
p.add_argument("--resume", action="store_true")
p.add_argument("--out", default=None)
args = p.parse_args()

comm = MPI.COMM_WORLD
E0, nu, Gc, ft, ell = args.E, args.nu, args.Gc, args.ft, args.ell
A_, B_ = 500.0, 250.0                 # canh ngoai, canh khoet
XC, YC = B_, B_                       # goc lom
XL = 470.0                            # diem dat tai
nx = args.nx
h = A_ / nx
if abs(B_ / h - round(B_ / h)) > 1e-9 or abs(XL / h - round(XL / h)) > 1e-9:
    raise SystemExit("nx phai de 250/h va 470/h la so nguyen (vd 250, 125, 500)")

n2 = max(1, args.steps - args.n1)
_ramp = np.concatenate([np.linspace(0.0, args.u1, args.n1 + 1)[1:],
                        np.linspace(args.u1, args.umax, n2 + 1)[1:]])
if comm.rank == 0:
    print(f"BENCHMARK O (tam chu L)  model={args.model}  ell={ell} mm  nx={nx}  h={h:.3f} mm  "
          f"ell/h={ell/h:.2f}  plane={args.plane}", flush=True)
    print(f"  ramp: {args.n1} buoc toi {args.u1} mm, roi {n2} buoc toi {args.umax} mm", flush=True)

# ------------------------------------------------------------------ luoi
_full = mesh.create_rectangle(comm, [np.array([0.0, 0.0]), np.array([A_, A_])], [nx, nx],
                              mesh.CellType.quadrilateral)
_tdim = _full.topology.dim
# Giu o co TAM ngoai phan khoet (x > 250, y < 250).
_ex = fem.Expression(ufl.SpatialCoordinate(_full), np.array([[0.5, 0.5]]))
_nc_all = _full.topology.index_map(_tdim).size_local
_cen = np.asarray(_ex.eval(_full, np.arange(_nc_all, dtype=np.int32))).reshape(_nc_all, 2)
_keep = np.nonzero(~((_cen[:, 0] > B_) & (_cen[:, 1] < B_)))[0].astype(np.int32)
domain = mesh.create_submesh(_full, _tdim, _keep)[0]
del _full
if comm.rank == 0:
    print(f"  LUOI chu L: so phan tu = {domain.topology.index_map(_tdim).size_global}  "
          f"so dinh = {domain.topology.index_map(0).size_global}", flush=True)

V_u = functionspace(domain, ("Lagrange", 1, (2,)))
V_d = functionspace(domain, ("Lagrange", 1))
V_H = functionspace(domain, ("DG", 0)) if args.hspace == "dg0" else V_d
if comm.rank == 0:
    print(f"  KHONG GIAN H: --hspace {args.hspace}"
          + ("   *** NUT Q1 GHI DE (ban truoc khi sua)" if args.hspace == "cg1" else ""), flush=True)
    print(f"  PHEP TACH NANG LUONG: --split {args.split}", flush=True)

u = Function(V_u, name="u")
d = Function(V_d, name="damage")
d_old = Function(V_d)
H = Function(V_H, name="history")

mu = E0 / (2 * (1 + nu))
lam = E0 * nu / (1.0 - nu ** 2) if args.plane == "stress" else E0 * nu / ((1 + nu) * (1 - 2 * nu))
K2 = lam + mu


def eps(v):
    return ufl.sym(ufl.grad(v))


def sigma0(v):
    return 2.0 * mu * eps(v) + lam * ufl.tr(eps(v)) * ufl.Identity(2)


def psi_pos(v):
    # Chep tu benchmark_A.py (psi_pos): Amor voi K = lambda + mu (2D).
    e = eps(v)
    tr_p = 0.5 * (ufl.tr(e) + abs(ufl.tr(e)))
    e_dev = e - ufl.tr(e) / 2.0 * ufl.Identity(2)
    if args.split == "spectral":
        _q = ufl.sqrt(((e[0, 0] - e[1, 1]) / 2.0) ** 2 + e[0, 1] ** 2 + 1.0e-30)
        _e1 = ufl.tr(e) / 2.0 + _q
        _e2 = ufl.tr(e) / 2.0 - _q
        return 0.5 * lam * tr_p ** 2 + mu * ((0.5 * (_e1 + abs(_e1))) ** 2 + (0.5 * (_e2 + abs(_e2))) ** 2)
    if args.split == "amor":
        return 0.5 * K2 * tr_p ** 2 + mu * ufl.inner(e_dev, e_dev)
    return 0.5 * lam * ufl.tr(e) ** 2 + mu * ufl.inner(e, e)


if args.model == "AT2":
    c_w = 2.0
    def w(dd):
        return dd ** 2
    def g(dd):
        return (1.0 - dd) ** 2 + 1e-7
else:
    a1 = 4.0 * E0 * Gc / (np.pi * ell * ft ** 2)
    a2, a3 = -0.5, 0.0
    c_w = np.pi
    def w(dd):
        return 2.0 * dd - dd ** 2
    def g(dd):
        num = (1.0 - dd) ** 2
        return num / (num + a1 * dd * (1.0 + a2 * dd + a2 * a3 * dd ** 2) + 1e-9)


def _g_num(arr):
    a = np.clip(arr, 0.0, 1.0)
    if args.model == "AT2":
        return (1.0 - a) ** 2 + 1e-7
    num = (1.0 - a) ** 2
    return num / (num + a1 * a * (1.0 + a2 * a + a2 * a3 * a ** 2) + 1e-9)


# ------------------------------------------------------- dieu kien bien
fdim = _tdim - 1
f_bot = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 0.0))
_hl = 0.5 * args.lpatch + 1e-6 * A_
f_load = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.isclose(x[1], B_) & (np.abs(x[0] - XL) <= _hl) & (x[0] > B_ + 1e-6))
V_uy, _ = V_u.sub(1).collapse()
V_ux, _ = V_u.sub(0).collapse()
zy = Function(V_uy); zy.x.array[:] = 0.0
zx = Function(V_ux); zx.x.array[:] = 0.0
u_load = Function(V_uy)
bc_u = [dirichletbc(zx, locate_dofs_topological((V_u.sub(0), V_ux), fdim, f_bot), V_u.sub(0)),
        dirichletbc(zy, locate_dofs_topological((V_u.sub(1), V_uy), fdim, f_bot), V_u.sub(1)),
        dirichletbc(u_load, locate_dofs_topological((V_u.sub(1), V_uy), fdim, f_load), V_u.sub(1))]
_nfl = comm.allreduce(len(f_load), op=MPI.SUM)
if comm.rank == 0:
    print(f"  TAI: u_y tren {_nfl} canh, y = {B_}, |x - {XL}| <= {0.5*args.lpatch} mm; "
          f"ngam mep y = 0", flush=True)

# ------------------------------------------------------------ hai bai toan
v = ufl.TestFunction(V_u)
du = ufl.TrialFunction(V_u)
prob_u = LinearProblem(ufl.inner(g(d) * sigma0(du), eps(v)) * ufl.dx,
                       ufl.inner(Constant(domain, PETSc.ScalarType((0.0, 0.0))), v) * ufl.dx,
                       bcs=bc_u, u=u, petsc_options_prefix="u_",
                       petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10})
q = ufl.TestFunction(V_d)
Gc_c = Constant(domain, PETSc.ScalarType(Gc))
ell_c = Constant(domain, PETSc.ScalarType(ell))
# snes_atol theo THANG cua bai toan (09/11, sau thu khoi): 1e-10 tuyet doi cua
# benchmark_A (mien 1 mm^2) khong dat duoc o day -> SNES=-5 moi buoc tu buoc 1.
# Thang phan du: (Gc/ell) * sqrt(dien tich mien); atol = 1e-9 lan thang do.
_area = comm.allreduce(fem.assemble_scalar(fem.form(Constant(domain, PETSc.ScalarType(1.0)) * ufl.dx)),
                       op=MPI.SUM)
_snes_atol = 1e-9 * Gc / ell * np.sqrt(_area)
if comm.rank == 0:
    print(f"  dien tich = {_area:.1f} mm^2 ; snes_atol = {_snes_atol:.3e}", flush=True)
E_damage = (g(d) * H + Gc_c / c_w * (w(d) / ell_c + ell_c * ufl.inner(ufl.grad(d), ufl.grad(d)))) * ufl.dx
F_d = ufl.derivative(E_damage, d, q)
J_d = ufl.derivative(F_d, d, ufl.TrialFunction(V_d))
if _SNES_STYLE:
    prob_d = NonlinearProblem(F_d, d, bcs=[], J=J_d, petsc_options_prefix="d_",
                              petsc_options={"snes_type": "vinewtonrsls", "snes_rtol": 1e-8,
                                             "snes_atol": _snes_atol, "snes_max_it": 40,
                                             "ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10})
    _diag = [0, 0.0]
    _lb_d = Function(V_d)
    _ub_d = Function(V_d); _ub_d.x.array[:] = 1.0

    def solve_damage():
        _lb_d.x.array[:] = d.x.array
        prob_d.solver.setVariableBounds(_lb_d.x.petsc_vec, _ub_d.x.petsc_vec)
        prob_d.solve()
        try:
            r = prob_d.solver.getConvergedReason()
            _diag[:] = [prob_d.solver.getIterationNumber(), prob_d.solver.getFunctionNorm()]
            return r
        except Exception:
            return None
else:
    prob_d = NonlinearProblem(F_d, d, bcs=[], J=J_d)
    solver_d = NewtonSolver(comm, prob_d)
    solver_d.rtol, solver_d.atol, solver_d.max_it = 1e-8, 1e-10, 40

    def solve_damage():
        its, ok = solver_d.solve(d)
        return 1 if ok else -1

# luc phan: truong ao = 1 theo y tren doan dat tai
one_y = Function(V_uy); one_y.x.array[:] = 1.0
bc_reac = dirichletbc(one_y, locate_dofs_topological((V_u.sub(1), V_uy), fdim, f_load), V_u.sub(1))
v_reac = Function(V_u); v_reac.x.array[:] = 0.0
_set_bc(v_reac.x.array, [bc_reac]); v_reac.x.scatter_forward()
react_form = fem.form(ufl.inner(g(d) * sigma0(u), eps(v_reac)) * ufl.dx)
Gamma_form = fem.form((1.0 / c_w) * (w(d) / ell_c + ell_c * ufl.inner(ufl.grad(d), ufl.grad(d))) * ufl.dx)

# chi bao: d lon nhat trong vanh 2ell..6ell quanh goc lom (khoi dau vet nut)
_xc = V_d.tabulate_dof_coordinates()
_r = np.hypot(_xc[:, 0] - XC, _xc[:, 1] - YC)
ring = (_r > 2.0 * ell) & (_r < 6.0 * ell)

_ip = V_H.element.interpolation_points
if callable(_ip):
    _ip = _ip()
psi_expr = fem.Expression(psi_pos(u), _ip)
psi_fn = Function(V_H)

hist = []
n_bad = 0
start_step = 1
Gamma0 = None

# --------------------------------------------- diem kiem (chep benchmark_A.py)
_ck_dir = args.ckpt
_ck_funcs = (("u", u), ("d", d), ("d_old", d_old), ("H", H))


def _ck_xyz():
    return np.concatenate([V_u.tabulate_dof_coordinates().ravel(),
                           V_d.tabulate_dof_coordinates().ravel(),
                           V_H.tabulate_dof_coordinates().ravel()])


def _ck_path(slot):
    return _os.path.join(_ck_dir, f"r{comm.rank}of{comm.size}_s{slot}.npz")


def _ck_save(step):
    if not _ck_dir:
        return
    if comm.rank == 0:
        _os.makedirs(_ck_dir, exist_ok=True)
    comm.barrier()
    f = _ck_path(step % 2)
    tmp = f + ".tmp.npz"
    np.savez(tmp, step=step, n_bad=n_bad, gamma0=np.nan if Gamma0 is None else Gamma0,
             hist=np.array(hist, dtype=float).reshape(-1, 5), xyz=_ck_xyz(),
             **{k: fn.x.array for k, fn in _ck_funcs})
    _os.replace(tmp, f)


def _ck_remap(z):
    nu_, nd, nH = len(u.x.array) // 2, len(d.x.array), len(H.x.array)
    if (len(z["u"]) != 2 * nu_ or len(z["d"]) != nd or len(z["H"]) != nH
            or z["xyz"].size != 3 * (nu_ + nd + nH)):
        return None
    xo = z["xyz"].reshape(-1, 3); xn = _ck_xyz().reshape(-1, 3)
    out = {}
    for ten, a, b in (("u", 0, nu_), ("d", nu_, nu_ + nd), ("H", nu_ + nd, nu_ + nd + nH)):
        ko = np.round(xo[a:b], 8); kn = np.round(xn[a:b], 8)
        io_ = np.lexsort(ko.T[::-1]); i_n = np.lexsort(kn.T[::-1])
        if not np.array_equal(ko[io_], kn[i_n]) or (len(io_) > 1 and
                np.any(np.all(ko[io_][1:] == ko[io_][:-1], axis=1))):
            return None
        pm = np.empty(len(i_n), dtype=np.int64); pm[i_n] = io_
        out[ten] = pm
    return out


if args.resume:
    if not _ck_dir:
        raise SystemExit("--resume can --ckpt")
    _have = {}
    for slot in (0, 1):
        try:
            with np.load(_ck_path(slot)) as z:
                _have[int(z["step"])] = slot
        except (OSError, KeyError, ValueError):
            pass
    _steps = comm.allgather(sorted(_have))
    _common = set(_steps[0]).intersection(*map(set, _steps[1:])) if _steps else set()
    if not _common:
        raise SystemExit("KHONG CO DIEM KIEM CHUNG cho moi nhan -- chay lai tu dau")
    _s = max(_common)
    with np.load(_ck_path(_have[_s])) as z:
        _same = z["xyz"].shape == _ck_xyz().shape and np.array_equal(z["xyz"], _ck_xyz())
        _perm = None if _same else _ck_remap(z)
        _ok = comm.allreduce(1 if (_same or _perm is not None) else 0, op=MPI.MIN)
        if not _ok:
            raise SystemExit("PHAN HOACH KHAC diem kiem -- khong chay tiep")
        for k, fn in _ck_funcs:
            a = z[k]
            if _perm is not None:
                a = (a.reshape(-1, 2)[_perm["u"]].ravel() if k == "u"
                     else a[_perm["H" if k == "H" else "d"]])
            fn.x.array[:] = a
            fn.x.scatter_forward()
        n_bad = int(z["n_bad"])
        Gamma0 = None if np.isnan(float(z["gamma0"])) else float(z["gamma0"])
        hist = [tuple(r) for r in z["hist"]]
    start_step = _s + 1
    if comm.rank == 0:
        print(f"  RESUME: chay tiep tu buoc {start_step} (diem kiem buoc {_s})", flush=True)

# ------------------------------------------------------------------ vong tai
for step in range(start_step, args.steps + 1):
    ubar = float(_ramp[step - 1])
    u_load.x.array[:] = ubar
    reason = None
    for k in range(args.maxstagg):
        prob_u.solve()
        psi_fn.interpolate(psi_expr)          # dg0: tam o; cg1: nut, phan tu ghi sau thang
        psi_fn.x.scatter_forward()
        H.x.array[:] = np.maximum(H.x.array, psi_fn.x.array)
        H.x.scatter_forward()
        d_old.x.array[:] = d.x.array
        reason = solve_damage()
        d.x.scatter_forward()
        d.x.array[:] = np.clip(np.maximum(d.x.array, d_old.x.array), 0.0, 1.0)
        d.x.scatter_forward()
        gn = _g_num(d.x.array); go = _g_num(d_old.x.array)
        dd = comm.allreduce(float(np.max(np.abs(gn - go))) if gn.size else 0.0, op=MPI.MAX)
        if dd < 1e-6:
            break
    if reason is not None and reason <= 0:
        n_bad += 1
    F = comm.allreduce(fem.assemble_scalar(react_form), op=MPI.SUM)
    dr = d.x.array[ring]
    dmax = comm.allreduce(float(dr.max()) if dr.size else 0.0, op=MPI.MAX)
    Gamma = comm.allreduce(fem.assemble_scalar(Gamma_form), op=MPI.SUM)
    if Gamma0 is None:
        Gamma0 = Gamma
    hist.append((ubar, F, dmax, Gamma - Gamma0, k + 1))
    if comm.rank == 0:
        print(f"  step {step:3d}  u = {ubar:.4e}  F = {F:9.4f}  d_goc = {dmax:.3f}  "
              f"Gamma = {Gamma:.3f}  stagg = {k+1}"
              + ("" if reason is None or reason > 0 else f"  SNES={reason}")
              + (f"  [snes its={_diag[0]} |F|={_diag[1]:.2e}]" if _SNES_STYLE and
                 (step <= 3 or (reason is not None and reason <= 0)) else ""), flush=True)
    _ck_save(step)
    Fs = np.abs(np.array([r[1] for r in hist]))
    if len(Fs) > 6 and Fs.max() > 0 and Fs[-1] < 0.25 * Fs.max():
        if comm.rank == 0:
            print("  load has fallen below 25 % of peak -- stopping")
        break
    if n_bad > 20:
        if comm.rank == 0:
            print(f"  {n_bad} failed damage solves -- stopping")
        break

tag = args.out or f"O_{args.hspace}"
_nown = V_d.dofmap.index_map.size_local
_parts = comm.gather((_xc[:_nown, :2].copy(), d.x.array[:_nown].copy()), root=0)
if comm.rank == 0:
    hh = np.array(hist)
    Fabs = np.abs(hh[:, 1])
    i_pk = int(np.argmax(Fabs))
    np.savetxt(f"loaddisp_{tag}.txt", hh,
               header="u_mm  F_N_per_mm  dmax_vanh_goc  Gamma_minus_Gamma0_mm  staggered_its")
    X = np.concatenate([a for a, _ in _parts]); D = np.concatenate([b for _, b in _parts])
    np.savez(f"dfield_{tag}.npz", x=X, d=D, ell=ell, h=h, hspace=args.hspace,
             nprocs=comm.size, F_pk=Fabs[i_pk], u_pk=hh[i_pk, 0])
    print("\n" + "=" * 60)
    print(f"  BENCHMARK O  model {args.model}  ell = {ell} mm  hspace = {args.hspace}  nhan = {comm.size}")
    print(f"  PEAK LOAD = {Fabs[i_pk]:.4f} N/mm   at step {i_pk+1} of {len(Fabs)}   "
          f"u = {hh[i_pk,0]:.4e} mm   (x 100 mm day = {Fabs[i_pk]*0.1:.3f} kN)")
    print(f"  final load = {100*Fabs[-1]/Fabs[i_pk]:.1f} % of peak ; Gamma - Gamma0 = {hh[-1,3]:.2f} mm")
    if n_bad:
        print(f"  *** {n_bad} of {len(Fabs)} steps had a failed damage solve")
    print(f"  wrote loaddisp_{tag}.txt dfield_{tag}.npz")
    print("=" * 60, flush=True)

with io.XDMFFile(comm, f"damage_{args.model}_ell{ell:g}.xdmf", "w") as xf:
    xf.write_mesh(domain)
    xf.write_function(d)
