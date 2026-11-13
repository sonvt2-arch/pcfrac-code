"""
benchmark_P3d.py -- Miehe SENT dun thanh TAM MONG 3D, luoi luc dien Q1 (hex).

Muc dich (vong 8, nghien cuu P)
-------------------------------
O 2D (benchmark_A.py, tu giac Q1), luu H tren Q1 bang noi suy nut "phan tu ghi
sau thang" (--hspace cg1) lam vet nut Miehe SENT lech -35...-38 do, con DG0
(H hang tung phan tu) cho vet nut thang.  Cau hoi phan bien: o 3D, voi luoi
luc dien Q1 (moi nut trong co TAM phan tu ke thay vi bon), co con lech khong?

Hinh hoc va dieu kien bien
--------------------------
Tam 1 x 1 x t mm (t = nz * h, mac dinh nz = 2, t = 0.02 mm), luoi hex DEU
nx x nx x nz.  Khia sac nhu 2D: d = 1 tren MAT nut y = 0.5, x <= 0.5 (moi z).
Tai nhu 2D: u_y = 0 / u_y = ubar tren mat y = 0 / y = 1, va (--grip clamp)
u_x = 0 tren hai mat do.
Phuong z (--zbc):
  faces (mac dinh): u_z = 0 tren HAI mat z = 0 va z = t.  Day la cach gan bien
      dang phang nhat: voi Q1 dun deu theo z, nghiem dan hoi la nghiem bien
      dang phang 2D (eps_zz = 0), tai moi lop.  Nut giua (z = t/2, nz = 2) van
      tu do theo z va co dung 8 phan tu ke -- dung cau hinh can hoi.
  free: khong rang buoc z (ngoai mot day nut chong troi cung) -- gan ung suat
      phang; chi de chan doan, khong dung cho ca ghi truoc.
nz = 2 la toi thieu de lat giua co nut 8 phan tu ke (nz = 1 thi moi nut nam
tren mat bien, chi 4 phan tu ke nhu 2D).

Vat lieu, l, tai: giong .github/chay-noi-vong5-A.json (E 210000, nu 0.3, Gc 2.7,
AT2, l 0.01, n1 100, u1 0.001, umax 0.008, 800 buoc).
Phep tach --split amor la Amor 3D: K = lambda + 2 mu / 3, lech 3D (dung cho
vat the 3D; khac Amor "2D" cua benchmark_A.py, K2 = lambda + mu, nen F_pk khong
can trung 2D; buoc dau dan hoi thi trung vi g(d) sigma0 khong qua phep tach).

Luc bao cao: F = phan luc tong / t (N/mm), cung don vi voi 2D.

--hspace dg0: H hang tung phan tu (tam o).
--hspace cg1: H tren Q1 hex = V_d, psi+ noi suy vao nut (tam dinh moi o),
    nut dung chung lay gia tri o "ghi sau" -- dung ban 2D truoc khi sua.
--hspace quad (vong 9-T): H va psi+ tai diem cau phuong bac 2 cua hex (Gauss
    2x2x2), so hang g(d) H tich phan dung quy tac do -- nhu --hspace quad cua
    benchmark_A.py.  dg0/cg1 giu nguyen tung lenh.

Dau ra (tuong thich .github/workflows/chay-noi*.yml va measure_kink.py):
  damage_<model>_ell<ell>.xdmf/.h5 : d tren LAT GIUA z = t/2, luoi 2D nx x nx
  damage_<model>_ell<ell>_z0 / _zt : d tren hai mat z = 0 va z = t
  loaddisp_<out>.txt, dong "PEAK LOAD", diem kiem --ckpt/--resume nhu 2D.
"""
import argparse
import os as _os
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import fem, mesh, io
from dolfinx.fem import functionspace, Function, Constant, dirichletbc, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
import inspect as _inspect

_SNES_STYLE = "petsc_options_prefix" in _inspect.signature(
    NonlinearProblem.__init__).parameters
if not _SNES_STYLE:
    from dolfinx.nls.petsc import NewtonSolver
try:
    from dolfinx.fem import set_bc as _set_bc
except ImportError:
    from dolfinx.fem.petsc import set_bc as _set_bc

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["AT2", "WU"], default="AT2")
p.add_argument("--ell", type=float, default=0.01)
p.add_argument("--nx", type=int, default=100)
p.add_argument("--nz", type=int, default=2, help="so lop hex theo z; t = nz * h")
p.add_argument("--zbc", choices=["faces", "free"], default="faces")
p.add_argument("--steps", type=int, default=800)
p.add_argument("--maxstagg", type=int, default=200)
p.add_argument("--umax", type=float, default=0.008)
p.add_argument("--u1", type=float, default=0.001)
p.add_argument("--n1", type=int, default=100)
p.add_argument("--grip", choices=["clamp", "pin"], default="clamp")
p.add_argument("--split", choices=["amor", "full"], default="amor")
p.add_argument("--sharpnotch", action="store_true")
p.add_argument("--hspace", choices=["dg0", "cg1", "quad"], default="dg0")
p.add_argument("--load", choices=["tension"], default="tension")
p.add_argument("--ckpt", default=None)
p.add_argument("--resume", action="store_true")
p.add_argument("--E", type=float, default=210000.0)
p.add_argument("--nu", type=float, default=0.3)
p.add_argument("--Gc", type=float, default=2.7)
p.add_argument("--ft", type=float, default=3.5, help="chi dung cho WU")
p.add_argument("--out", default=None)
args = p.parse_args()
if not args.sharpnotch:
    raise SystemExit("benchmark_P3d chi ho tro --sharpnotch (khia sac nhu 2D)")

comm = MPI.COMM_WORLD
L = 1.0
E0, nu, Gc, ft, ell = args.E, args.nu, args.Gc, args.ft, args.ell
nx, nz = args.nx, args.nz
h = L / nx
t = nz * h

n2 = max(1, args.steps - args.n1)
_ramp = np.concatenate([np.linspace(0.0, args.u1, args.n1 + 1)[1:],
                        np.linspace(args.u1, args.umax, n2 + 1)[1:]])
if comm.rank == 0:
    print(f"  BENCHMARK P3D (tam mong 3D, hex Q1)  dolfinx {dolfinx.__version__}")
    print(f"model={args.model}  ell={ell} mm  nx={nx}  h={h:.4f} mm  h/ell={h/ell:.2f}"
          f"  nz={nz}  t={t:.4f} mm  --zbc {args.zbc}")
    print(f"  ramp: {args.n1} fine steps to {args.u1:.3e}, then {n2} steps to {args.umax:.3e}")

domain = mesh.create_box(comm, [np.array([0.0, 0.0, 0.0]), np.array([L, L, t])],
                         [nx, nx, nz], mesh.CellType.hexahedron)
if comm.rank == 0:
    print(f"  LUOI: hex Q1  so phan tu = {domain.topology.index_map(3).size_global}"
          f"  so dinh = {domain.topology.index_map(0).size_global}", flush=True)

V_u = functionspace(domain, ("Lagrange", 1, (3,)))
V_d = functionspace(domain, ("Lagrange", 1))
_QDEG = 2
if args.hspace == "quad":
    import basix, basix.ufl
    V_H = functionspace(domain, basix.ufl.quadrature_element(
        domain.topology.cell_name(), value_shape=(), degree=_QDEG))
    _QPTS = basix.make_quadrature(basix.CellType[domain.topology.cell_name()], _QDEG)[0]
else:
    V_H = functionspace(domain, ("DG", 0)) if args.hspace == "dg0" else V_d
if comm.rank == 0:
    print(f"  KHONG GIAN H: --hspace {args.hspace}"
          + ("   *** NOI SUY NUT GHI DE (8 phan tu ke) -- CHI DE DO TAT"
             if args.hspace == "cg1" else
             (f"   (H tai diem cau phuong, bac {_QDEG}, {len(_QPTS)} diem/o)"
              if args.hspace == "quad" else "")), flush=True)

u = Function(V_u, name="u")
d = Function(V_d, name="damage")
d_old = Function(V_d)
H = Function(V_H, name="history")


def _H_xyz():
    """Toa do bac tu do cua V_H; voi quad: toa do vat ly diem cau phuong."""
    if args.hspace != "quad":
        return V_H.tabulate_dof_coordinates()
    _tm = domain.topology.index_map(domain.topology.dim)
    _nc = _tm.size_local + _tm.num_ghosts
    _ex = fem.Expression(ufl.SpatialCoordinate(domain), _QPTS)
    _v = np.asarray(_ex.eval(domain, np.arange(_nc, dtype=np.int32))).reshape(_nc, -1, 3)
    _dl = np.asarray(V_H.dofmap.list)[:_nc]
    _out = np.zeros((len(H.x.array), 3))
    _out[_dl.ravel(), :] = _v.reshape(-1, 3)
    return _out

mu = E0 / (2 * (1 + nu))
lam = E0 * nu / ((1 + nu) * (1 - 2 * nu))
K3 = lam + 2.0 * mu / 3.0


def eps(v):
    return ufl.sym(ufl.grad(v))


def sigma0(v):
    return 2.0 * mu * eps(v) + lam * ufl.tr(eps(v)) * ufl.Identity(3)


def psi_pos(v):
    e = eps(v)
    if args.split == "full":
        return 0.5 * lam * ufl.tr(e) ** 2 + mu * ufl.inner(e, e)
    tr_p = 0.5 * (ufl.tr(e) + abs(ufl.tr(e)))
    e_dev = e - ufl.tr(e) / 3.0 * ufl.Identity(3)
    return 0.5 * K3 * tr_p ** 2 + mu * ufl.inner(e_dev, e_dev)


if comm.rank == 0:
    print(f"  PHEP TACH NANG LUONG: --split {args.split} (Amor 3D, K = lambda + 2mu/3)")

if args.model == "AT2":
    c_w = 2.0
    w = lambda dd: dd ** 2
    g = lambda dd: (1.0 - dd) ** 2 + 1e-7
    H_crit = Gc / (2.0 * ell)
else:
    a1 = 4.0 * E0 * Gc / (np.pi * ell * ft ** 2)
    a2 = -0.5
    c_w = np.pi
    w = lambda dd: 2.0 * dd - dd ** 2
    g = lambda dd: (1.0 - dd) ** 2 / ((1.0 - dd) ** 2 + a1 * dd * (1.0 + a2 * dd) + 1e-9)
    H_crit = 2.0 * Gc / (np.pi * ell * a1)


def _g_num(arr):
    a = np.clip(arr, 0.0, 1.0)
    if args.model == "AT2":
        return (1.0 - a) ** 2 + 1e-7
    num = (1.0 - a) ** 2
    return num / (num + a1 * a * (1.0 + a2 * a) + 1e-9)


# ------------------------------------------------------------- dieu kien bien
fdim = 2
f_bot = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 0.0))
f_top = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 1.0))
f_z = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.isclose(x[2], 0.0) | np.isclose(x[2], t))
Vs = [V_u.sub(i).collapse()[0] for i in range(3)]
zero = [Function(Vs[i]) for i in range(3)]
u_top = Function(Vs[1])


def _bc(val, i, facets, dim=fdim):
    return dirichletbc(val, locate_dofs_topological((V_u.sub(i), Vs[i]), dim, facets), V_u.sub(i))


bc_u = [_bc(zero[1], 1, f_bot), _bc(u_top, 1, f_top)]
if args.grip == "clamp":
    bc_u += [_bc(zero[0], 0, f_bot), _bc(zero[0], 0, f_top)]
else:
    c0 = mesh.locate_entities_boundary(
        domain, 0, lambda x: np.isclose(x[0], 0) & np.isclose(x[1], 0))
    bc_u.append(_bc(zero[0], 0, c0, 0))
if args.zbc == "faces":
    bc_u.append(_bc(zero[2], 2, f_z))
else:
    ez = mesh.locate_entities_boundary(
        domain, 0, lambda x: np.isclose(x[1], 0.0) & np.isclose(x[2], 0.0))
    bc_u.append(_bc(zero[2], 2, ez, 0))
if comm.rank == 0:
    print(f"  DIEU KIEN NGAM: --grip {args.grip};  z: --zbc {args.zbc}"
          + ("  (u_z = 0 tren hai mat z = 0, z = t: bien dang phang)"
             if args.zbc == "faces" else "  (mat z tu do -- chi chan doan)"), flush=True)


def notch(x):
    return (np.abs(x[1] - 0.5) < 0.4 * h) & (x[0] < 0.5 + 0.4 * h)


notch_dofs = fem.locate_dofs_geometrical(V_d, notch)
_nn = comm.allreduce(int(np.sum(notch_dofs < V_d.dofmap.index_map.size_local)), op=MPI.SUM)
if comm.rank == 0:
    print(f"  NOTCH DOFS (toan cuc) = {_nn}   (mong doi {(nx//2+1)*(nz+1)})", flush=True)

_xc = V_d.tabulate_dof_coordinates()
ahead = (_xc[:, 0] > 0.5 + 3.0 * ell) & (np.abs(_xc[:, 1] - 0.5) < 6.0 * ell)
_fm = np.ones(len(_xc), bool)
_fm[notch_dofs] = False
ahead &= _fm
_xH = _H_xyz()
ahead_H = (_xH[:, 0] > 0.5 + 3.0 * ell) & (np.abs(_xH[:, 1] - 0.5) < 6.0 * ell)
one = Function(V_d)
one.x.array[:] = 1.0
bc_d = [dirichletbc(one, notch_dofs)]

# ----------------------------------------------------------------- bai toan
v = ufl.TestFunction(V_u)
a_u = ufl.inner(g(d) * sigma0(ufl.TrialFunction(V_u)), eps(v)) * ufl.dx
L_u = ufl.inner(Constant(domain, PETSc.ScalarType((0.0, 0.0, 0.0))), v) * ufl.dx
prob_u = LinearProblem(a_u, L_u, bcs=bc_u, u=u, petsc_options_prefix="u_",
                       petsc_options={"ksp_type": "cg", "pc_type": "hypre",
                                      "ksp_rtol": 1e-10})

q = ufl.TestFunction(V_d)
Gc_c = Constant(domain, PETSc.ScalarType(Gc))
ell_c = Constant(domain, PETSc.ScalarType(ell))
if args.hspace == "quad":
    E_damage = (g(d) * H * ufl.dx(metadata={"quadrature_degree": _QDEG})
                + Gc_c / c_w * (w(d) / ell_c
                                + ell_c * ufl.inner(ufl.grad(d), ufl.grad(d))) * ufl.dx)
else:
    E_damage = (g(d) * H + Gc_c / c_w * (w(d) / ell_c
                + ell_c * ufl.inner(ufl.grad(d), ufl.grad(d)))) * ufl.dx
F_d = ufl.derivative(E_damage, d, q)
J_d = ufl.derivative(F_d, d, ufl.TrialFunction(V_d))
if _SNES_STYLE:
    _OPT_D = {"snes_type": "vinewtonrsls", "snes_rtol": 1e-8,
              "snes_atol": 1e-10, "snes_max_it": 40,
              "ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10}
    # Du phong LU (13/11, vong 9-T): doi loai KSP/PC tren SNES co san KHONG co
    # hieu luc (P3_quad_n4_r2: sau "doi sang LU" chan doan van in KSP=cg/hypre --
    # NonlinearProblem dat lai tuy chon cua no moi lan giai), nen du phong tao
    # MOT NonlinearProblem MOI (tien to rieng "dlu_") voi preonly + LU (MUMPS),
    # cung F_d, J_d, d, bc.  Ban 10/11 (doi tren SNES cu) vi vay chua tung chay
    # LU that; P3_dg0_n4_r2 khong bao gio vo nen khong bi anh huong.
    _OPT_LU = dict(_OPT_D, ksp_type="preonly", pc_type="lu",
                   pc_factor_mat_solver_type="mumps")
    _PD = [NonlinearProblem(F_d, d, bcs=bc_d, J=J_d, petsc_options_prefix="d_",
                            petsc_options=_OPT_D)]
    _lb_d, _ub_d = Function(V_d), Function(V_d)
    _LU = [False]
    _ndiag = [0]

    def _dung_LU(ly_do):
        _LU[0] = True
        _PD[0] = NonlinearProblem(F_d, d, bcs=bc_d, J=J_d, petsc_options_prefix="dlu_",
                                  petsc_options=_OPT_LU)
        if comm.rank == 0:
            print(f"  *** {ly_do}: BO GIAI d = NonlinearProblem MOI, preonly + LU (MUMPS)",
                  flush=True)

    def _chan_doan_d(r, nhan):
        """Chi in khi bo giai d vo (13/11, vong 9-T): loai KSP/PC dang dung,
        ly do KSP, so vong, va so phan tu khong huu han trong H, u, d."""
        if _ndiag[0] >= 8:
            return
        _ndiag[0] += 1
        sn = _PD[0].solver
        ksp = sn.getKSP()
        nf = [comm.allreduce(int(np.count_nonzero(~np.isfinite(f.x.array))), op=MPI.SUM)
              for f in (H, u, d)]
        hmin = comm.allreduce(float(H.x.array.min()) if H.x.array.size else 0.0, op=MPI.MIN)
        if comm.rank == 0:
            print(f"  CHAN DOAN d [{nhan}]: SNES={r} its={sn.getIterationNumber()} "
                  f"KSP={ksp.getType()}/{ksp.getPC().getType()} KSP_reason={ksp.getConvergedReason()} "
                  f"KSP_its={ksp.getIterationNumber()} khong_huu_han H/u/d={nf} H_min={hmin:.3e}",
                  flush=True)
    _ub_d.x.array[:] = 1.0

    def _giai_d():
        _PD[0].solver.setVariableBounds(_lb_d.x.petsc_vec, _ub_d.x.petsc_vec)
        _PD[0].solve()
        return _PD[0].solver.getConvergedReason()

    def solve_damage():
        _lb_d.x.array[:] = d.x.array
        r = _giai_d()
        if r <= 0:
            _chan_doan_d(r, "LU" if _LU[0] else "CG+hypre")
        if r <= 0 and not _LU[0]:
            # Lan vo dau: chuyen VINH VIEN sang LU, dat lai d ve diem xuat phat
            # va giai lai buoc nay.  Truoc lan vo dau: trung tung bit ban cu.
            _dung_LU(f"SNES={r}")
            d.x.array[:] = _lb_d.x.array
            d.x.scatter_forward()
            r = _giai_d()
            _chan_doan_d(r, "LU (giai lai, kiem loai KSP)")
        return r
else:
    prob_d = NonlinearProblem(F_d, d, bcs=bc_d, J=J_d)
    solver_d = NewtonSolver(comm, prob_d)
    solver_d.rtol, solver_d.atol, solver_d.max_it = 1e-8, 1e-10, 40

    def solve_damage():
        its, ok = solver_d.solve(d)
        return 1 if ok else -1

one_y = Function(Vs[1])
one_y.x.array[:] = 1.0
v_reac = Function(V_u)
_set_bc(v_reac.x.array, [_bc(one_y, 1, f_top)])
v_reac.x.scatter_forward()
react_form = fem.form(ufl.inner(g(d) * sigma0(u), eps(v_reac)) * ufl.dx)
# Chia t: Gamma/t la dien tich vet nut tren don vi day = do dai vet nut nhu 2D.
Gamma_form = fem.form((1.0 / c_w) * (w(d) / ell_c
                      + ell_c * ufl.inner(ufl.grad(d), ufl.grad(d))) * ufl.dx)
Gamma0 = None
if comm.rank == 0:
    print(f"  damage threshold H_crit = {H_crit:.4e}")

hist, n_bad, start_step = [], 0, 1

# ------------------------------------------------------------- diem kiem
_ck_dir = args.ckpt
_ck_funcs = (("u", u), ("d", d), ("d_old", d_old), ("H", H))


def _ck_xyz():
    return np.concatenate([V_u.tabulate_dof_coordinates().ravel(),
                           V_d.tabulate_dof_coordinates().ravel(),
                           _H_xyz().ravel()])


def _ck_path(slot):
    return _os.path.join(_ck_dir, f"r{comm.rank}of{comm.size}_s{slot}.npz")


def _ck_save(step):
    if not _ck_dir:
        return
    if comm.rank == 0:
        _os.makedirs(_ck_dir, exist_ok=True)
    comm.barrier()
    f = _ck_path(step % 2)
    np.savez(f + ".tmp.npz", step=step, n_bad=n_bad, lu=int(_SNES_STYLE and _LU[0]),
             gamma0=np.nan if Gamma0 is None else Gamma0,
             hist=np.array(hist, dtype=float).reshape(-1, 5), xyz=_ck_xyz(),
             **{k: fn.x.array for k, fn in _ck_funcs})
    _os.replace(f + ".tmp.npz", f)


def _ck_remap(z):
    nu_, nd, nH = len(u.x.array) // 3, len(d.x.array), len(H.x.array)
    if (len(z["u"]) != 3 * nu_ or len(z["d"]) != nd or len(z["H"]) != nH
            or z["xyz"].size != 3 * (nu_ + nd + nH)):
        return None
    xo = z["xyz"].reshape(-1, 3)
    xn = _ck_xyz().reshape(-1, 3)
    out = {}
    for ten, a, b in (("u", 0, nu_), ("d", nu_, nu_ + nd), ("H", nu_ + nd, nu_ + nd + nH)):
        ko, kn = np.round(xo[a:b], 10), np.round(xn[a:b], 10)
        io_, i_n = np.lexsort(ko.T[::-1]), np.lexsort(kn.T[::-1])
        if not np.array_equal(ko[io_], kn[i_n]) or (len(io_) > 1 and np.any(
                np.all(ko[io_][1:] == ko[io_][:-1], axis=1))):
            return None
        pm = np.empty(len(i_n), dtype=np.int64)
        pm[i_n] = io_
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
        if not comm.allreduce(1 if (_same or _perm is not None) else 0, op=MPI.MIN):
            raise SystemExit("PHAN HOACH KHAC diem kiem -- khong chay tiep")
        for k, fn in _ck_funcs:
            a = z[k]
            if _perm is not None:
                a = (a.reshape(-1, 3)[_perm["u"]].ravel() if k == "u"
                     else a[_perm["H" if k == "H" else "d"]])
            fn.x.array[:] = a
            fn.x.scatter_forward()
        n_bad = int(z["n_bad"])
        if _SNES_STYLE and "lu" in z.files and int(z["lu"]):
            _dung_LU("RESUME (diem kiem dang o LU)")
        Gamma0 = None if np.isnan(float(z["gamma0"])) else float(z["gamma0"])
        hist = [tuple(r) for r in z["hist"]]
    start_step = _s + 1
    if comm.rank == 0:
        print(f"  RESUME: chay tiep tu buoc {start_step} (diem kiem buoc {_s})", flush=True)

if args.hspace == "quad":
    _ip = _QPTS
else:
    _ip = V_H.element.interpolation_points
    if callable(_ip):
        _ip = _ip()
psi_expr = fem.Expression(psi_pos(u), _ip)
psi_fn = Function(V_H)

import time as _time
_t0 = _time.time()
reason = None
for step in range(start_step, args.steps + 1):
    ubar = float(_ramp[step - 1])
    u_top.x.array[:] = ubar
    for k in range(args.maxstagg):
        prob_u.solve()
        psi_fn.interpolate(psi_expr)
        psi_fn.x.scatter_forward()
        H.x.array[:] = np.maximum(H.x.array, psi_fn.x.array)
        H.x.scatter_forward()
        d_old.x.array[:] = d.x.array
        reason = solve_damage()
        d.x.scatter_forward()
        d.x.array[:] = np.clip(np.maximum(d.x.array, d_old.x.array), 0.0, 1.0)
        d.x.scatter_forward()
        dd = float(np.max(np.abs(_g_num(d.x.array) - _g_num(d_old.x.array)))) if d.x.array.size else 0.0
        if comm.allreduce(dd, op=MPI.MAX) < 1e-6:
            break
    if reason is not None and reason <= 0:
        n_bad += 1
    F = comm.allreduce(fem.assemble_scalar(react_form), op=MPI.SUM) / t
    dah = d.x.array[ahead]
    dmax = comm.allreduce(float(dah.max()) if dah.size else 0.0, op=MPI.MAX)
    Hah = H.x.array[ahead_H]
    Hmax = comm.allreduce(float(Hah.max()) if Hah.size else 0.0, op=MPI.MAX)
    Gamma = comm.allreduce(fem.assemble_scalar(Gamma_form), op=MPI.SUM) / t
    if Gamma0 is None:
        Gamma0 = Gamma
    a_len = Gamma - Gamma0
    hist.append((ubar, F, dmax, a_len, k + 1))
    if comm.rank == 0:
        print(f"  step {step:3d}  u = {ubar:.3e}  F = {F:8.4f}  d_tip = {dmax:.3f}  "
              f"a = {a_len:.4f}  H/Hc = {Hmax/H_crit:8.2f}  stagg = {k+1}  t = {_time.time()-_t0:.0f}s"
              + ("" if reason is None or reason > 0 else f"  SNES={reason}"), flush=True)
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

if comm.rank == 0:
    hh = np.array(hist)
    Fabs = np.abs(hh[:, 1])
    peak, i_peak = Fabs.max(), int(np.argmax(Fabs))
    tag = args.out or f"P3d_{args.model}_ell{ell:g}"
    np.savetxt(f"loaddisp_{tag}.txt", hh,
               header="u_mm  F_N_per_mm(=F_tong/t)  dmax  crack_extension_mm  staggered_its")
    print("\n" + "=" * 60)
    print(f"  model {args.model}   ell = {ell} mm   3D hex nx={nx} nz={nz} t={t:g}")
    print(f"  PEAK LOAD = {peak:.4f} N/mm   at step {i_peak+1} of {len(Fabs)}")
    tail = Fabs[-1] / peak if peak > 0 else 1.0
    print(f"  final load = {100*tail:.1f} % of peak ; final d ahead = {hh[-1,2]:.3f} ; "
          f"crack extension = {hh[-1,3]:.4f} mm")
    if n_bad:
        print(f"  *** {n_bad} of {len(Fabs)} steps had a failed damage solve")
    print(f"  wrote loaddisp_{tag}.txt")
    print("=" * 60, flush=True)

# ------------------------------------------------- lat 2D: giua va hai mat z
# Gom d (bac tu do SO HUU) ve nhan 0, chon nut tren mat z = z0, dat vao luoi 2D
# nx x nx (Q1) tren COMM_SELF qua chi so luoi (i, j) = round(x/h, y/h), roi ghi
# XDMF dung dang cua 2D -- measure_kink.py doc khong doi.
_n_own = V_d.dofmap.index_map.size_local
_parts = comm.gather((_xc[:_n_own].copy(), d.x.array[:_n_own].copy()), root=0)
if comm.rank == 0:
    X = np.concatenate([a for a, _ in _parts])
    D = np.concatenate([b for _, b in _parts])
    kk = np.rint(X[:, 2] / h).astype(int)
    ij = np.rint(X[:, :2] / h).astype(int)
    m2 = mesh.create_unit_square(MPI.COMM_SELF, nx, nx, mesh.CellType.quadrilateral)
    V2 = functionspace(m2, ("Lagrange", 1))
    x2 = V2.tabulate_dof_coordinates()
    i2 = np.rint(x2[:, :2] / h).astype(int)
    key2 = i2[:, 0] * (nx + 1) + i2[:, 1]
    base = f"damage_{args.model}_ell{ell:g}"
    for k_sl, suf in ((nz // 2, ""), (0, "_z0"), (nz, "_zt")):
        s = kk == k_sl
        grid = np.full((nx + 1) * (nx + 1), np.nan)
        grid[ij[s, 0] * (nx + 1) + ij[s, 1]] = D[s]
        d2 = Function(V2, name="damage")
        d2.x.array[:] = grid[key2]
        if np.isnan(d2.x.array).any():
            print(f"  *** lat k={k_sl}: thieu {int(np.isnan(d2.x.array).sum())} nut")
            d2.x.array[np.isnan(d2.x.array)] = 0.0
        with io.XDMFFile(MPI.COMM_SELF, base + suf + ".xdmf", "w") as xf:
            xf.write_mesh(m2)
            xf.write_function(d2)
        print(f"  LAT z = {k_sl*h:.4f} ({'giua' if suf == '' else suf}): {int(s.sum())} nut"
              f" -> {base + suf}.xdmf ; so nut d>0.9: {int((d2.x.array > 0.9).sum())}", flush=True)
    if nz % 2:
        print("  *** nz le: lat 'giua' la lop nut k = nz//2, khong dung z = t/2")
