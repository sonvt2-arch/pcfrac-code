# Ported copy of original.py (forum post #19720/8, author Zylora126, accessed 28/10/2026).
# Source: https://fenicsproject.discourse.group/t/challenges-in-reproducing-phase-field-fracture-benchmark-unphysical-crack-trajectory-and-mesh-refinement-behavior/19720
# Reproduced for research; unmodified except where marked:
#   "# PORT:"   changes needed to run on dolfinx/dolfinx:stable (DOLFINx 0.11, headless CI)
#   "# OUTPUT:" added output (final damage/history fields, load-displacement file); no effect on the solve
# Physics, mesh, parameters and loading are those of original.py.

"""
Uses a globally uniform fine mesh and
two-stage fixed displacement increments (paper PFM_C approach).
"""

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh, io, plot, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import os  # OUTPUT:
OUT_TAG = os.environ.get("OUT_TAG", "port")  # OUTPUT: tag for output file names

# ==========================================
# 1. Physical tags and parameters
# ==========================================
TAG_DOMAIN = 1
TAG_BOTTOM = 2
TAG_RIGHT = 3
TAG_TOP = 4
TAG_LEFT = 5
TAG_CRACK = 6

# Material parameters (mm, kN)
E_val = 210.0          # kN/mm^2
nu_val = 0.3
gc_val = 2.7e-3        # kN/mm (2.7 N/mm)
l0_val = 0.015          # mm

# Mesh: uniform fine mesh, no AMR
lc_initial = 0.005     # = l0/2, matches paper PFM_C (~92k cells)
# To reduce runtime, try lc_initial = 0.01 (~23k cells)

# Fixed two-stage displacement increments (paper PFM_C)
delta_u_stage1 = 1.0e-5    # mm, up to u = 5e-3
delta_u_stage2 = 1.0e-6    # mm, after u = 5e-3
u_stage_switch = 5.0e-3
u_max = 0.007

# Staggered solver
tol_staggered = 1.0e-6
iter_max = 15


# ==========================================
# 2. Mesh generation
# ==========================================
def generate_initial_mesh(L=1.0, W=1.0, a=0.5, lc=lc_initial):
    """
    Symmetric mesh via OCC copy+mirror+fragment.
    Bottom half mirrored about y=W/2, then fragmented for shared interface.
    Guaranteed node symmetry above/below y=0.5.
    """
    import gmsh
    from dolfinx.io.gmsh import model_to_mesh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("symmetric_mirror")

    eps = 1e-6
    y_mid = W / 2.0

    # 1. Bottom half rectangle [0,L] x [0, y_mid]
    s_bottom = gmsh.model.occ.addRectangle(0, 0, 0, L, y_mid)
    gmsh.model.occ.synchronize()

    # 2. Copy + mirror about y=y_mid
    copied = gmsh.model.occ.copy([(2, s_bottom)])
    gmsh.model.occ.mirror(copied, 0, 1, 0, -y_mid)
    gmsh.model.occ.synchronize()

    # 3. Fragment to merge overlapping curves on y=y_mid
    s_top = copied[0][1]
    gmsh.model.occ.fragment([(2, s_bottom)], [(2, s_top)])
    gmsh.model.occ.synchronize()

    # 4. Identify boundaries via bounding box
    all_surfaces = gmsh.model.getEntities(2)

    bottom_curves = gmsh.model.getEntitiesInBoundingBox(
        -eps, -eps, -eps, L+eps, eps, eps, 1)
    right_curves = gmsh.model.getEntitiesInBoundingBox(
        L-eps, -eps, -eps, L+eps, W+eps, eps, 1)
    top_curves = gmsh.model.getEntitiesInBoundingBox(
        -eps, W-eps, -eps, L+eps, W+eps, eps, 1)
    left_all = gmsh.model.getEntitiesInBoundingBox(
        -eps, -eps, -eps, eps, W+eps, eps, 1)
    # Crack: all curves on y=y_mid (interface), used for physical group
    crack_curves = gmsh.model.getEntitiesInBoundingBox(
        -eps, y_mid-eps, -eps, L+eps, y_mid+eps, eps, 1)

    # Filter left curves: exclude interface curves
    left_curves = []
    for dim, tag in left_all:
        bbox = gmsh.model.getBoundingBox(dim, tag)
        my = (bbox[1] + bbox[4]) / 2
        if abs(my - y_mid) > eps * 10:
            left_curves.append((dim, tag))

    # 5. Physical groups
    gmsh.model.addPhysicalGroup(
        2, [s[1] for s in all_surfaces], TAG_DOMAIN, name="Domain")
    gmsh.model.addPhysicalGroup(
        1, [c[1] for c in bottom_curves], TAG_BOTTOM, name="Bottom")
    gmsh.model.addPhysicalGroup(
        1, [c[1] for c in right_curves], TAG_RIGHT, name="Right")
    gmsh.model.addPhysicalGroup(
        1, [c[1] for c in top_curves], TAG_TOP, name="Top")
    gmsh.model.addPhysicalGroup(
        1, [c[1] for c in left_curves], TAG_LEFT, name="Left")
    gmsh.model.addPhysicalGroup(
        1, [c[1] for c in crack_curves], TAG_CRACK, name="Crack")

    # 6. Uniform mesh size
    gmsh.option.setNumber("Mesh.MeshSizeMin", lc)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
    gmsh.model.mesh.generate(2)

    mesh_data = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    domain, cell_tags, facet_tags = mesh_data.mesh, mesh_data.cell_tags, mesh_data.facet_tags
    gmsh.finalize()
    return domain, facet_tags

def setup_problem(domain, facet_tags):
    V_u = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    V_phi = fem.functionspace(domain, ("Lagrange", 1))

    u = fem.Function(V_u, name="Displacement")
    phi = fem.Function(V_phi, name="PhaseField")
    phi_old = fem.Function(V_phi)
    H = fem.Function(V_phi, name="History")

    E = fem.Constant(domain, default_scalar_type(E_val))
    nu = fem.Constant(domain, default_scalar_type(nu_val))
    gc = fem.Constant(domain, default_scalar_type(gc_val))
    l0 = fem.Constant(domain, default_scalar_type(l0_val))
    k_res = fem.Constant(domain, default_scalar_type(1e-6))

    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    def eps(u_): return ufl.sym(ufl.grad(u_))

    def psi_plus(u_):
        e = eps(u_)
        tr_e = ufl.tr(e)
        det_e = ufl.det(e)
        gap = ufl.sqrt(ufl.max_value((tr_e / 2.0)**2 - det_e, 0.0))
        eps1 = tr_e / 2.0 + gap
        eps2 = tr_e / 2.0 - gap
        def mac_plus(x): return ufl.max_value(x, 0)
        return 0.5 * lmbda * mac_plus(tr_e)**2 + mu * (mac_plus(eps1)**2 + mac_plus(eps2)**2)

    def sigma(u_): return lmbda * ufl.tr(eps(u_)) * ufl.Identity(domain.geometry.dim) + 2.0 * mu * eps(u_)
    def g(phi_): return (1.0 - phi_)**2 + k_res

    u_trial, v = ufl.TrialFunction(V_u), ufl.TestFunction(V_u)
    phi_trial, q = ufl.TrialFunction(V_phi), ufl.TestFunction(V_phi)

    a_u = ufl.inner(g(phi) * sigma(u_trial), eps(v)) * ufl.dx
    L_u = ufl.inner(fem.Constant(domain, default_scalar_type((0.0, 0.0))), v) * ufl.dx

    a_phi = (gc * l0 * ufl.inner(ufl.grad(phi_trial), ufl.grad(q)) + (gc / l0 + 2.0 * H) * phi_trial * q) * ufl.dx
    L_phi = (2.0 * H * q) * ufl.dx

    t_disp = fem.Constant(domain, default_scalar_type(0.0))

    # BC: bottom fixed, top uy = t_disp (same as PMB2_1/paper)
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

    bottom_facets = facet_tags.find(TAG_BOTTOM)
    bottom_dofs = fem.locate_dofs_topological(V_u, domain.topology.dim - 1, bottom_facets)
    bc_bottom = fem.dirichletbc(np.array([0.0, 0.0], dtype=default_scalar_type), bottom_dofs, V_u)

    top_facets = facet_tags.find(TAG_TOP)
    top_dofs_y = fem.locate_dofs_topological(V_u.sub(1), domain.topology.dim - 1, top_facets)
    bc_top = fem.dirichletbc(t_disp, top_dofs_y, V_u.sub(1))


    # top_dofs_x = fem.locate_dofs_topological(V_u.sub(0), domain.topology.dim - 1, top_facets)
    # bc_top_x = fem.dirichletbc(default_scalar_type(0.0), top_dofs_x, V_u.sub(0))

    bcs_u = [bc_bottom, bc_top]
    # bcs_u = [bc_bottom, bc_top, bc_top_x]

    problem_u = LinearProblem(a_u, L_u, bcs=bcs_u,
        petsc_options={"ksp_type": "cg", "pc_type": "gamg", "ksp_rtol": 1e-8}, petsc_options_prefix='p_u')

    problem_phi = LinearProblem(a_phi, L_phi, bcs=[],
        petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-8}, petsc_options_prefix='p_phi')

    return {
        "mesh": domain, "facet_tags": facet_tags,
        "V_u": V_u, "V_phi": V_phi,
        "u": u, "phi": phi, "phi_old": phi_old, "H": H,
        "E": E, "nu": nu, "gc": gc, "l0": l0, "k_res": k_res,
        "lmbda": lmbda, "mu": mu,
        "eps": eps, "psi_plus": psi_plus, "g": g, "sigma": sigma,
        "t_disp": t_disp,
        "problem_u": problem_u, "problem_phi": problem_phi,
        "ds": ufl.Measure("ds", domain=domain, subdomain_data=facet_tags),
    }

# ==========================================
# 4. Main simulation (no AMR, no ATS)
# ==========================================
domain, facet_tags = generate_initial_mesh()
sys = setup_problem(domain, facet_tags)

# Initial crack: phi=1 on y=0.5, x<=0.5 (geometric, no TAG_CRACK needed)
def crack_marker(x):
    return np.logical_and(x[0] <= 0.5 + 1e-10, np.isclose(x[1], 0.5, atol=1e-10))
crack_dofs = fem.locate_dofs_geometrical(sys["V_phi"], crack_marker)
sys["phi"].x.array[crack_dofs] = 1.0
sys["phi_old"].x.array[:] = sys["phi"].x.array[:]
sys["H"].x.array[crack_dofs] = 1e10

# Reaction force form
load_form = fem.form(sys["g"](sys["phi"]) * (
    sys["lmbda"] * ufl.tr(sys["eps"](sys["u"])) * ufl.Identity(2)
    + 2 * sys["mu"] * sys["eps"](sys["u"]))[1, 1] * sys["ds"](TAG_TOP))

# OUTPUT: load-displacement file, peak load, final damage and history fields (XDMF/HDF5),
# OUTPUT: and two global integrals for the rank-dependence check.  Nothing here changes the solve.
# OUTPUT: Called every 50 steps (files overwritten, so a run cut by the CI time limit keeps its
# OUTPUT: latest state) and once after the loop.
def _write_outputs(final):  # OUTPUT:
    _comm = sys["mesh"].comm  # OUTPUT:
    if _comm.rank == 0:  # OUTPUT:
        np.savetxt(f"loaddisp_{OUT_TAG}.txt", np.column_stack([results_disp, np.array(results_load) * 1000.0]),
                   header="u_top [mm]   F [N] (reaction on top, F = global_load*1000 as in the step print)")  # OUTPUT:
        _k = int(np.argmax(results_load))  # OUTPUT:
        print(f"{'PEAK LOAD' if final else '  partial: peak so far'} {results_load[_k]*1000:.4f} N at u={results_disp[_k]:.6f} mm (step {_k+1}/{len(results_load)}); "
              f"final load {results_load[-1]*1000:.4f} N = {100*results_load[-1]/results_load[_k]:.2f}% of peak")  # OUTPUT:
    _iphi = _comm.allreduce(fem.assemble_scalar(fem.form(sys["phi"] * ufl.dx)), op=MPI.SUM)  # OUTPUT:
    _H = sys["H"]  # OUTPUT:
    _Hcap = fem.Function(_H.function_space)  # OUTPUT: H capped at 1 (excludes the 1e10 initial-crack values)
    _Hcap.x.array[:] = np.minimum(_H.x.array, 1.0)  # OUTPUT:
    _iH = _comm.allreduce(fem.assemble_scalar(fem.form(_Hcap * ufl.dx)), op=MPI.SUM)  # OUTPUT:
    if _comm.rank == 0:  # OUTPUT:
        print(f"{'INTEGRALS' if final else '  partial integrals'}: int phi dx = {_iphi:.12e} ; int min(H,1) dx = {_iH:.12e} ; ranks = {_comm.size}")  # OUTPUT:
    with io.XDMFFile(_comm, f"damage_{OUT_TAG}.xdmf", "w") as _xf:  # OUTPUT:
        _xf.write_mesh(sys["mesh"])  # OUTPUT:
        _xf.write_function(sys["phi"], u_total)  # OUTPUT:
    with io.XDMFFile(_comm, f"history_{OUT_TAG}.xdmf", "w") as _xf:  # OUTPUT:
        _xf.write_mesh(sys["mesh"])  # OUTPUT:
        _xf.write_function(_Hcap, u_total)  # OUTPUT: min(H,1); H itself holds 1e10 on the initial crack

results_disp = []
results_load = []

u_total = 0.0
num_cells_global = domain.topology.index_map(domain.topology.dim).size_global

print(f"Start PFM simulation (uniform mesh, no AMR, no ATS)")
print(f"Mesh: {num_cells_global} cells, lc={lc_initial}")
print(f"Loading: du1={delta_u_stage1:.0e} up to u={u_stage_switch}, du2={delta_u_stage2:.0e} to u={u_max}")
print()

while u_total < u_max:
    # Two-stage fixed step size (paper PFM_C)
    if u_total < u_stage_switch:
        delta_u = delta_u_stage1
    else:
        delta_u = delta_u_stage2

    delta_u = min(delta_u, u_max - u_total)
    u_trial = u_total + delta_u
    sys["t_disp"].value = default_scalar_type(u_trial)

    # Staggered iteration
    converged = False
    _ksp_bad = []  # OUTPUT: diagnostic only
    for iter_count in range(iter_max):
        # Solve displacement
        u_res = sys["problem_u"].solve()
        _r = sys["problem_u"].solver.getConvergedReason()  # OUTPUT: read the KSP converged reason (no effect on the solve)
        if _r < 0:  # OUTPUT:
            _ksp_bad.append((_r, sys["problem_u"].solver.getIterationNumber()))  # OUTPUT:
        sys["u"].x.array[:] = u_res.x.array[:]

        # Update history field
        psi_expr = fem.Expression(sys["psi_plus"](sys["u"]), sys["V_phi"].element.interpolation_points)
        psi_vals = fem.Function(sys["V_phi"])
        psi_vals.interpolate(psi_expr)
        sys["H"].x.array[:] = np.maximum(psi_vals.x.array, sys["H"].x.array)

        # Solve phase field
        phi_res = sys["problem_phi"].solve()
        sys["phi"].x.array[:] = phi_res.x.array[:]
        sys["phi"].x.array[:] = np.clip(sys["phi"].x.array, 0.0, 1.0)

        # Convergence check
        error_local = fem.assemble_scalar(fem.form((sys["phi"] - sys["phi_old"])**2 * ufl.dx))
        error_global = sys["mesh"].comm.allreduce(error_local, op=MPI.SUM)
        norm_local = fem.assemble_scalar(fem.form(sys["phi"]**2 * ufl.dx))
        norm_global = sys["mesh"].comm.allreduce(norm_local, op=MPI.SUM)
        rel_error = np.sqrt(error_global) / (np.sqrt(norm_global) + 1e-12)

        if rel_error < tol_staggered:
            converged = True
            break
        sys["phi_old"].x.array[:] = sys["phi"].x.array[:]

    N_iter = iter_count + 1
    if _ksp_bad and sys["mesh"].comm.rank == 0:  # OUTPUT:
        print(f"  KSP-U DIVERGED in {len(_ksp_bad)}/{N_iter} displacement solves of this step "
              f"(reason, its): {_ksp_bad[:3]}")  # OUTPUT:

    if not converged:
        print(f"  Warning: step at u={u_trial:.6f} did not converge ({N_iter} iters, err={rel_error:.2e})")

    u_total = u_trial
    sys["phi_old"].x.array[:] = sys["phi"].x.array[:]

    # Reaction force
    local_load = fem.assemble_scalar(load_form)
    global_load = sys["mesh"].comm.allreduce(local_load, op=MPI.SUM)
    results_load.append(global_load)
    results_disp.append(u_total)

    print(f"Step {len(results_disp):4d} | u={u_total:.6f} | iter={N_iter} | "
          f"max_phi={np.max(sys["phi"].x.array):.3f} | F={global_load*1000:.2f} N")
    if len(results_disp) % 50 == 0:  # OUTPUT:
        _write_outputs(final=False)  # OUTPUT:

print()
print(f"Simulation finished. Steps={len(results_disp)}, cells={num_cells_global}")

_write_outputs(final=True)  # OUTPUT:

# PORT: the pyvista visualization below is disabled: pyvista is not installed in
# PORT: dolfinx/dolfinx:stable and plotter.show() needs a display (headless CI runner).
# PORT: The final fields are written to XDMF instead (see # OUTPUT: above).  Lines kept, commented out.
# PORT: # ==========================================
# PORT: # ==========================================
# PORT: # 5. Visualization
# PORT: # ==========================================
# PORT: # Phase field with mesh edges
# PORT: import pyvista
# PORT:
# PORT: topology_phi, cell_types_phi, geometry_phi = plot.vtk_mesh(sys["V_phi"])
# PORT: grid_phi = pyvista.UnstructuredGrid(topology_phi, cell_types_phi, geometry_phi)
# PORT: grid_phi.point_data["phi"] = sys["phi"].x.array.real
# PORT: grid_phi.set_active_scalars("phi")
# PORT:
# PORT: plotter = pyvista.Plotter(window_size=(900, 800))
# PORT: plotter.set_background("white")
# PORT: plotter.add_mesh(grid_phi, show_edges=False,
# PORT:                 cmap="coolwarm", clim=[0, 1])
# PORT: plotter.add_text(f"Phase Field (u={u_total:.4f} mm, {num_cells_global} cells)",
# PORT:                  font_size=12, color="black")
# PORT: plotter.view_xy()
# PORT: plotter.show()
# PORT:
# PORT: # Displacement magnitude
# PORT: u_mag = np.sqrt(sys["u"].x.array[0::2]**2 + sys["u"].x.array[1::2]**2)
# PORT:
# PORT: topology_u, cell_types_u, geometry_u = plot.vtk_mesh(sys["V_u"])
# PORT: grid_u = pyvista.UnstructuredGrid(topology_u, cell_types_u, geometry_u)
# PORT: grid_u.point_data["|u|"] = u_mag.real
# PORT: grid_u.set_active_scalars("|u|")
# PORT:
# PORT: plotter2 = pyvista.Plotter(window_size=(900, 800))
# PORT: plotter2.set_background("white")
# PORT: plotter2.add_mesh(grid_u, show_edges=False, 
# PORT:                 cmap="viridis")
# PORT: plotter2.add_text(f"Displacement Magnitude (u={u_total:.4f} mm)",
# PORT:                    font_size=12, color="black")
# PORT: plotter2.view_xy()
# PORT: plotter2.show()
