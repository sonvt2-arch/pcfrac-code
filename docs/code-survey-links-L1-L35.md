# Survey of public codes: links L1 to L35

This file belongs to Section S12 ("Survey of public codes") of the Supplementary Material of the manuscript. It lists the links L1 to L35 of Table S11, moved here verbatim from the Supplementary Material. Each link points to the commit and lines that were read to find where the code stores the history field $H$. Table S11 is repeated below so that each link can be matched to its code; the classes (a) to (d) are defined in Section S12.

**Table S11.** Public codes with a history field and the storage of $H$.

| # | Code | Link | Storage of $H$ | Class |
|---|------------------------|----|--------------------------|----|
| 1 | FEniCS forum post #19720 (2026), FEniCSx | L1 | P1 Lagrange; $\psi_0^+$ interpolated into P1, then a nodal maximum | (a) |
| 2 | FEniCS forum post #10937 (2023), FEniCSx | L2 | CG1; $\max(\psi, H)$ interpolated into CG1 | (a) |
| 3 | PhaseFieldX (Castillón), FEniCSx; J. Open Source Softw., doi:10.21105/joss.07307 | L3 | Lagrange degree 1; $\psi_0^+$ projected in $L^2$, then a nodal maximum | (b) |
| 4 | Sobia020, Lunar-Floor-Fractured-Craters, legacy FEniCS | L4 | P1, $L^2$ projection | (b) |
| 5 | FEniCS forum posts #17205/​#17206 (2025), legacy FEniCS | L5 | CG1, $L^2$ projection | (b) |
| 6 | PhaseFieldFEniCS (Martínez-Pañeda); Compos. Part B Eng. 2019, doi:10.1016/j.compositesb.2019.04.003 | L6 | CG1 declared, DG0 projection assigned | (c), CG1 declared |
| 7 | saugatsn/​Phase-Field-Modeling (copy of code 6) | L7 | as code 6 | (c), CG1 declared |
| 8 | maryhzd/​Phase-Field-Fracture-T-stress (copy of code 6) | L8 | as code 6 | (c), CG1 declared |
| 9 | jzghal/​PhaFiDyn | L9 | CG1 declared, DG0 projection assigned | (c), CG1 declared |
| 10 | hyoungsuksuh/​micropolar_​phasefield | L10 | CG1 declared, DG0 projection assigned | (c), CG1 declared |
| 11 | Hirshikesh/​FEniCS-PhaseField-brittle-fracture, TwoElem_​upHist.py | L11 | DG0 | (c) |
| 12 | Vinutr11/​FEniCS-Explicit_​PhaseField_​dynamic_​fracture | L12 | DG0, local projection | (c) |
| 13 | iitrabhi/​fenics-phase-field-fracture (notebook src/​main.ipynb, cell 5) | L13 | DG0 | (c) |
| 14 | erfanhamdi/​pfm_​dataset; Comput. Methods Appl. Mech. Eng. 2026, doi:10.1016/j.cma.2025.118526 | L14 | DG0, interpolated at element centers | (c) |
| 15 | AlistairB99124/​Gairezi, FEniCSx | L15 | DG0 | (c) |
| 16 | NanxiiChen/​irr-pinns, fracture-fem-cuda.py | L16 | DG0 | (c) |
| 17 | ssmg-unitn-it/​Slide_​Gate_​Plate_​simulation | L17 | DG0 | (c) |
| 18 | giozu/​z3st; SoftwareX 2026, doi:10.1016/j.softx.2026.103138 | L18 | DG0 (default) or quadrature element | (c) |
| 19 | T-Hageman/​Cohesive-PhaseField-FenicsX; Eng. Fract. Mech. 2026, doi:10.1016/j.engfracmech.2026.112292 | L19 | quadrature element | (c) |
| 20 | PhaRoughLife (Jiménez Alfaro and Martínez-Pañeda), fatigue | L20 | $H$ on a quadrature element; fatigue history variable on CG1, filled by interpolation | (c) for $H$; (a) for the fatigue variable |
| 21 | uekerman/​Coupled-Brittle-Fracture, nutils | L21 | Gauss points | (c) |
| 22 | CEMS-Lab/​PhAST, PyTorch | L22 | per element; a nodal average for output only | (c) |
| 23 | FEniCS forum posts #16800/​#17454 (2025), FEniCSx | L23 | DG0 | (c) |
| 24 | MOOSE, solid_​mechanics, ComputeLinearElasticPFFractureStress | L24 | material property at quadrature points | (c) |
| 25 | Molnár and Gravouil, Abaqus UEL (copy in a thesis repository); Finite Elem. Anal. Des. 2017, doi:10.1016/j.finel.2017.03.002 | L25 | state variables at integration points | (c) |
| 26 | Navidtehrani, Betegón and Martínez-Pañeda, Abaqus UMAT; Materials 2021, doi:10.3390/ma14081913 | L26 | state variables at integration points | (c) |
| 27 | Martínez-Pañeda, Golahmar and Niordson, Abaqus UEL; Comput. Methods Appl. Mech. Eng. 2018, doi:10.1016/j.cma.2018.07.021 | L27 | state variables at integration points | (c) |
| 28 | Martínez-Pañeda, Julia/​Ferrite | L28 | state at quadrature points | (c) |
| 29 | MdMasiurRahaman/​PhaseFieldModelForMechanicalFracture, Julia/​Gridap | L29 | cell state at quadrature points | (c) |
| 30 | PM-Hu/​MatPFF, MATLAB | L30 | Gauss points | (c) |
| 31 | somdattagoswami/​IGAPack-PhaseField, isogeometric, MATLAB | L31 | Gauss points | (c) |
| 32 | taojinllnl/​Phasefield_​L-BFGS_​monolithic_​solver, deal.II; Int. J. Numer. Methods Eng. 2024, doi:10.1002/nme.7572 | L32 | point history at quadrature points | (c) |
| 33 | LuoXueling/​deal.II-phase-field-fracture | L33 | quadrature-point data | (c) |
| 34 | ritukeshbharali/​PhaseFieldFractureCOMSOL, Miehe_​staggered_​tension.mph | L34 | COMSOL domain ordinary differential equation, discontinuous Lagrange shape functions of order 1 | (d) |
| 35 | jianyingwu/​pfczm-comsol, 1_​2Purely_​mechanical_​concrete_​beam.mph | L35 | COMSOL state variables; location (Gauss points or nodes) not determined | (d) |

*Note.* All links accessed on 28 October 2026. "CG1 declared": see the class (c) definition above.

## Links

All links accessed on 28 October 2026.

- L1. <https://fenicsproject.discourse.group/t/challenges-in-reproducing-phase-field-fracture-benchmark-unphysical-crack-trajectory-and-mesh-refinement-behavior/19720/5>
- L2. <https://fenicsproject.discourse.group/t/non-linear-solver-in-fenicsx-comparison-to-snes-in-legacy-fenics/10937/5>
- L3. <https://github.com/CastillonMiguel/phasefieldx/blob/1e25b6150e3f0d9bf6a2339b565b29c463099bb8/src/phasefieldx/Element/Phase_Field_Fracture/solver/solver_history.py#L162-L364>
- L4. <https://github.com/Sobia020/Modelling-and-Simulation-of-Lunar-Floor-Fractured-Craters/blob/2abdfca2d58190eca948d45ffa5b9f5281d28cca/Python%20Source%20Files/c6_lunar_crater_fracture_2d_slice.py#L44-L150>
- L5. <https://fenicsproject.discourse.group/t/crack-not-propagating-in-phase-field-simulation-hydrogen-embrittlement-fenics/17205>
- L6. https://www.empaneda.com/wp-content/uploads/2019/04/PhaseFieldFEniCS.zip (file PhaseField.py, lines 18 to 95); copy: https://github.com/SpringNuance/Abaqus-COMSOL-Subroutine-References/blob/bdd909ab3ab3753773cce23d7b8740693b64bceb/PHASE%20FIELD%20FRACTURE%20IMPLEMENTATION%20IN%20FENICS/PhaseField.py#L59-L95
- L7. <https://github.com/saugatsn/Phase-Field-Modeling/blob/67177edb6594f44e53208f2798202ca9c7c6225f/Phase_Field_Simulation.py#L93-L137>
- L8. <https://github.com/maryhzd/Phase-Field-Fracture-T-stress/blob/2edcca6258315515912954c0c98afdd580a2fe06/VolDev_split.py#L94-L137>
- L9. <https://github.com/jzghal/PhaFiDyn/blob/e9f7c5aa6b396de73f0694e79f118ec09c704a04/PhaFiDyn_KalthoffWinkler_Simplified_AT2.py#L1103-L1188>
- L10. <https://github.com/hyoungsuksuh/micropolar_phasefield/blob/df59a2b9ebc29f225d306dce2f8de871170c8c01/fenics_code/micropolar_phasefield.py#L83-L476>
- L11. <https://github.com/Hirshikesh/FEniCS-PhaseField-brittle-fracture/blob/23dff9fae5b92f9d67cba0cef44b9acd5582b6b5/Twoelement/TwoElem_upHist.py#L79-L160>
- L12. <https://github.com/Vinutr11/FEniCS-Explicit_PhaseField_dynamic_fracture/blob/fb27a943dbaee56eea103874419d16197cedffc7/FEniCS_codes/2D_Dynamic_crack_branch.py#L171-L189>
- L13. <https://github.com/iitrabhi/fenics-phase-field-fracture/blob/65f282512882e4a7eb6ed936c9393046d4e458c2/src/main.ipynb>
- L14. <https://github.com/erfanhamdi/pfm_dataset/blob/150e95488f27bc909266b2cdd2b398c3b768ff64/src/main.py#L114-L391>
- L15. <https://github.com/AlistairB99124/Gairezi/blob/3db8b0dc4a5e72f2df1132f48f09a9c54c18aa1f/FEniCSx/phase_field_solver.py#L414-L437>
- L16. <https://github.com/NanxiiChen/irr-pinns/blob/e1794f324a603689462d0db9da2debc2778d63a7/examples/fracture/fracture-fem-cuda.py#L82-L220>
- L17. <https://github.com/ssmg-unitn-it/Slide_Gate_Plate_simulation/blob/68fdcb6a59b5a1735d8a4787ed0730c7829fdb08/Numerical_experiment/ThermoMecaFractureSolver.py#L301-L1146>
- L18. <https://github.com/giozu/z3st/blob/f8c3c2011a4cc7aa37026e0713d6b7e0facf9aeb/z3st/core/spine.py#L484-L498>
- L19. <https://github.com/T-Hageman/Cohesive-PhaseField-FenicsX/blob/42e7cce8ba929e3600919bd10d004616d8cc4c58/Models/SolidModels/PhaseField.py#L38-L93>
- L20. https://www.empaneda.com/wp-content/uploads/2025/05/PhaRoughLife.zip (file pfsolver_fatigue.py); copy: https://github.com/SpringNuance/Abaqus-COMSOL-Subroutine-References/blob/bdd909ab3ab3753773cce23d7b8740693b64bceb/PHAROUGHLIFE%20A%20FENICSX%20%26%20GMSH%20CODE%20TO%20PREDICT%20THE%20EFFECT%20OF%20SURFACE%20ROUGHNESS%20IN%20FATIGUE/phasefield_pycodes/pfsolver_fatigue.py#L69-L371
- L21. <https://github.com/uekerman/Coupled-Brittle-Fracture/blob/606f2034193409c52c0aea18809f21fd58c3bf0a/fracture.py#L40-L138>
- L22. <https://github.com/CEMS-Lab/PhAST/blob/3a7320fae70b31aa32037be672f37b55bd8e05f9/src/phast/solvers/staggered_solver.py#L879-L890>
- L23. <https://fenicsproject.discourse.group/t/residual-mismatch-with-custom-newton-solver/16800>
- L24. <https://github.com/idaholab/moose/blob/974e272c685f95854359644295053a1fff0e9ccd/modules/solid_mechanics/src/materials/ComputeLinearElasticPFFractureStress.C#L227-L243>
- L25. <https://github.com/PRVSTUDENT/adaptive-remeshing-phase-field-thesis/blob/b459717f498e9bad3dc8afabbed89fee150dddb4/models/baseline_original/molnar_gravouil_2017/02_Single_Notch_Tension/SingleNotch.for#L204-L210>
- L26. https://www.empaneda.com/wp-content/uploads/2021/07/PhaseFieldUMAT.zip (file UMATs.f, lines 34 to 65); copy: https://github.com/SpringNuance/Abaqus-COMSOL-Subroutine-References/blob/bdd909ab3ab3753773cce23d7b8740693b64bceb/ABAQUS%20UMAT%20SUBROUTINE%20TO%20IMPLEMENT%20PHASE%20FIELD%20FRACTURE/UMATs.f#L34-L65
- L27. https://www.empaneda.com/wp-content/uploads/2018/12/PhaseFieldUEL.zip (file PhaseField.for, lines 92 to 110)
- L28. https://www.empaneda.com/wp-content/uploads/2023/04/JuliaPhaseField.zip (file Subroutines.jl, lines 119 to 133)
- L29. <https://github.com/MdMasiurRahaman/PhaseFieldModelForMechanicalFracture/blob/2db10770a19aff959223f75ae4b7cd5d29ca7915/AsymmetricNotchedThree-pointBendingTest/AsymmetricThreePointBendingCase3.ipynb>
- L30. <https://github.com/PM-Hu/MatPFF/blob/d88902d62564fc4093b705c5bf0d8904a3046187/Func/updateRefEnerg.m#L46-L47>
- L31. <https://github.com/somdattagoswami/IGAPack-PhaseField/blob/ee77c8e47e8832c4229b3f11b149fae1bb5873bc/3D%20Examples/utils/gStiffnessPhiPhi3Dcubic.m#L48-L49>
- L32. <https://github.com/taojinllnl/Phasefield_L-BFGS_monolithic_solver/blob/5127ed3e2d3818189080b80184b324497645da07/SourceCode/main.cc#L935-L941>
- L33. <https://github.com/LuoXueling/deal.II-phase-field-fracture/blob/e368f1c6b5fe0431ecc4cf0e0237d8a7362521ac/include/controller.h#L12-L35>
- L34. <https://github.com/ritukeshbharali/PhaseFieldFractureCOMSOL/blob/2a3b0a8101aa4db19f74b2fe806e4c97bf86483f/Miehe_staggered_tension.mph>
- L35. <https://github.com/jianyingwu/pfczm-comsol/blob/e6c940e08359e3fc0a575d7a3582a5ccaad469d3/1_2Purely_mechanical_concrete_beam.mph>
