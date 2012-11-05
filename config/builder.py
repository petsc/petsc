#!/usr/bin/env python

from __future__ import with_statement  # For python-2.5

import os, sys
import shutil
import tempfile

sys.path.insert(0, os.path.join(os.environ['PETSC_DIR'], 'config'))
sys.path.insert(0, os.path.join(os.environ['PETSC_DIR'], 'config', 'BuildSystem'))

import logger, script

regressionRequirements = {'src/vec/vec/examples/tests/ex31':  set(['Matlab'])
                          }

regressionParameters = {'src/sys/comm/examples/tests/ex1':    [{'numProcs': 2},
                                                               {'numProcs': 5}],
                        'src/vec/vec/examples/tests/ex1_2':    {'numProcs': 2},
                        'src/vec/vec/examples/tests/ex3':      {'numProcs': 2},
                        'src/vec/vec/examples/tests/ex4':      {'numProcs': 2},
                        'src/vec/vec/examples/tests/ex5':      {'numProcs': 2},
                        'src/vec/vec/examples/tests/ex9':      {'numProcs': 2},
                        'src/vec/vec/examples/tests/ex10':     {'numProcs': 2},
                        'src/vec/vec/examples/tests/ex11':     {'numProcs': 2},
                        'src/vec/vec/examples/tests/ex12':     {'numProcs': 2},
                        'src/vec/vec/examples/tests/ex13':     {'numProcs': 2},
                        'src/vec/vec/examples/tests/ex14':     {'numProcs': 2},
                        'src/vec/vec/examples/tests/ex16':     {'numProcs': 2},
                        'src/vec/vec/examples/tests/ex17':     {'numProcs': 2},
                        'src/vec/vec/examples/tests/ex17f':    {'numProcs': 3},
                        'src/vec/vec/examples/tests/ex21_2':   {'numProcs': 2},
                        'src/vec/vec/examples/tests/ex22':     {'numProcs': 4},
                        'src/vec/vec/examples/tests/ex23':     {'numProcs': 2},
                        'src/vec/vec/examples/tests/ex24':     {'numProcs': 3},
                        'src/vec/vec/examples/tests/ex25':     {'numProcs': 3},
                        'src/vec/vec/examples/tests/ex26':     {'numProcs': 4},
                        'src/vec/vec/examples/tests/ex28':     {'numProcs': 3},
                        'src/vec/vec/examples/tests/ex29':     {'numProcs': 3, 'args': '-n 126'},
                        'src/vec/vec/examples/tests/ex30f':    {'numProcs': 4},
                        'src/vec/vec/examples/tests/ex33':     {'numProcs': 4},
                        'src/vec/vec/examples/tests/ex36':     {'numProcs': 2, 'args': '-set_option_negidx -set_values_negidx -get_values_negidx'},
                        'src/vec/vec/examples/tutorials/ex10':  [{'numProcs': 1, 'args': '-hdf5'},
                                                                 {'numProcs': 2, 'args': '-binary'},
                                                                 {'numProcs': 3, 'args': '-binary'},
                                                                 {'numProcs': 5, 'args': '-binary'}],
                        'src/dm/impls/complex/examples/tests/ex1': [{'numProcs': 1, 'args': '-dim 3 -ctetgen_verbose 4 -dm_view_detail -info -info_exclude null'},
                                                                    {'numProcs': 1, 'args': '-dim 3 -ctetgen_verbose 4 -refinement_limit 0.0625 -dm_view_detail -info -info_exclude null'}],
                        'src/dm/impls/complex/examples/tests/ex1f90': [{'numProcs': 1, 'args': ''}],
                        'src/dm/impls/complex/examples/tests/ex2f90': [{'numProcs': 1, 'args': ''}],
                        'src/dm/impls/mesh/examples/tests/ex1': [{'numProcs': 1, 'args': '-dim 2 -dm_view'},
                                                                 {'numProcs': 1, 'args': '-dim 2 -dm_view -interpolate'},
                                                                 {'numProcs': 1, 'args': '-dim 3 -dm_view'},
                                                                 {'numProcs': 1, 'args': '-dim 3 -dm_view -interpolate'},
                                                                 {'numProcs': 3, 'args': '-dim 2 -dm_view'},
                                                                 {'numProcs': 3, 'args': '-dim 2 -dm_view -interpolate'},
                                                                 {'numProcs': 8, 'args': '-dim 2 -dm_view'},
                                                                 {'numProcs': 8, 'args': '-dim 2 -dm_view -interpolate'},
                                                                 {'numProcs': 6, 'args': '-dim 3 -dm_view'},
                                                                 {'numProcs': 6, 'args': '-dim 3 -dm_view -interpolate'},
                                                                 {'numProcs': 10, 'args': '-dim 2 -dm_view -refinement_limit 7.63e-6'},
                                                                 {'numProcs': 10, 'args': '-dim 3 -dm_view -refinement_limit 1.00e-7'},
                                                                 {'numProcs': 50, 'args': '-dim 2 -dm_view -refinement_limit 7.63e-6'},
                                                                 {'numProcs': 50, 'args': '-dim 3 -dm_view -refinement_limit 1.00e-7'}],
                        'src/dm/impls/mesh/examples/tutorials/ex1': [{'numProcs': 1},
                                                                     {'numProcs': 2, 'args': '-base_file src/dm/impls/mesh/examples/tutorials/data/ex1_3d_big -dim 3'}],
                        'src/dm/impls/mesh/examples/tests/ex3': [{'numProcs': 1, 'args': '-malloc_dump -dm_mesh_view -dm_mesh_new_impl'},
                                                                 {'numProcs': 2, 'args': '-malloc_dump -dm_mesh_view -dm_mesh_new_impl'},
                                                                 {'numProcs': 3, 'args': '-malloc_dump -dm_mesh_view -dm_mesh_new_impl'},
                                                                 {'numProcs': 5, 'args': '-malloc_dump -dm_mesh_view -dm_mesh_new_impl'}],
                        'src/dm/impls/mesh/examples/tests/ex4': [{'numProcs': 1, 'args': '-i src/dm/impls/mesh/examples/tests/meshes/Square-tri.gen'},
                                                                 {'numProcs': 2, 'args': '-i src/dm/impls/mesh/examples/tests/meshes/Square-tri.gen'}],
                        'src/dm/impls/mesh/examples/tests/ex5f90': [{'numProcs': 1, 'args': '-i src/dm/impls/mesh/examples/tests/meshes/Square-tri.gen'},
                                                                    {'numProcs': 2, 'args': '-i src/dm/impls/mesh/examples/tests/meshes/Square-tri.gen'}],
                        'src/dm/impls/mesh/examples/tests/ex6': [{'numProcs': 1, 'args': '-i src/dm/impls/mesh/examples/tests/meshes/Square-tri.gen'},
                                                                 {'numProcs': 2, 'args': '-i src/dm/impls/mesh/examples/tests/meshes/Square-tri.gen'}],
                        'src/dm/impls/mesh/examples/tests/ex7f90': [{'numProcs': 1, 'args': '-i src/dm/impls/mesh/examples/tests/meshes/Square-tri.gen'},
                                                                    {'numProcs': 2, 'args': '-i src/dm/impls/mesh/examples/tests/meshes/Square-tri.gen'}],
                        'src/dm/impls/mesh/examples/tests/ex8': [{'numProcs': 1, 'args': '-i src/dm/impls/mesh/examples/tests/meshes/Square-tri.gen'},
                                                                 {'numProcs': 2, 'args': '-i src/dm/impls/mesh/examples/tests/meshes/Square-tri.gen'}],
                        'src/dm/impls/mesh/examples/tests/ex9f90': [{'numProcs': 1, 'args': '-i src/dm/impls/mesh/examples/tests/meshes/Square-tri.gen'},
                                                                    {'numProcs': 2, 'args': '-i src/dm/impls/mesh/examples/tests/meshes/Square-tri.gen'}],
                        'src/dm/impls/mesh/examples/tests/ex10': [{'numProcs': 1, 'args': '-i src/dm/impls/mesh/examples/tests/meshes/Square-tri.gen'},
                                                                  {'numProcs': 2, 'args': '-i src/dm/impls/mesh/examples/tests/meshes/Square-tri.gen'}],
                        'src/dm/impls/mesh/examples/tests/ex11f90': [{'numProcs': 1, 'args': '-i src/dm/impls/mesh/examples/tests/meshes/Square-tri.gen'},
                                                                     {'numProcs': 2, 'args': '-i src/dm/impls/mesh/examples/tests/meshes/Square-tri.gen'}],
                        'src/dm/impls/mesh/examples/tests/TestMeshExodus': [{'numProcs': 1, 'args': '-i src/dm/impls/mesh/examples/tests/meshes/Square-tri.gen'},
                                                                            {'numProcs': 1, 'args': '-i src/dm/impls/mesh/examples/tests/meshes/Square-tri2.gen'},
                                                                            {'numProcs': 1, 'args': '-i src/dm/impls/mesh/examples/tests/meshes/Square-quad.gen'},
                                                                           #{'numProcs': 1, 'args': '-i src/dm/impls/mesh/examples/tests/meshes/Square-mixed.gen'},
                                                                            {'numProcs': 1, 'args': '-i src/dm/impls/mesh/examples/tests/meshes/Tet.gen'},
                                                                            {'numProcs': 1, 'args': '-i src/dm/impls/mesh/examples/tests/meshes/Hex.gen'},
                                                                            {'numProcs': 1, 'args': '-i src/dm/impls/mesh/examples/tests/meshes/Tet.gen -interpolate'},
                                                                            {'numProcs': 1, 'args': '-i src/dm/impls/mesh/examples/tests/meshes/Hex.gen -interpolate'}],
                        'src/ksp/ksp/examples/tests/ex10':    [{'numProcs': 1, 'args': '-m 3 -matconvert_type seqaij -ksp_monitor_short'}],
                        'src/ksp/ksp/examples/tutorials/ex4': [{'numProcs': 1, 'args': '-info -info_exclude null,vec'},
                                                               {'numProcs': 1, 'args': '-da_grid_x 10 -da_grid_y 10 -solve -ksp_monitor'},
                                                               {'numProcs': 2, 'args': '-da_grid_x 10 -da_grid_y 10'}],
                                                               #{'numProcs': 1, 'args': '-info'}],
                        'src/ksp/ksp/examples/tutorials/ex12': {'numProcs': 2, 'args': '-ksp_gmres_cgs_refinement_type refine_always'},
                        'src/ksp/ksp/examples/tutorials/ex40': {'numProcs': 1, 'args': '-mat_no_inode -ksp_monitor_short'},
                        'src/ksp/ksp/examples/tutorials/ex54':[{'numProcs': 4, 'args': '-ne 40 -alpha 1.e-3 -ksp_monitor_short -ksp_type cg -ksp_norm_type unpreconditioned'},
                                                               {'numProcs': 4, 'args': '-ne 40 -alpha 1.e-3 -ksp_monitor_short -ksp_type cg -ksp_norm_type unpreconditioned -pc_gamg_type sa'}],
                        'src/ksp/ksp/examples/tutorials/ex55':[{'numProcs': 4, 'args': '-ne 40 -alpha 1.e-3 -ksp_monitor_short -ksp_type cg -ksp_norm_type unpreconditioned'},
                                                               {'numProcs': 4, 'args': '-ne 40 -alpha 1.e-3 -ksp_monitor_short -ksp_type cg -ksp_norm_type unpreconditioned -pc_gamg_type sa'}],
                        'src/ksp/ksp/examples/tutorials/ex56':[{'numProcs': 8, 'args': '-ne 11 -alpha 1.e-3 -ksp_monitor_short -ksp_type cg -ksp_norm_type unpreconditioned -pc_gamg_type sa'}],
                        'src/snes/examples/tutorials/ex5':    [{'numProcs': 4, 'args': '-snes_mf -da_processors_x 4 -da_processors_y 1 -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always'},
                                                               {'numProcs': 1, 'args': '-pc_type mg -ksp_monitor_short  -snes_view -pc_mg_levels 3 -pc_mg_galerkin -da_grid_x 17 -da_grid_y 17 -mg_levels_ksp_monitor_short -snes_monitor_short -mg_levels_pc_type sor -pc_mg_type full'},
                                                               {'numProcs': 1, 'args': '-pc_type mg -ksp_monitor_short  -snes_view -pc_mg_galerkin -snes_grid_sequence 3 -mg_levels_ksp_monitor_short -snes_monitor_short -mg_levels_pc_type sor -pc_mg_type full'},
                                                               {'numProcs': 2, 'args': '-snes_grid_sequence 2 -snes_mf_operator -snes_converged_reason -snes_view -pc_type mg'},
                                                               {'numProcs': 2, 'args': '-snes_grid_sequence 2 -snes_monitor_short -ksp_monitor_short -ksp_converged_reason -snes_converged_reason -snes_view -pc_type mg'}],
                        'src/snes/examples/tutorials/ex5f90':  {'numProcs': 4, 'args': '-snes_mf -da_processors_x 4 -da_processors_y 1 -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always'},
                        'src/snes/examples/tutorials/ex5f90t': {'numProcs': 4, 'args': '-snes_mf -da_processors_x 4 -da_processors_y 1 -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always'},
                        'src/snes/examples/tutorials/ex9':    [{'numProcs': 1, 'args': '-snes_mf -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always'}],
#                        'src/snes/examples/tutorials/ex19':    {'numProcs': 2, 'args': '-dmmg_nlevels 4 -snes_monitor_short'},
                        'src/snes/examples/tutorials/ex19':   [{'numProcs': 1, 'args': '-dm_vec_type seq     -dm_mat_type seqaij     -pc_type none -dmmg_nlevels 1 -da_grid_x 200 -da_grid_y 200 -mat_no_inode -preload off -log_summary -log_summary_py'},
                                                               {'numProcs': 1, 'args': '-dm_vec_type seqcusp -dm_mat_type seqaijcusp -pc_type none -dmmg_nlevels 1 -da_grid_x 200 -da_grid_y 200 -mat_no_inode -preload off -log_summary -log_summary_py'}],
                        'src/snes/examples/tutorials/ex11':   [{'numProcs': 1, 'args': '-snes_mf -snes_monitor_short -snes_converged_reason'}],
                        'src/snes/examples/tutorials/ex12':   [{'numProcs': 1, 'args': '-lambda 0.0 -snes_monitor -ksp_monitor -snes_converged_reason'},
                                                               {'numProcs': 1, 'args': '-lambda 0.0 -snes_monitor -ksp_monitor -snes_converged_reason -refinement_limit 0.0625'},
                                                               {'numProcs': 2, 'args': '-lambda 0.0 -snes_monitor -ksp_monitor -snes_converged_reason'},
                                                               {'numProcs': 2, 'args': '-lambda 0.0 -snes_monitor -ksp_monitor -snes_converged_reason -refinement_limit 0.0625'},
                                                               {'numProcs': 1, 'args': '-lambda 0.0 -snes_monitor -ksp_monitor -snes_converged_reason -bc_type neumann'},
                                                               {'numProcs': 2, 'args': '-lambda 0.0 -snes_monitor -ksp_monitor -snes_converged_reason -bc_type neumann'}],
                                                               #{'numProcs': 1, 'args': '-dim 3 -lambda 0.0 -ksp_rtol 1.0e-9 -snes_monitor -ksp_monitor -snes_converged_reason -refinement_limit 0.0625'},
                                                               #{'numProcs': 2, 'args': '-dim 3 -lambda 0.0 -ksp_rtol 1.0e-9 -snes_monitor -ksp_monitor -snes_converged_reason -refinement_limit 0.005'}],
                        'src/snes/examples/tutorials/ex10':   [{'numProcs': 2, 'args': '-da_grid_x 5 -snes_converged_reason -snes_monitor_short -problem_type 0'},
                                                               {'numProcs': 1, 'args': '-da_grid_x 20 -snes_converged_reason -snes_monitor_short -problem_type 1'},
                                                               {'numProcs': 1, 'args': '-da_grid_x 20 -snes_converged_reason -snes_monitor_short -problem_type 2'},
                                                               {'numProcs': 1, 'args': '-da_grid_x 20 -snes_converged_reason -snes_monitor_short -ksp_monitor_short -problem_type 2 \
-snes_mf_operator -pack_dm_mat_type aij -pc_type fieldsplit -pc_fieldsplit_type additive -fieldsplit_u_ksp_type gmres -fieldsplit_k_pc_type jacobi'},
                                                               {'numProcs': 1, 'args': '-da_grid_x 20 -snes_converged_reason -snes_monitor_short -ksp_monitor_short -problem_type 2 \
-snes_mf_operator -pack_dm_mat_type nest -pc_type fieldsplit -pc_fieldsplit_type additive -fieldsplit_u_ksp_type gmres -fieldsplit_k_pc_type jacobi'}],
                        'src/snes/examples/tutorials/ex28':   [{'numProcs': 3, 'args': '-da_grid_x 20 -snes_converged_reason -snes_monitor_short -problem_type 0'},
                                                               {'numProcs': 1, 'args': '-da_grid_x 20 -snes_converged_reason -snes_monitor_short -problem_type 1'},
                                                               {'numProcs': 1, 'args': '-da_grid_x 20 -snes_converged_reason -snes_monitor_short -problem_type 2'},
                                                               {'numProcs': 1, 'args': '-da_grid_x 20 -snes_converged_reason -snes_monitor_short -ksp_monitor_short -problem_type 2 \
-snes_mf_operator -pack_dm_mat_type aij -pc_type fieldsplit -pc_fieldsplit_type additive -fieldsplit_u_ksp_type gmres -fieldsplit_k_pc_type jacobi'},
                                                               {'numProcs': 1, 'args': '-da_grid_x 20 -snes_converged_reason -snes_monitor_short -ksp_monitor_short -problem_type 2 \
-snes_mf_operator -pack_dm_mat_type nest -pc_type fieldsplit -pc_fieldsplit_type additive -fieldsplit_u_ksp_type gmres -fieldsplit_k_pc_type jacobi'}],
                        'src/snes/examples/tutorials/ex31':   [# Decoupled field Dirichlet tests 0-4
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0     -forcing_type constant -bc_type dirichlet -interpolate 1 -show_initial -dm_complex_print_fem 1',
                                                                'setup': './bin/pythonscripts/PetscGenerateFEMQuadrature.py 2 2 2 1 laplacian 2 1 1 1 gradient 2 1 1 1 identity src/snes/examples/tutorials/ex31.h'},
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.0     -forcing_type constant -bc_type dirichlet -interpolate 1 -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_0_fields 0,1 -pc_fieldsplit_1_fields 2 -pc_fieldsplit_type additive -fieldsplit_0_ksp_type fgmres -fieldsplit_0_pc_type fieldsplit -fieldsplit_0_pc_fieldsplit_type schur -fieldsplit_0_pc_fieldsplit_schur_factorization_type full -fieldsplit_0_fieldsplit_velocity_ksp_type preonly -fieldsplit_0_fieldsplit_velocity_pc_type lu -fieldsplit_0_fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_0_fieldsplit_pressure_pc_type jacobi -fieldsplit_temperature_ksp_type preonly -fieldsplit_temperature_pc_type lu -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view -show_solution 0'},
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -forcing_type constant -bc_type dirichlet -interpolate 1 -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_0_fields 0,1 -pc_fieldsplit_1_fields 2 -pc_fieldsplit_type additive -fieldsplit_0_ksp_type fgmres -fieldsplit_0_pc_type fieldsplit -fieldsplit_0_pc_fieldsplit_type schur -fieldsplit_0_pc_fieldsplit_schur_factorization_type full -fieldsplit_0_fieldsplit_velocity_ksp_type preonly -fieldsplit_0_fieldsplit_velocity_pc_type lu -fieldsplit_0_fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_0_fieldsplit_pressure_pc_type jacobi -fieldsplit_temperature_ksp_type preonly -fieldsplit_temperature_pc_type lu -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view -show_solution 0'},
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.0     -forcing_type linear   -bc_type dirichlet -interpolate 1 -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_0_fields 0,1 -pc_fieldsplit_1_fields 2 -pc_fieldsplit_type additive -fieldsplit_0_ksp_type fgmres -fieldsplit_0_pc_type fieldsplit -fieldsplit_0_pc_fieldsplit_type schur -fieldsplit_0_pc_fieldsplit_schur_factorization_type full -fieldsplit_0_fieldsplit_velocity_ksp_type preonly -fieldsplit_0_fieldsplit_velocity_pc_type lu -fieldsplit_0_fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_0_fieldsplit_pressure_pc_type jacobi -fieldsplit_temperature_ksp_type preonly -fieldsplit_temperature_pc_type lu -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view -show_solution 0'},
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -forcing_type linear   -bc_type dirichlet -interpolate 1 -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_0_fields 0,1 -pc_fieldsplit_1_fields 2 -pc_fieldsplit_type additive -fieldsplit_0_ksp_type fgmres -fieldsplit_0_pc_type fieldsplit -fieldsplit_0_pc_fieldsplit_type schur -fieldsplit_0_pc_fieldsplit_schur_factorization_type full -fieldsplit_0_fieldsplit_velocity_ksp_type preonly -fieldsplit_0_fieldsplit_velocity_pc_type lu -fieldsplit_0_fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_0_fieldsplit_pressure_pc_type jacobi -fieldsplit_temperature_ksp_type preonly -fieldsplit_temperature_pc_type lu -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view -show_solution 0'},
                                                               # 2D serial freeslip tests 5-7
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0     -forcing_type linear   -bc_type freeslip  -interpolate 1 -show_initial -dm_complex_print_fem 1',
                                                                'setup': './bin/pythonscripts/PetscGenerateFEMQuadrature.py 2 2 2 1 laplacian 2 1 1 1 gradient 2 1 1 1 identity src/snes/examples/tutorials/ex31.h'},
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.0     -forcing_type linear   -bc_type freeslip  -interpolate 1 -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_0_fields 0,1 -pc_fieldsplit_1_fields 2 -pc_fieldsplit_type additive -fieldsplit_0_ksp_type fgmres -fieldsplit_0_pc_type fieldsplit -fieldsplit_0_pc_fieldsplit_type schur -fieldsplit_0_pc_fieldsplit_schur_factorization_type full -fieldsplit_0_fieldsplit_velocity_ksp_type preonly -fieldsplit_0_fieldsplit_velocity_pc_type lu -fieldsplit_0_fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_0_fieldsplit_pressure_pc_type jacobi -fieldsplit_temperature_ksp_type preonly -fieldsplit_temperature_pc_type lu -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view -show_solution 0'},
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -forcing_type linear   -bc_type freeslip  -interpolate 1 -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_0_fields 0,1 -pc_fieldsplit_1_fields 2 -pc_fieldsplit_type additive -fieldsplit_0_ksp_type fgmres -fieldsplit_0_pc_type fieldsplit -fieldsplit_0_pc_fieldsplit_type schur -fieldsplit_0_pc_fieldsplit_schur_factorization_type full -fieldsplit_0_fieldsplit_velocity_ksp_type preonly -fieldsplit_0_fieldsplit_velocity_pc_type lu -fieldsplit_0_fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_0_fieldsplit_pressure_pc_type jacobi -fieldsplit_temperature_ksp_type preonly -fieldsplit_temperature_pc_type lu -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view -show_solution 0'}],
                        'src/snes/examples/tutorials/ex33':   [{'numProcs': 1, 'args': '-snes_converged_reason -snes_monitor_short'}],
                        'src/snes/examples/tutorials/ex52':   [# 2D Laplacian 0-3
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0 -compute_function',
                                                                'setup': './bin/pythonscripts/PetscGenerateFEMQuadrature.py 2 1 1 1 laplacian src/snes/examples/tutorials/ex52.h',
                                                                'source': ['src/snes/examples/tutorials/ex52_integrateElement.cu']},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0 -compute_function -batch'},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0 -compute_function -batch -gpu'},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0 -compute_function -batch -gpu -gpu_batches 2'},
                                                               # 2D Laplacian refined 4-8
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0625 -compute_function'},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0625 -compute_function -batch'},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0625 -compute_function -batch -gpu'},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0625 -compute_function -batch -gpu -gpu_batches 2'},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0625 -compute_function -batch -gpu -gpu_batches 4'},
                                                               # 2D Elasticity 9-12
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0 -compute_function -op_type elasticity',
                                                                'setup': './bin/pythonscripts/PetscGenerateFEMQuadrature.py 2 1 2 1 elasticity src/snes/examples/tutorials/ex52.h'},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0 -compute_function -op_type elasticity -batch'},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0 -compute_function -op_type elasticity -batch -gpu'},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0 -compute_function -op_type elasticity -batch -gpu -gpu_batches 2'},
                                                               # 2D Elasticity refined 13-17
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0625 -compute_function -op_type elasticity'},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0625 -compute_function -op_type elasticity -batch'},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0625 -compute_function -op_type elasticity -batch -gpu'},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0625 -compute_function -op_type elasticity -batch -gpu -gpu_batches 2'},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0625 -compute_function -op_type elasticity -batch -gpu -gpu_batches 4'},
                                                               # 3D Laplacian 18-20
                                                               {'numProcs': 1, 'args': '-dim 3 -dm_view -refinement_limit 0.0 -compute_function',
                                                                'setup': './bin/pythonscripts/PetscGenerateFEMQuadrature.py 3 1 1 1 laplacian src/snes/examples/tutorials/ex52.h'},
                                                               {'numProcs': 1, 'args': '-dim 3 -dm_view -refinement_limit 0.0 -compute_function -batch'},
                                                               {'numProcs': 1, 'args': '-dim 3 -dm_view -refinement_limit 0.0 -compute_function -batch -gpu'},
                                                               # 3D Laplacian refined 21-24
                                                               {'numProcs': 1, 'args': '-dim 3 -dm_view -refinement_limit 0.0125 -compute_function'},
                                                               {'numProcs': 1, 'args': '-dim 3 -dm_view -refinement_limit 0.0125 -compute_function -batch'},
                                                               {'numProcs': 1, 'args': '-dim 3 -dm_view -refinement_limit 0.0125 -compute_function -batch -gpu'},
                                                               {'numProcs': 1, 'args': '-dim 3 -dm_view -refinement_limit 0.0125 -compute_function -batch -gpu -gpu_batches 2'},
                                                               # 3D Elasticity 25-27
                                                               {'numProcs': 1, 'args': '-dim 3 -dm_view -refinement_limit 0.0 -compute_function -op_type elasticity',
                                                                'setup': './bin/pythonscripts/PetscGenerateFEMQuadrature.py 3 1 3 1 elasticity src/snes/examples/tutorials/ex52.h'},
                                                               {'numProcs': 1, 'args': '-dim 3 -dm_view -refinement_limit 0.0 -compute_function -op_type elasticity -batch'},
                                                               {'numProcs': 1, 'args': '-dim 3 -dm_view -refinement_limit 0.0 -compute_function -op_type elasticity -batch -gpu'},
                                                               # 3D Elasticity 28-31
                                                               {'numProcs': 1, 'args': '-dim 3 -dm_view -refinement_limit 0.0125 -compute_function -op_type elasticity'},
                                                               {'numProcs': 1, 'args': '-dim 3 -dm_view -refinement_limit 0.0125 -compute_function -op_type elasticity -batch'},
                                                               {'numProcs': 1, 'args': '-dim 3 -dm_view -refinement_limit 0.0125 -compute_function -op_type elasticity -batch -gpu'},
                                                               {'numProcs': 1, 'args': '-dim 3 -dm_view -refinement_limit 0.0125 -compute_function -op_type elasticity -batch -gpu -gpu_batches 2'},
                                                               # 'source': ['src/snes/examples/tutorials/ex52_integrateElement.cu']},
                                                               ],
                        'src/snes/examples/tutorials/ex56':   [{'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0 -bc_type dirichlet -interpolate 0 -show_initial -show_residual -show_jacobian',
                                                                'setup': './bin/pythonscripts/PetscGenerateFEMQuadrature.py 2 1 2 1 laplacian 2 1 1 1 gradient src/snes/examples/tutorials/ex56.h'},
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0 -bc_type dirichlet -interpolate 1 -show_initial -show_residual -show_jacobian'},
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0625 -bc_type dirichlet -interpolate 0 -show_initial -show_residual -show_jacobian'},
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -show_initial -show_residual -show_jacobian'},
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0 -bc_type dirichlet -interpolate 1 -show_initial -show_residual -show_jacobian',
                                                                'setup': './bin/pythonscripts/PetscGenerateFEMQuadrature.py 2 2 2 1 laplacian 2 1 1 1 gradient src/snes/examples/tutorials/ex56.h'},
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.0 -bc_type dirichlet -interpolate 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view'},
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -show_initial -show_residual -show_jacobian'},
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -refinement_limit 0.0 -bc_type dirichlet -interpolate 0 -show_initial -show_residual -show_jacobian',
                                                                'setup': './bin/pythonscripts/PetscGenerateFEMQuadrature.py 3 1 3 1 laplacian 3 1 1 1 gradient src/snes/examples/tutorials/ex56.h'},
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -refinement_limit 0.0 -bc_type dirichlet -interpolate 1 -show_initial -show_residual -show_jacobian'},
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -refinement_limit 0.1 -bc_type dirichlet -interpolate 0 -show_initial -show_residual -show_jacobian'},
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -refinement_limit 0.1 -bc_type dirichlet -interpolate 1 -show_initial -show_residual -show_jacobian'},
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -refinement_limit 0.0 -bc_type dirichlet -interpolate 1 -show_initial -show_residual -show_jacobian',
                                                                'setup': './bin/pythonscripts/PetscGenerateFEMQuadrature.py 3 2 3 1 laplacian 3 1 1 1 gradient src/snes/examples/tutorials/ex56.h'},
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -refinement_limit 0.1 -bc_type dirichlet -interpolate 1 -show_initial -show_residual -show_jacobian'}
],
                        'src/snes/examples/tutorials/ex57':   [{'numProcs': 1, 'args': '-run_type test -bc_type dirichlet -interpolate 0 -show_initial -show_residual -show_jacobian',
                                                                'setup': './bin/pythonscripts/PetscGenerateFEMQuadratureTensorProduct.py 2 1 2 1 laplacian 2 1 1 1 gradient src/snes/examples/tutorials/ex57.h'},
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -bc_type dirichlet -interpolate 0 -show_initial -show_residual -show_jacobian',
                                                                'setup': './bin/pythonscripts/PetscGenerateFEMQuadratureTensorProduct.py 3 1 3 1 laplacian 3 1 1 1 gradient src/snes/examples/tutorials/ex57.h'}],
                        'src/snes/examples/tutorials/ex62':   [# 2D serial P1 tests 0-3
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0    -bc_type dirichlet -interpolate 0 -show_initial -dm_complex_print_fem 1',
                                                                'setup': './bin/pythonscripts/PetscGenerateFEMQuadrature.py 2 1 2 1 laplacian 2 1 1 1 gradient src/snes/examples/tutorials/ex62.h'},
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0    -bc_type dirichlet -interpolate 1 -show_initial -dm_complex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0625 -bc_type dirichlet -interpolate 0 -show_initial -dm_complex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -show_initial -dm_complex_print_fem 1'},
                                                               # 2D serial P2 tests 4-5
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0    -bc_type dirichlet -interpolate 1 -show_initial -dm_complex_print_fem 1',
                                                                'setup': './bin/pythonscripts/PetscGenerateFEMQuadrature.py 2 2 2 1 laplacian 2 1 1 1 gradient src/snes/examples/tutorials/ex62.h'},

                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -show_initial -dm_complex_print_fem 1'},
                                                               # Parallel tests 6-17
                                                               {'numProcs': 2, 'args': '-run_type test -refinement_limit 0.0    -bc_type dirichlet -interpolate 0 -dm_complex_print_fem 1',
                                                                'setup': './bin/pythonscripts/PetscGenerateFEMQuadrature.py 2 1 2 1 laplacian 2 1 1 1 gradient src/snes/examples/tutorials/ex62.h'},
                                                               {'numProcs': 3, 'args': '-run_type test -refinement_limit 0.0    -bc_type dirichlet -interpolate 0 -dm_complex_print_fem 1'},
                                                               {'numProcs': 5, 'args': '-run_type test -refinement_limit 0.0    -bc_type dirichlet -interpolate 0 -dm_complex_print_fem 1'},
                                                               {'numProcs': 2, 'args': '-run_type test -refinement_limit 0.0    -bc_type dirichlet -interpolate 1 -dm_complex_print_fem 1'},
                                                               {'numProcs': 3, 'args': '-run_type test -refinement_limit 0.0    -bc_type dirichlet -interpolate 1 -dm_complex_print_fem 1'},
                                                               {'numProcs': 5, 'args': '-run_type test -refinement_limit 0.0    -bc_type dirichlet -interpolate 1 -dm_complex_print_fem 1'},
                                                               {'numProcs': 2, 'args': '-run_type test -refinement_limit 0.0625 -bc_type dirichlet -interpolate 0 -dm_complex_print_fem 1'},
                                                               {'numProcs': 3, 'args': '-run_type test -refinement_limit 0.0625 -bc_type dirichlet -interpolate 0 -dm_complex_print_fem 1'},
                                                               {'numProcs': 5, 'args': '-run_type test -refinement_limit 0.0625 -bc_type dirichlet -interpolate 0 -dm_complex_print_fem 1'},
                                                               {'numProcs': 2, 'args': '-run_type test -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -dm_complex_print_fem 1'},
                                                               {'numProcs': 3, 'args': '-run_type test -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -dm_complex_print_fem 1'},
                                                               {'numProcs': 5, 'args': '-run_type test -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -dm_complex_print_fem 1'},
                                                               # Full solutions 18-29
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -interpolate 0 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view'},
                                                               {'numProcs': 2, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -interpolate 0 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view'},
                                                               {'numProcs': 3, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -interpolate 0 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_converged_reason -snes_view'},
                                                               {'numProcs': 5, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -interpolate 0 -pc_type jacobi -ksp_rtol 1.0e-10 -snes_converged_reason -snes_view'},
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view'},
                                                               {'numProcs': 2, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view'},
                                                               {'numProcs': 3, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_converged_reason -snes_view'},
                                                               {'numProcs': 5, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_converged_reason -snes_view'},
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view',
                                                                'setup': './bin/pythonscripts/PetscGenerateFEMQuadrature.py 2 2 2 1 laplacian 2 1 1 1 gradient src/snes/examples/tutorials/ex62.h'},
                                                               {'numProcs': 2, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view'},
                                                               {'numProcs': 3, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_converged_reason -snes_view'},
                                                               {'numProcs': 5, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_converged_reason -snes_view'},
                                                               # Stokes preconditioners 30-36
                                                               #   Jacobi
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -ksp_gmres_restart 100 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view -show_solution 0',
                                                                'setup': './bin/pythonscripts/PetscGenerateFEMQuadrature.py 2 2 2 1 laplacian 2 1 1 1 gradient src/snes/examples/tutorials/ex62.h'},
                                                               #  Block diagonal \begin{pmatrix} A & 0 \\ 0 & I \end{pmatrix}
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type additive -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view -show_solution 0'},
                                                               #  Block triangular \begin{pmatrix} A & B \\ 0 & I \end{pmatrix}
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type multiplicative -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view -show_solution 0'},
                                                               #  Diagonal Schur complement \begin{pmatrix} A & 0 \\ 0 & S \end{pmatrix}
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type diag -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view -show_solution 0'},
                                                               #  Upper triangular Schur complement \begin{pmatrix} A & B \\ 0 & S \end{pmatrix}
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type upper -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view -show_solution 0'},
                                                               #  Lower triangular Schur complement \begin{pmatrix} A & B \\ 0 & S \end{pmatrix}
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type lower -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view -show_solution 0'},
                                                               #  Full Schur complement \begin{pmatrix} I & 0 \\ B^T A^{-1} & I \end{pmatrix} \begin{pmatrix} A & 0 \\ 0 & S \end{pmatrix} \begin{pmatrix} I & A^{-1} B \\ 0 & I \end{pmatrix}
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view -show_solution 0'},
                                                               #  SIMPLE \begin{pmatrix} I & 0 \\ B^T A^{-1} & I \end{pmatrix} \begin{pmatrix} A & 0 \\ 0 & B^T diag(A)^{-1} B \end{pmatrix} \begin{pmatrix} I & diag(A)^{-1} B \\ 0 & I \end{pmatrix}
                                                               #{'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -fieldsplit_pressure_inner_ksp_type preonly -fieldsplit_pressure_inner_pc_type jacobi -fieldsplit_pressure_upper_ksp_type preonly -fieldsplit_pressure_upper_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view -show_solution 0'},
                                                               #  SIMPLEC \begin{pmatrix} I & 0 \\ B^T A^{-1} & I \end{pmatrix} \begin{pmatrix} A & 0 \\ 0 & B^T rowsum(A)^{-1} B \end{pmatrix} \begin{pmatrix} I & rowsum(A)^{-1} B \\ 0 & I \end{pmatrix}
                                                               #{'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -fieldsplit_pressure_inner_ksp_type preonly -fieldsplit_pressure_inner_pc_type jacobi -fieldsplit_pressure_inner_pc_jacobi_rowsum -fieldsplit_pressure_upper_ksp_type preonly -fieldsplit_pressure_upper_pc_type jacobi -fieldsplit_pressure_upper_pc_jacobi_rowsum -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view -show_solution 0'},
                                                               # Stokes preconditioners with MF Jacobian action 37-42
                                                               #   Jacobi
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -jacobian_mf -ksp_gmres_restart 100 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view -show_solution 0'},
                                                               #  Block diagonal \begin{pmatrix} A & 0 \\ 0 & I \end{pmatrix}
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -jacobian_mf -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type additive -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view -show_solution 0'},
                                                               #  Block triangular \begin{pmatrix} A & B \\ 0 & I \end{pmatrix}
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -jacobian_mf -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type multiplicative -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view -show_solution 0'},
                                                               #  Diagonal Schur complement \begin{pmatrix} A & 0 \\ 0 & S \end{pmatrix}
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -jacobian_mf -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type diag -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view -show_solution 0'},
                                                               #  Upper triangular Schur complement \begin{pmatrix} A & B \\ 0 & S \end{pmatrix}
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -jacobian_mf -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type upper -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view -show_solution 0'},
                                                               #  Lower triangular Schur complement \begin{pmatrix} A & B \\ 0 & S \end{pmatrix}
                                                               #{'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -jacobian_mf -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type lower -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view -show_solution 0'},
                                                               #  Full Schur complement \begin{pmatrix} A & B \\ B^T & S \end{pmatrix}
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -jacobian_mf -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view -show_solution 0'},
                                                               # 3D serial P1 tests 43-45
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -refinement_limit 0.0    -bc_type dirichlet -interpolate 0 -show_initial -dm_complex_print_fem 1',
                                                                'setup': './bin/pythonscripts/PetscGenerateFEMQuadrature.py 3 1 3 1 laplacian 3 1 1 1 gradient src/snes/examples/tutorials/ex62.h'},
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -refinement_limit 0.0    -bc_type dirichlet -interpolate 1 -show_initial -dm_complex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -refinement_limit 0.0125 -bc_type dirichlet -interpolate 0 -show_initial -dm_complex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -refinement_limit 0.0125 -bc_type dirichlet -interpolate 1 -show_initial -dm_complex_print_fem 1'},
                                                               ],

                        'src/snes/examples/tutorials/ex67':   [{'numProcs': 1, 'args': '-dm_view -snes_monitor -ksp_monitor -snes_view',
                                                                'setup': 'bin/pythonscripts/PetscGenerateFEMQuadratureTensorProduct.py 2 1 2 1 laplacian 2 0 1 1 gradient src/snes/examples/tutorials/ex67.h'},
                                                               {'numProcs': 2, 'args': '-dm_view -snes_monitor -ksp_monitor -snes_view'}],
                        'src/snes/examples/tutorials/ex68':   [{'numProcs': 1, 'args': '-snes_monitor -ksp_monitor -snes_view'},
                                                               {'numProcs': 1, 'args': '-snes_monitor -ksp_monitor -snes_view -problem 2 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_detect_saddle_point -pc_fieldsplit_schur_fact_type lower -fieldsplit_0_pc_type lu -fieldsplit_1_ksp_rtol 1e-10'}],
                        'src/snes/examples/tutorials/ex70':   [{'numProcs': 1, 'args': '-nx 32 -ny 48 -ksp_type fgmres -ksp_initial_guess_nonzero -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type lower -user_ksp -ksp_monitor -ksp_view'},
                                                               {'numProcs': 2, 'args': '-nx 32 -ny 48 -ksp_type fgmres -ksp_initial_guess_nonzero -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type lower -user_ksp -ksp_monitor -ksp_view'}],
                        'src/snes/examples/tutorials/ex72':   [# 2D serial P1 tests 0-1
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0    -bc_type dirichlet -show_initial -dm_complex_print_fem 1 -show_jacobian',
                                                                'setup': './bin/pythonscripts/PetscGenerateFEMQuadratureTensorProduct.py 2 1 2 1 laplacian 2 1 1 1 gradient src/snes/examples/tutorials/ex72.h'},
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0625 -bc_type dirichlet -show_initial -dm_complex_print_fem 1 -show_jacobian'},
                                                               # 2D serial P2 tests 2-3
                                                               #{'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0    -bc_type dirichlet -show_initial -dm_complex_print_fem 1 -show_jacobian',
                                                               # 'setup': './bin/pythonscripts/PetscGenerateFEMQuadratureTensorProduct.py 2 2 2 1 laplacian 2 1 1 1 gradient src/snes/examples/tutorials/ex62.h'},

                                                               #{'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0625 -bc_type dirichlet -show_initial -dm_complex_print_fem 1 -show_jacobian'},
                                                               # Parallel tests 4-9
                                                               {'numProcs': 2, 'args': '-run_type test -refinement_limit 0.0    -bc_type dirichlet -show_jacobian',
                                                                'setup': './bin/pythonscripts/PetscGenerateFEMQuadratureTensorProduct.py 2 1 2 1 laplacian 2 1 1 1 gradient src/snes/examples/tutorials/ex72.h'},
                                                               {'numProcs': 3, 'args': '-run_type test -refinement_limit 0.0    -bc_type dirichlet -show_jacobian'},
                                                               {'numProcs': 2, 'args': '-run_type test -refinement_limit 0.0625 -bc_type dirichlet -show_jacobian'},
                                                               {'numProcs': 3, 'args': '-run_type test -refinement_limit 0.0625 -bc_type dirichlet -show_jacobian'},
                                                               {'numProcs': 5, 'args': '-run_type test -refinement_limit 0.0625 -bc_type dirichlet -show_jacobian'},
                                                               # Full solutions 10-17
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.0    -bc_type dirichlet -pc_type jacobi -ksp_rtol 1.0e-9 -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view'},
                                                               #{'numProcs': 1, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -pc_type jacobi -ksp_rtol 1.0e-9 -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view'},
                                                               #{'numProcs': 2, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -pc_type jacobi -ksp_rtol 1.0e-9 -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view'},
                                                               #{'numProcs': 3, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -pc_type jacobi -ksp_rtol 1.0e-9 -snes_converged_reason -snes_view'},
                                                               #{'numProcs': 5, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -pc_type jacobi -ksp_rtol 1.0e-9 -snes_converged_reason -snes_view'},
                                                               #{'numProcs': 1, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -pc_type jacobi -ksp_rtol 1.0e-9 -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view',
                                                               # 'setup': './bin/pythonscripts/PetscGenerateFEMQuadrature.py 2 2 2 1 laplacian 2 1 1 1 gradient src/snes/examples/tutorials/ex62.h'},
                                                               #{'numProcs': 2, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -pc_type jacobi -ksp_rtol 1.0e-9 -snes_monitor_short -ksp_monitor_short -snes_converged_reason -snes_view'},
                                                               #{'numProcs': 3, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -pc_type jacobi -ksp_rtol 1.0e-9 -snes_converged_reason -snes_view'},
                                                               #{'numProcs': 5, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -pc_type jacobi -ksp_rtol 1.0e-9 -snes_converged_reason -snes_view'}
],
                        'src/ts/examples/tutorials/ex18':      {'numProcs': 1, 'args': '-snes_mf -ts_monitor_solution -ts_monitor -snes_monitor'},
                        }

def noCheckCommand(command, status, output, error):
  ''' Do no check result'''
  return

class MakeParser(object):
  def __init__(self, maker):
    self.maker = maker
    return

  def getTargets(self, text):
    '''Extract all targets from a makefile
    - Returns a dictionary of target names that map to a tuple of (dependencies, action)'''
    import re
    rulePat   = re.compile('^([\w]+):(.*)')
    targets   = {}
    foundRule = False
    for line in text.split('\n'):
      if foundRule:
        if line.startswith('\t'):
          l = line.strip()
          if not l.startswith('#'): rule[2].append(l)
          continue
        else:
          targets[rule[0]] = (rule[1], rule[2])
          foundRule = False
      m = rulePat.match(line)
      if m:
        target    = m.group(1)
        deps      = [d for d in m.group(2).split(' ') if d and d.endswith('.o')]
        rule      = (target, deps, [])
        foundRule = True
    return targets

  def parseAction(self, lines):
    '''Parses a PETSc action
    - Return a dictionary for the portions of a run'''
    import re
    m = re.match('-@\$\{MPIEXEC\} -n (?P<numProcs>\d+) ./(?P<ex>ex\w+)(?P<args>[-.,\w ]+)>', lines[0])
    if not m:
      raise RuntimeError('Could not parse launch sequence:\n'+lines[0])
    comparison = 'Not present'
    m2 = None
    if len(lines) > 1:
      m2 = re.search('\$\{DIFF\} output/%s_(?P<num>\w+)\.out' % m.group('ex'), lines[1])
      comparsion = lines[1]
    if not m2:
      raise RuntimeError('Could not parse comparison:\n'+comparison+'\n pattern: '+'\$\{DIFF\} output/%s_(?P<num>\w+).out' % m.group('ex'))
    return {'numProcs': m.group('numProcs'), 'args': m.group('args'), 'num': m2.group('num')}

  def parseTest(self, filename, testTarget):
    '''Parse a PETSc test target
    - Returns a dictionary compatible with builder2.py for regression tests'''
    from distutils.sysconfig import parse_makefile
    makevars = parse_makefile(filename)
    with file(filename) as f:
      maketext = f.read()
    targets = self.getTargets(maketext)
    srcDir  = os.path.dirname(filename)
    regressionParameters = {}
    testTargets = [r for r in makevars.get(testTarget, '').split(' ') if r]
    examples    = [e for e in testTargets if e.endswith('.PETSc')]
    for ex in examples:
      base   = os.path.splitext(ex)[0]
      source = base+'.c'
      exc    = os.path.join(os.path.relpath(srcDir, self.maker.petscDir), base)
      runs   = [e for e in testTargets if e.startswith('run'+base)]
      regressionParameters[exc] = []
      for r in runs:
        if not r in targets:
          raise RuntimeError('Could not find rule:',r)
        else:
          try:
            run = self.parseAction(targets[r][1])
            regressionParameters[exc].append(run)
          except RuntimeError, e:
            self.maker.logPrint('ERROR in '+str(r)+' for source '+source+'\n'+str(e))
    return regressionParameters

  def extractTests(self, filename):
    '''Extract valid test targets in a PETSc makefile
    - returns a list of test targets'''
    from distutils.sysconfig import parse_makefile
    makevars = parse_makefile(filename)
    return [t for t in makevars.keys() if t.startswith('TESTEXAMPLES')]

localRegressionParameters = {}

def getRegressionParameters(maker, exampleDir):
  if not exampleDir in localRegressionParameters:
    filename = os.path.join(exampleDir, 'makefile')
    if os.path.exists(filename):
      # Should parse all compatible tests here
      localRegressionParameters[exampleDir] = MakeParser(maker).parseTest(filename, 'TESTEXAMPLES_C')
    else:
      localRegressionParameters[exampleDir] = {}
  return localRegressionParameters[exampleDir]

class Future(logger.Logger):
  def __init__(self, argDB, log, pipe, cmd, errorMsg = '', func = None):
    logger.Logger.__init__(self, argDB = argDB, log = log)
    self.setup()
    self.pipe     = pipe
    self.cmd      = cmd
    self.errorMsg = errorMsg
    self.funcs    = [func]
    self.cwd      = os.getcwd()
    return

  def addFunc(self, func):
    self.funcs.append(func)
    return

  def finish(self):
    (out, err) = self.pipe.communicate()
    ret = self.pipe.returncode
    if ret:
      #[os.remove(o) for o in objects if os.path.isfile(o)]
      self.logPrint(self.errorMsg, debugSection = 'screen')
      self.logPrint(cmd,           debugSection = 'screen')
      self.logPrint(out+err,       debugSection = 'screen')
    else:
      self.logPrint('Successful execution')
      self.logPrint(out+err)
    (self.out, self.store, self.ret) = (out, err, ret)
    output = []
    curDir = os.getcwd()
    os.chdir(self.cwd)
    for func in self.funcs:
      if not func is None:
        output += func()
    os.chdir(curDir)
    return output

class NullSourceDatabase(object):
  def __init__(self, verbose = 0):
    return

  def __len__(self):
    return 0

  def setNode(self, vertex, deps):
    return

  def updateNode(self, vertex):
    return

  def rebuild(self, vertex):
    return True

class SourceDatabaseDict(object):
  '''This can be replaced by the favorite software of Jed'''
  def __init__(self, verbose = 0):
    # Vertices are filenames
    #   Arcs indicate a dependence and are decorated with consistency markers
    self.dependencyGraph = {}
    self.verbose         = verbose
    return

  def __str__(self):
    return str(self.dependencyGraph)

  @staticmethod
  def marker(dep):
    import hashlib
    with file(dep) as f:
      mark = hashlib.sha1(f.read()).digest()
    return mark

  def setNode(self, vertex, deps):
    self.dependencyGraph[vertex] = [(dep, SourceDatabaseDict.marker(dep)) for dep in deps]
    return

  def updateNode(self, vertex):
    self.dependencyGraph[vertex] = [(dep, SourceDatabaseDict.marker(dep)) for dep,mark in self.dependencyGraph[vertex]]
    return

  def rebuildArc(self, vertex, dep, mark):
    import hashlib
    with file(dep) as f:
      newMark = hashlib.sha1(f.read()).digest()
    return not mark == newMark

  def rebuild(self, vertex):
    if self.verbose: print('Checking for rebuild of',vertex)
    try:
      for dep,mark in self.dependencyGraph[vertex]:
        if self.rebuildArc(vertex, dep, mark):
          if self.verbose: print('    dep',dep,'is changed')
          return True
    except KeyError:
      return True
    return False

class SourceNode:
  def __init__(self, filename, marker):
    self.filename = filename
    self.marker   = marker
    return

  def __str__(self):
    #return self.filename+' ('+str(self.marker)+')'
    return self.filename

  def __repr__(self):
    return self.__str__()

  def __getitem__(self, pos):
    if   pos == 0: return self.filename
    elif pos == 1: return self.marker
    raise IndexError

  def __eq__(self, n):
    return n.filename == self.filename and n.marker == self.marker

  def __hash__(self):
    return self.filename.__hash__()

class SourceDatabase(logger.Logger):
  '''This can be replaced by the favorite software of Jed'''
  def __init__(self, argDB, log):
    logger.Logger.__init__(self, argDB = argDB, log = log)
    self.setup()
    # Vertices are (filename, consistency marker) pairs
    #   Arcs indicate a dependence
    import graph
    self.dependencyGraph = graph.DirectedGraph()
    return

  def __len__(self):
    return len(self.dependencyGraph)

  def __str__(self):
    return str(self.dependencyGraph)

  @staticmethod
  def marker(dep):
    import hashlib
    if not os.path.isfile(dep):
      return 0
    with file(dep) as f:
      mark = hashlib.sha1(f.read()).digest()
    return mark

  @staticmethod
  def vertex(filename):
    return SourceNode(filename, SourceDatabase.marker(filename))

  def hasNode(self, filename):
    return len([v for v in self.dependencyGraph.vertices if v[0] == filename])

  def setNode(self, vertex, deps):
    self.dependencyGraph.addEdges(SourceDatabase.vertex(vertex), [SourceDatabase.vertex(dep) for dep in deps])
    return

  def removeNode(self, filename):
    self.dependencyGraph.removeVertex(self.vertex(filename))
    return

  def updateNode(self, vertex):
    # This currently makes no sense
    v = SourceDatabase.vertex(vertex)
    self.dependencyGraph.clearEdges(v, inOnly = True)
    self.dependencyGraph.addEdges([SourceDatabase.vertex(dep) for dep,mark in self.dependencyGraph.getEdges(v)[0]])
    return

  def rebuildArc(self, vertex, dep, mark):
    import hashlib
    with file(dep) as f:
      newMark = hashlib.sha1(f.read()).digest()
    return not mark == newMark

  def rebuild(self, vertex):
    self.logPrint('Checking for rebuild of '+str(vertex))
    v = SourceDatabase.vertex(vertex)
    try:
      if not os.path.isfile(vertex):
        self.logPrint('    %s does not exist' % vertex)
      for dep,mark in self.dependencyGraph.getEdges(v)[0]:
        if self.rebuildArc(vertex, dep, mark):
          self.logPrint('    dep '+str(dep)+' is changed')
          return True
    except KeyError as e:
      self.logPrint('    %s not in database' % vertex)
      return True
    return False

  def topologicalSort(self, predicate):
    import graph
    for vertex,marker in graph.DirectedGraph.topologicalSort(self.dependencyGraph):
      if predicate(vertex):
        yield vertex
    return

class DirectoryTreeWalker(logger.Logger):
  def __init__(self, argDB, log, configInfo, allowFortran = None, allowExamples = False):
    logger.Logger.__init__(self, argDB = argDB, log = log)
    self.configInfo = configInfo
    if allowFortran is None:
      self.allowFortran  = hasattr(self.configInfo.compilers, 'FC')
    else:
      self.allowFortran  = allowFortran
    self.allowExamples   = allowExamples
    self.setup()
    self.collectDefines()
    return

  def collectDefines(self):
    self.defines = {}
    self.defines.update(self.configInfo.base.defines)
    self.defines.update(self.configInfo.compilers.defines)
    self.defines.update(self.configInfo.libraryOptions.defines)
    for p in self.configInfo.framework.packages:
      self.defines.update(p.defines)
    return

  def checkSourceDir(self, dirname):
    '''Checks makefile to see if compiler is allowed to visit this directory for this configuration'''
    # Require makefile
    makename = os.path.join(dirname, 'makefile')
    if not os.path.isfile(makename):
      if os.path.isfile(os.path.join(dirname, 'Makefile')): self.logPrint('ERROR: Change Makefile to makefile in '+dirname, debugSection = 'screen')
      return False

    # Parse makefile
    import re
    reg = re.compile(' [ ]*')
    with file(makename) as f:
      for line in f.readlines():
        if not line.startswith('#requires'): continue
        # Remove leader and redundant spaces and split into names
        reqType, reqValue = reg.sub(' ', line[9:-1].strip()).split(' ')[0:2]
        # Check requirements
        if reqType == 'scalar':
          if not self.configInfo.scalarType.scalartype == reqValue:
            self.logPrint('Rejecting '+dirname+' because scalar type '+self.configInfo.scalarType.scalartype+' is not '+reqValue)
            return False
        elif reqType == 'language':
          if reqValue == 'CXXONLY' and self.configInfo.languages.clanguage == 'C':
            self.logPrint('Rejecting '+dirname+' because language is '+self.configInfo.languages.clanguage+' is not C++')
            return False
        elif reqType == 'precision':
          if not self.configInfo.scalarType.precision == reqValue:
            self.logPrint('Rejecting '+dirname+' because precision '+self.configInfo.scalarType.precision+' is not '+reqValue)
            return False
        elif reqType == 'function':
          if not reqValue in ['\'PETSC_'+f+'\'' for f in self.configInfo.functions.defines]:
            self.logPrint('Rejecting '+dirname+' because function '+reqValue+' does not exist')
            return False
        elif reqType == 'define':
          if not reqValue in ['\'PETSC_'+d+'\'' for d in self.defines]:
            self.logPrint('Rejecting '+dirname+' because define '+reqValue+' does not exist')
            return False
        elif reqType == 'package':
          if not self.allowFortran and reqValue in ['\'PETSC_HAVE_FORTRAN\'', '\'PETSC_USING_F90\'', '\'PETSC_USING_F2003\'']:
            self.logPrint('Rejecting '+dirname+' because fortran is not being used')
            return False
          elif not self.configInfo.libraryOptions.useLog and reqValue == '\'PETSC_USE_LOG\'':
            self.logPrint('Rejecting '+dirname+' because logging is turned off')
            return False
          elif not self.configInfo.libraryOptions.useFortranKernels and reqValue == '\'PETSC_USE_FORTRAN_KERNELS\'':
            self.logPrint('Rejecting '+dirname+' because fortran kernels are turned off')
            return False
          elif not self.configInfo.mpi.usingMPIUni and reqValue == '\'PETSC_HAVE_MPIUNI\'':
            self.logPrint('Rejecting '+dirname+' because we are not using MPIUNI')
            return False
          elif not reqValue in ['\'PETSC_HAVE_'+p.PACKAGE+'\'' for p in self.configInfo.framework.packages]:
            self.logPrint('Rejecting '+dirname+' because package '+reqValue+' is not installed')
            return False
        else:
          self.logPrint('ERROR: Invalid requirement type %s in %s' % (reqType, makename), debugSection = 'screen')
          return False
    return True

  def checkDir(self, dirname):
    '''Checks whether we should recurse into this directory
    - Excludes ftn-* and f90-* if self.allowFortran is False
    - Excludes examples directory if self.allowExamples is False
    - Excludes contrib, tutorials, and benchmarks directory
    - Otherwise calls checkSourceDir()'''
    base = os.path.basename(dirname)

    if base in ['examples', 'tutorials']:
      return self.allowExamples
    elif base in ['externalpackages', 'projects', 'benchmarks', 'contrib']:
      return False
    elif base in ['ftn-auto', 'ftn-custom', 'f90-custom']:
      return self.allowFortran
    return self.checkSourceDir(dirname)

  def walk(self, rootDir):
    if not self.checkDir(rootDir):
      self.logPrint('Nothing to be done in '+rootDir)
    for root, dirs, files in os.walk(rootDir):
      yield root, files
      for badDir in [d for d in dirs if not self.checkDir(os.path.join(root, d))]:
        dirs.remove(badDir)
    return

class SourceFileManager(logger.Logger):
  def __init__(self, argDB, log, configInfo):
    logger.Logger.__init__(self, argDB = argDB, log = log)
    self.configInfo = configInfo
    self.setup()
    return

  def getObjectName(self, source, objDir = None):
    '''Get object file name corresponding to a source file'''
    if objDir is None:
      compilerObj = self.configInfo.compiler['C'].getTarget(source)
    else:
      compilerObj = os.path.join(objDir, self.configInfo.compiler['C'].getTarget(os.path.basename(source)))
    return compilerObj

  def sortSourceFiles(self, fnames, objDir = None):
    '''Sorts source files by language (returns dictionary with language keys)'''
    cnames    = []
    cxxnames  = []
    cudanames = []
    f77names  = []
    f90names  = []
    for f in fnames:
      ext = os.path.splitext(f)[1]
      if ext == '.c':
        cnames.append(f)
      elif ext in ['.cxx', '.cpp', '.cc']:
        if self.configInfo.languages.clanguage == 'Cxx':
          cxxnames.append(f)
      elif ext == '.cu':
        cudanames.append(f)
      elif ext == '.F':
        if hasattr(self.configInfo.compilers, 'FC'):
          f77names.append(f)
      elif ext == '.F90':
        if hasattr(self.configInfo.compilers, 'FC') and self.configInfo.compilers.fortranIsF90:
          f90names.append(f)
    source = cnames+cxxnames+cudanames+f77names+f90names
    if self.argDB['maxSources'] >= 0:
      cnames    = cnames[:self.argDB['maxSources']]
      cxxnames  = cxxnames[:self.argDB['maxSources']]
      cudanames = cudanames[:self.argDB['maxSources']]
      f77names  = f77names[:self.argDB['maxSources']]
      f90names  = f90names[:self.argDB['maxSources']]
      source    = source[:self.argDB['maxSources']]
    return {'C': cnames, 'Cxx': cxxnames, 'CUDA': cudanames, 'F77': f77names, 'F90': f90names, 'Fortran': f77names+f90names, 'Objects': [self.getObjectName(s, objDir) for s in source]}

class DependencyBuilder(logger.Logger):
  def __init__(self, argDB, log, sourceManager, sourceDatabase, objDir):
    logger.Logger.__init__(self, argDB = argDB, log = log)
    self.sourceManager  = sourceManager
    self.sourceDatabase = sourceDatabase
    self.objDir         = objDir
    self.setup()
    return

  def readDependencyFile(self, dirname, source, depFile):
    '''Read *.d file with dependency information and store it in the source database
    - Source files depend on headers
    '''
    with file(depFile) as f:
      try:
        target, deps = f.read().split(':')
      except ValueError as e:
        self.logPrint('ERROR in dependency file %s: %s' % (depFile, str(e)))
    target = target.split()[0]
    assert(target == self.sourceManager.getObjectName(source))
    deps = [d for d in deps.replace('\\','').split() if not os.path.splitext(d)[1] == '.mod']
    if not os.path.basename(deps[0]) == source:
      raise RuntimeError('ERROR: first dependency %s should be %s' % (deps[0], source))
    self.sourceDatabase.setNode(os.path.join(dirname, source), [os.path.join(dirname, d) for d in deps[1:]])
    return

  def buildDependency(self, dirname, source):
    self.logPrint('Rebuilding dependency info for '+os.path.join(dirname, source))
    depFile = os.path.join(self.objDir, os.path.splitext(os.path.basename(source))[0]+'.d')
    if os.path.isfile(depFile):
      self.logPrint('Found dependency file '+depFile)
      self.readDependencyFile(dirname, source, depFile)
    return

  def buildDependencies(self, dirname, fnames):
    ''' This is run in a PETSc source directory'''
    self.logPrint('Building dependencies in '+dirname)
    oldDir = os.getcwd()
    os.chdir(dirname)
    sourceMap = self.sourceManager.sortSourceFiles(fnames, self.objDir)
    for language in sourceMap.keys():
      if language == 'Objects': continue
      for source in sourceMap[language]:
        self.buildDependency(dirname, source)
    os.chdir(oldDir)
    return

  def buildDependenciesF90(self):
    '''We have to hardcode F90 module dependencies (shaking fist)'''
    pdir = self.sourceManager.configInfo.petscdir.dir
    if self.sourceManager.configInfo.mpi.usingMPIUni:
      self.sourceDatabase.setNode(os.path.join(pdir, 'src', 'sys', 'f90-mod', 'petscsysmod.F'), [os.path.join(pdir, 'src', 'sys', 'mpiuni', 'f90-mod', 'mpiunimod.F')])
    self.sourceDatabase.setNode(os.path.join(pdir, 'src', 'vec',  'f90-mod', 'petscvecmod.F'),  [os.path.join(pdir, 'src', 'sys',  'f90-mod', 'petscsysmod.F')])
    self.sourceDatabase.setNode(os.path.join(pdir, 'src', 'mat',  'f90-mod', 'petscmatmod.F'),  [os.path.join(pdir, 'src', 'vec',  'f90-mod', 'petscvecmod.F')])
    self.sourceDatabase.setNode(os.path.join(pdir, 'src', 'dm',   'f90-mod', 'petscdmmod.F'),   [os.path.join(pdir, 'src', 'mat',  'f90-mod', 'petscmatmod.F')])
    self.sourceDatabase.setNode(os.path.join(pdir, 'src', 'ksp',  'f90-mod', 'petsckspmod.F'),  [os.path.join(pdir, 'src', 'dm',   'f90-mod', 'petscdmmod.F')])
    self.sourceDatabase.setNode(os.path.join(pdir, 'src', 'snes', 'f90-mod', 'petscsnesmod.F'), [os.path.join(pdir, 'src', 'ksp',  'f90-mod', 'petsckspmod.F')])
    self.sourceDatabase.setNode(os.path.join(pdir, 'src', 'ts',   'f90-mod', 'petsctsmod.F'),   [os.path.join(pdir, 'src', 'snes', 'f90-mod', 'petscsnesmod.F')])
    return

class PETScConfigureInfo(object):
  def __init__(self, framework):
    self.framework = framework
    self.setupModules()
    self.compiler = {}
    self.compiler['C'] = self.framework.getCompilerObject(self.languages.clanguage)
    self.compiler['C'].checkSetup()
    return

  def setupModules(self):
    self.mpi             = self.framework.require('config.packages.MPI',         None)
    self.base            = self.framework.require('config.base',                 None)
    self.setCompilers    = self.framework.require('config.setCompilers',         None)
    self.arch            = self.framework.require('PETSc.utilities.arch',        None)
    self.petscdir        = self.framework.require('PETSc.utilities.petscdir',    None)
    self.languages       = self.framework.require('PETSc.utilities.languages',   None)
    self.debugging       = self.framework.require('PETSc.utilities.debugging',   None)
    self.debuggers       = self.framework.require('PETSc.utilities.debuggers',   None)
    self.make            = self.framework.require('config.programs',        None)
    self.CHUD            = self.framework.require('PETSc.utilities.CHUD',        None)
    self.compilers       = self.framework.require('config.compilers',            None)
    self.types           = self.framework.require('config.types',                None)
    self.headers         = self.framework.require('config.headers',              None)
    self.functions       = self.framework.require('config.functions',            None)
    self.libraries       = self.framework.require('config.libraries',            None)
    self.scalarType      = self.framework.require('PETSc.utilities.scalarTypes', None)
    self.memAlign        = self.framework.require('PETSc.utilities.memAlign',    None)
    self.libraryOptions  = self.framework.require('PETSc.utilities.libraryOptions', None)
    self.fortrancpp      = self.framework.require('PETSc.utilities.fortranCPP', None)
    self.debuggers       = self.framework.require('PETSc.utilities.debuggers', None)
    self.sharedLibraries = self.framework.require('PETSc.utilities.sharedLibraries', None)
    self.sowing          = self.framework.require('PETSc.packages.sowing', None)
    return

class PETScMaker(script.Script):
 def findArch(self):
   import nargs
   arch = nargs.Arg.findArgument('arch', sys.argv[1:])
   if arch is None:
     arch = os.environ.get('PETSC_ARCH', None)
   if arch is None:
     raise RuntimeError('You must provide a valid PETSC_ARCH')
   return arch

 def __init__(self):
   import RDict
   import os

   argDB = RDict.RDict(None, None, 0, 0, readonly = True)
   self.petscDir = os.environ['PETSC_DIR']
   arch  = self.findArch()
   argDB.saveFilename = os.path.join(self.petscDir, arch, 'conf', 'RDict.db')
   argDB.load()
   script.Script.__init__(self, argDB = argDB)
   self.logName = 'make.log'
   #self.log = sys.stdout
   return

 def setupHelp(self, help):
   import nargs

   help = script.Script.setupHelp(self, help)
   #help.addArgument('PETScMaker', '-rootDir', nargs.ArgDir(None, os.environ['PETSC_DIR'], 'The root directory for this build', isTemporary = 1))
   help.addArgument('PETScMaker', '-rootDir', nargs.ArgDir(None, os.getcwd(), 'The root directory for this build', isTemporary = 1))
   help.addArgument('PETScMaker', '-arch', nargs.Arg(None, os.environ.get('PETSC_ARCH', None), 'The root directory for this build', isTemporary = 1))
   help.addArgument('PETScMaker', '-dryRun',  nargs.ArgBool(None, False, 'Only output what would be run', isTemporary = 1))
   help.addArgument('PETScMaker', '-dependencies',  nargs.ArgBool(None, True, 'Use dependencies to control build', isTemporary = 1))
   help.addArgument('PETScMaker', '-buildLibraries', nargs.ArgBool(None, True, 'Build the PETSc libraries', isTemporary = 1))
   help.addArgument('PETScMaker', '-buildArchive', nargs.ArgBool(None, False, 'Build an archive of the object files', isTemporary = 1))
   help.addArgument('PETScMaker', '-regressionTests', nargs.ArgBool(None, False, 'Only run regression tests', isTemporary = 1))
   help.addArgument('PETScMaker', '-rebuildDependencies', nargs.ArgBool(None, False, 'Force dependency information to be recalculated', isTemporary = 1))
   help.addArgument('PETScMaker', '-verbose', nargs.ArgInt(None, 0, 'The verbosity level', min = 0, isTemporary = 1))

   help.addArgument('PETScMaker', '-maxSources', nargs.ArgInt(None, -1, 'The maximum number of source files in a directory', min = -1, isTemporary = 1))
   return help

 def setup(self):
   '''
   - Loads configure information
   - Loads dependency information (unless it will be recalculated)
   '''
   script.Script.setup(self)
   if self.dryRun or self.verbose:
     self.debugSection = 'screen'
   else:
     self.debugSection = None
   self.rootDir = os.path.abspath(self.argDB['rootDir'])
   # Load configure information
   self.framework  = self.loadConfigure()
   self.configInfo = PETScConfigureInfo(self.framework)
   # Setup directories
   self.petscDir     = self.configInfo.petscdir.dir
   self.petscArch    = self.configInfo.arch.arch
   self.petscConfDir = os.path.join(self.petscDir, self.petscArch, 'conf')
   self.petscLibDir  = os.path.join(self.petscDir, self.petscArch, 'lib')
   # Setup source database
   self.sourceDBFilename = os.path.join(self.petscConfDir, 'source.db')
   # Setup subobjects
   self.sourceManager = SourceFileManager(self.argDB, self.log, self.configInfo)
   return

 def cleanupLog(self, framework, confDir):
   '''Move configure.log to PROJECT_ARCH/conf - and update configure.log.bkp in both locations appropriately'''
   import os

   self.log.flush()
   if hasattr(framework, 'logName'):
     logName         = framework.logName
   else:
     logName         = 'make.log'
   logFile           = os.path.join(self.petscDir, logName)
   logFileBkp        = logFile + '.bkp'
   logFileArchive    = os.path.join(confDir, logName)
   logFileArchiveBkp = logFileArchive + '.bkp'

   # Keep backup in $PROJECT_ARCH/conf location
   if os.path.isfile(logFileArchiveBkp): os.remove(logFileArchiveBkp)
   if os.path.isfile(logFileArchive):    os.rename(logFileArchive, logFileArchiveBkp)
   if os.path.isfile(logFile):
     shutil.copyfile(logFile, logFileArchive)
     os.remove(logFile)
   if os.path.isfile(logFileArchive):    os.symlink(logFileArchive, logFile)
   # If the old bkp is using the same $PROJECT_ARCH/conf, then update bkp link
   if os.path.realpath(logFileBkp) == os.path.realpath(logFileArchive):
     if os.path.isfile(logFileBkp):        os.remove(logFileBkp)
     if os.path.isfile(logFileArchiveBkp): os.symlink(logFileArchiveBkp, logFileBkp)
   return

 def cleanup(self):
   '''
   - Move logs to proper location
   - Save dependency information
   '''
   confDir = self.petscConfDir
   self.cleanupLog(self, confDir)
   if self.argDB['dependencies']:
     if hasattr(self, 'sourceDatabase') and len(self.sourceDatabase):
       import cPickle
       with file(self.sourceDBFilename, 'wb') as f:
         cPickle.dump(self.sourceDatabase, f)
   return

 @property
 def verbose(self):
   '''The verbosity level'''
   return self.argDB['verbose']

 @property
 def dryRun(self):
   '''Flag for only output of what would be run'''
   return self.argDB['dryRun']

 def getObjDir(self, libname):
   return os.path.join(self.petscDir, self.petscArch, 'lib', libname+'-obj')

 def getPackageInfo(self):
   '''Get package include and library information from configure data'''
   packageIncludes = []
   packageLibs     = []
   for p in self.configInfo.framework.packages:
     # Could put on compile line, self.addDefine('HAVE_'+i.PACKAGE, 1)
     if hasattr(p, 'lib'):
       if not isinstance(p.lib, list):
         packageLibs.append(p.lib)
       else:
         packageLibs.extend(p.lib)
     if hasattr(p, 'include'):
       if not isinstance(p.include, list):
         packageIncludes.append(p.include)
       else:
         packageIncludes.extend(p.include)
   packageLibs     = self.configInfo.libraries.toStringNoDupes(packageLibs+self.configInfo.libraries.math)
   packageIncludes = self.configInfo.headers.toStringNoDupes(packageIncludes)
   return packageIncludes, packageLibs

 def storeObjects(self, objects):
   presentObjects = []
   for obj in objects:
     locObj = os.path.basename(obj)
     self.logPrint('Moving %s to %s' % (locObj, obj))
     if not self.dryRun:
       if not os.path.isfile(locObj):
         print('ERROR: Missing object file',locObj)
         self.operationFailed = True
       else:
         shutil.move(locObj, obj)
         presentObjects.append(obj)
     else:
       presentObjects.append(obj)
   return presentObjects

 def compile(self, language, source, objDir = None):
   if not len(source):
     self.logPrint('Nothing to build', debugSection = self.debugSection)
     return
   self.configInfo.setCompilers.pushLanguage(language)
   packageIncludes, packageLibs = self.getPackageInfo()
   compiler = self.configInfo.setCompilers.getCompiler()
   objects  = [self.sourceManager.getObjectName(s, objDir) for s in source]
   includes = ['-I'+inc for inc in [os.path.join(self.petscDir, self.petscArch, 'include'), os.path.join(self.petscDir, 'include')]]
   flags    = []
   flags.append(self.configInfo.setCompilers.getCompilerFlags())                        # Add PCC_FLAGS
   flags.extend([self.configInfo.setCompilers.CPPFLAGS, self.configInfo.CHUD.CPPFLAGS]) # Add CPP_FLAGS
   if self.configInfo.compilers.generateDependencies[language]:
     flags.append(self.configInfo.compilers.dependenciesGenerationFlag[language])
   if not language == 'FC':
     flags.append('-D__INSDIR__='+os.getcwd().replace(self.petscDir, ''))               # Define __INSDIR__
   cmd      = ' '.join([compiler]+['-c']+includes+[packageIncludes]+flags+source)
   self.logWrite(cmd+'\n', debugSection = self.debugSection, forceScroll = True)
   if not self.dryRun:
     (output, error, status) = self.executeShellCommand(cmd, checkCommand = noCheckCommand, log=self.log)
     if status:
       self.operationFailed = True
       [os.remove(o) for o in objects if os.path.isfile(o)]
       self.logPrint('ERROR IN %s COMPILE ******************************' % language, debugSection='screen')
       self.logPrint(output+error, debugSection='screen')
   self.configInfo.setCompilers.popLanguage()
   objects = self.storeObjects(objects)
   deps    = [os.path.splitext(o)[0]+'.d' for o in objects if os.path.isfile(os.path.splitext(os.path.basename(o))[0]+'.d')]
   self.storeObjects(deps)
   return objects

 def compileC(self, source, objDir = None):
   return self.compile(self.configInfo.languages.clanguage, source, objDir)

 def compileCUDA(self, source, objDir = None):
   return self.compile('CUDA', source, objDir)

 def compileCxx(self, source, objDir = None):
   return self.compile('Cxx', source, objDir)

 def compileFortran(self, source, objDir = None):
   objects = self.compile('FC', source, objDir)
   # Copy any module files produced into the include directory
   for locMod in os.listdir(os.getcwd()):
     if os.path.splitext(locMod)[1] == '.mod':
       mod = os.path.join(self.petscDir, self.petscArch, 'include', locMod)
       self.logPrint('Moving F90 module %s to %s' % (locMod, mod))
       shutil.move(locMod, mod)
   return objects

 def runShellCommandParallel(self, command, cwd = None):
   import subprocess

   self.logWrite('Executing: %s\n' % (command,), debugSection = self.debugSection, forceScroll = True)
   pipe = subprocess.Popen(command, cwd=cwd, stdin=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           bufsize=-1, shell=True, universal_newlines=True)
   return pipe

 def compileParallel(self, language, source, objDir = None):
   if not len(source):
     self.logPrint('Nothing to build', debugSection = self.debugSection)
     return
   self.configInfo.setCompilers.pushLanguage(language)
   packageIncludes, packageLibs = self.getPackageInfo()
   compiler = self.configInfo.setCompilers.getCompiler()
   objects  = [self.sourceManager.getObjectName(s, objDir) for s in source]
   includes = ['-I'+inc for inc in [os.path.join(self.petscDir, self.petscArch, 'include'), os.path.join(self.petscDir, 'include')]]
   flags    = []
   flags.append(self.configInfo.setCompilers.getCompilerFlags())                        # Add PCC_FLAGS
   flags.extend([self.configInfo.setCompilers.CPPFLAGS, self.configInfo.CHUD.CPPFLAGS]) # Add CPP_FLAGS
   if self.configInfo.compilers.generateDependencies[language]:
     flags.append(self.configInfo.compilers.dependenciesGenerationFlag[language])
   if not language == 'FC':
     flags.append('-D__INSDIR__='+os.getcwd().replace(self.petscDir, ''))               # Define __INSDIR__
   cmd      = ' '.join([compiler]+['-c']+includes+[packageIncludes]+flags+source)
   if not self.dryRun:
     pipe = self.runShellCommandParallel(cmd)
   else:
     pipe = None
   self.configInfo.setCompilers.popLanguage()

   def store():
     objs = self.storeObjects(objects)
     deps = [os.path.splitext(o)[0]+'.d' for o in objs if os.path.isfile(os.path.splitext(os.path.basename(o))[0]+'.d')]
     self.storeObjects(deps)
     return objs
   return [Future(self.argDB, self.log, pipe, cmd, 'ERROR IN %s COMPILE ******************************' % language, store)]

 def compileCParallel(self, source, objDir = None):
   return self.compileParallel(self.configInfo.languages.clanguage, source, objDir)

 def compileCxxParallel(self, source, objDir = None):
   return self.compileParallel('Cxx', source, objDir)

 def compileFortranParallel(self, source, objDir = None):
   futures = self.compileParallel('FC', source, objDir)
   def func():
     # Copy any module files produced into the include directory
     for locMod in os.listdir(os.getcwd()):
       if os.path.splitext(locMod)[1] == '.mod':
         mod = os.path.join(self.petscDir, self.petscArch, 'include', locMod)
         self.logPrint('Moving F90 module %s to %s' % (locMod, mod))
         shutil.move(locMod, mod)
     return []
   futures[0].addFunc(func)
   return futures

 def ranlib(self, library):
   '''${ranlib} ${LIBNAME} '''
   lib = os.path.splitext(library)[0]+'.'+self.configInfo.setCompilers.AR_LIB_SUFFIX
   cmd = ' '.join([self.configInfo.setCompilers.RANLIB, lib])
   self.logPrint('Running ranlib on '+lib)
   if not self.dryRun:
     (output, error, status) = self.executeShellCommand(cmd, checkCommand = noCheckCommand, log=self.log)
     if status:
       self.operationFailed = True
       self.logPrint("ERROR IN RANLIB ******************************", debugSection='screen')
       self.logPrint(output+error, debugSection='screen')
   return

 def expandArchive(self, archive, objDir):
   [shutil.rmtree(p) for p in os.listdir(objDir)]
   oldDir = os.getcwd()
   os.chdir(objDir)
   self.executeShellCommand(self.setCompilers.AR+' x '+archive, log = self.log)
   os.chdir(oldDir)
   return

 def buildArchive(self, library, objects):
   '''${AR} ${AR_FLAGS} ${LIBNAME} $*.o'''
   lib = os.path.splitext(library)[0]+'.'+self.configInfo.setCompilers.AR_LIB_SUFFIX
   self.logPrint('Archiving files '+str(objects)+' into '+lib)
   self.logWrite('Building archive '+lib+'\n', debugSection = 'screen', forceScroll = True)
   if self.rootDir == self.petscDir:
     cmd = ' '.join([self.configInfo.setCompilers.AR, self.configInfo.setCompilers.FAST_AR_FLAGS, lib]+objects)
   else:
     cmd = ' '.join([self.configInfo.setCompilers.AR, self.configInfo.setCompilers.AR_FLAGS, lib]+objects)
   self.logWrite(cmd+'\n', debugSection = self.debugSection, forceScroll = True)
   if not self.dryRun:
     (output, error, status) = self.executeShellCommand(cmd, checkCommand = noCheckCommand, log=self.log)
     if status:
       self.operationFailed = True
       self.logPrint("ERROR IN ARCHIVE ******************************", debugSection='screen')
       self.logPrint(output+error, debugSection='screen')
   self.ranlib(library)
   return [library]

 def linkShared(self, sharedLib, libDir, tmpDir):
   osName = sys.platform
   # PCC_LINKER PCC_LINKER_FLAGS
   linker      = self.configInfo.setCompilers.getLinker()
   linkerFlags = self.configInfo.setCompilers.getLinkerFlags()
   packageIncludes, packageLibs = self.getPackageInfo()
   extraLibs = self.configInfo.libraries.toStringNoDupes(self.configInfo.compilers.flibs+self.configInfo.compilers.cxxlibs+self.configInfo.compilers.LIBS.split(' '))+self.configInfo.CHUD.LIBS
   sysLib      = ''
   sysLib.replace('-Wl,-rpath', '-L')
   externalLib = packageLibs+' '+extraLibs
   externalLib.replace('-Wl,-rpath', '-L')
   # Move this switch into the sharedLibrary module
   if self.configInfo.setCompilers.isSolaris() and self.configInfo.setCompilers.isGNU(self.configInfo.framework.getCompiler()):
     cmd = self.configInfo.setCompilers.LD+' -G -h '+os.path.basename(sharedLib)+' *.o -o '+sharedLib+' '+sysLib+' '+externalLib
     oldDir = os.getcwd()
     os.chdir(tmpDir)
     self.executeShellCommand(cmd, log=self.log)
     os.chdir(oldDir)
   elif '-qmkshrobj' in self.configInfo.setCompilers.sharedLibraryFlags:
     cmd = linker+' '+linkerFlags+' -qmkshrobj -o '+sharedLib+' *.o '+externalLib
     oldDir = os.getcwd()
     os.chdir(tmpDir)
     self.executeShellCommand(cmd, log=self.log)
     os.chdir(oldDir)
   else:
     if osName == 'linux2':
       cmd = linker+' -shared -Wl,-soname,'+os.path.basename(sharedLib)+' -o '+sharedLib+' *.o '+externalLib
     elif osName.startswith('darwin'):
       cmd   = ''
       flags = ''
       if not 'MACOSX_DEPLOYMENT_TARGET' in os.environ:
         cmd += 'MACOSX_DEPLOYMENT_TARGET=10.5 '
       if self.configInfo.setCompilers.getLinkerFlags().find('-Wl,-commons,use_dylibs') > -1:
         flags += '-Wl,-commons,use_dylibs'
       cmd += self.configInfo.setCompilers.getSharedLinker()+' -g  -dynamiclib -single_module -multiply_defined suppress -undefined dynamic_lookup '+flags+' -o '+sharedLib+' *.o -L'+libDir+' '+packageLibs+' '+sysLib+' '+extraLibs+' -lm -lc'
     elif osName == 'cygwin':
       cmd = linker+' '+linkerFlags+' -shared -o '+sharedLib+' *.o '+externalLib
     else:
       raise RuntimeError('Do not know how to make shared library for your crappy '+osName+' OS')
     oldDir = os.getcwd()
     os.chdir(tmpDir)
     (output, error, status) = self.executeShellCommand(cmd, checkCommand = noCheckCommand, log=self.log)
     if status:
       self.operationFailed = True
       self.logPrint("ERROR IN SHARED LIBRARY LINK ******************************", debugSection='screen')
       self.logPrint(output+error, debugSection='screen')
     os.chdir(oldDir)
     if hasattr(self.configInfo.debuggers, 'dsymutil'):
       cmd = self.configInfo.debuggers.dsymutil+' '+sharedLib
       self.executeShellCommand(cmd, log=self.log)
   return

 def buildSharedLibrary(self, libname, objects):
   '''
   PETSC_LIB_DIR        = ${PETSC_DIR}/${PETSC_ARCH}/lib
   INSTALL_LIB_DIR	= ${PETSC_LIB_DIR}
   '''
   if self.configInfo.sharedLibraries.useShared:
     libDir = self.petscLibDir
     objDir = self.getObjDir(libname)
     self.logPrint('Making shared libraries in '+libDir)
     sharedLib = os.path.join(libDir, os.path.splitext(libname)[0]+'.'+self.configInfo.setCompilers.sharedLibraryExt)
     rebuild   = False
     if os.path.isfile(sharedLib):
       for obj in objects:
         if os.path.getmtime(obj) >= os.path.getmtime(sharedLib):
           rebuild = True
           break
     else:
       rebuild = True
     if rebuild:
       self.logWrite('Building shared library '+sharedLib+'\n', debugSection = 'screen', forceScroll = True)
       self.linkShared(sharedLib, libDir, objDir)
     else:
       self.logPrint('Nothing to rebuild for shared library '+libname)
   else:
     self.logPrint('Shared libraries disabled')
   return

 def link(self, executable, objects, language):
   '''${CLINKER} -o $@ $^ ${PETSC_LIB}
      ${DSYMUTIL} $@'''
   self.logWrite('Linking object '+str(objects)+' into '+executable+'\n', debugSection = self.debugSection, forceScroll = True)
   self.configInfo.compilers.pushLanguage(language)
   packageIncludes, packageLibs = self.getPackageInfo()
   cmd = self.configInfo.compilers.getFullLinkerCmd(' '.join(objects)+' -L'+self.petscLibDir+' -lpetsc '+packageLibs+' -L/usr/local/cuda/lib', executable)
   if not self.dryRun:
     (output, error, status) = self.executeShellCommand(cmd, checkCommand = noCheckCommand, log=self.log)
     if status:
       self.logPrint("ERROR IN LINK ******************************", debugSection='screen')
       self.logPrint(output+error, debugSection='screen')
     # TODO: Move dsymutil stuff from PETSc.utilities.debuggers to config.compilers
     if hasattr(self.configInfo.debuggers, 'dsymutil'):
       (output, error, status) = self.executeShellCommand(self.configInfo.debuggers.dsymutil+' '+executable, checkCommand = noCheckCommand, log=self.log)
       if status:
         self.operationFailed = True
         self.logPrint("ERROR IN LINK ******************************", debugSection='screen')
         self.logPrint(output+error, debugSection='screen')
   self.configInfo.compilers.popLanguage()
   return [executable]

 def buildDir(self, dirname, files, objDir):
   ''' This is run in a PETSc source directory'''
   self.logWrite('Building in '+dirname+'\n', debugSection = 'screen', forceScroll = True)
   oldDir = os.getcwd()
   os.chdir(dirname)
   sourceMap = self.sourceManager.sortSourceFiles(files, objDir)
   objects   = []
   for language in ['C', 'Cxx', 'Fortran', 'CUDA']:
     if sourceMap[language]:
       self.logPrint('Compiling %s files %s' % (language, str(sourceMap[language])))
       objects.extend(getattr(self, 'compile'+language)(sourceMap[language], objDir))
   os.chdir(oldDir)
   return objects

 def buildDirParallel(self, dirname, files, objDir):
   ''' This is run in a PETSc source directory'''
   self.logWrite('Building in '+dirname+'\n', debugSection = 'screen', forceScroll = True)
   oldDir = os.getcwd()
   os.chdir(dirname)
   sourceMap = self.sourceManager.sortSourceFiles(files, objDir)
   futures   = []
   for language in ['C', 'Cxx', 'Fortran', 'CUDA']:
     if sourceMap[language]:
       self.logPrint('Compiling %s files %s' % (language, str(sourceMap[language])))
       futures.extend(getattr(self, 'compile'+language+'Parallel')(sourceMap[language], objDir))
   os.chdir(oldDir)
   return futures

 def buildFile(self, filename, objDir):
   ''' This is run in a PETSc source directory'''
   if not isinstance(filename, list): filename = [filename]
   self.logWrite('Building '+str(filename)+'\n', debugSection = 'screen', forceScroll = True)
   sourceMap = self.sourceManager.sortSourceFiles(filename, objDir)
   objects   = []
   for language in ['C', 'Cxx', 'Fortran', 'CUDA']:
     if sourceMap[language]:
       self.logPrint('Compiling %s files %s' % (language, str(sourceMap['C'])))
       objects.extend(getattr(self, 'compile'+language)(sourceMap[language], objDir))
   return objects

 def buildLibraries(self, libname, rootDir, parallel = False):
   '''TODO: If a file fails to build, it still must go in the source database'''
   if not self.argDB['buildLibraries']: return
   totalRebuild = rootDir == self.petscDir and not len(self.sourceDatabase)
   self.logPrint('Building Libraries')
   library = os.path.join(self.petscDir, self.petscArch, 'lib', libname)
   objDir  = self.getObjDir(libname)
   # Remove old library and object files by default when rebuilding the entire package
   if totalRebuild:
     self.logPrint('Doing a total rebuild of PETSc')
     lib = os.path.splitext(library)[0]+'.'+self.configInfo.setCompilers.AR_LIB_SUFFIX
     if os.path.isfile(lib):
       self.logPrint('Removing '+lib)
       os.unlink(lib)
     if os.path.isfile(self.sourceDBFilename):
       self.logPrint('Removing '+self.sourceDBFilename)
       os.unlink(self.sourceDBFilename)
     if os.path.isdir(objDir):
       shutil.rmtree(objDir)
   if not os.path.isdir(objDir): os.mkdir(objDir)

   self.operationFailed = False
   objects = []
   if len(self.sourceDatabase):
     def check(filename):
       if self.sourceDatabase.rebuild(filename): return True
       for obj in self.sourceManager.sortSourceFiles([filename], objDir)['Objects']:
         if not os.path.isfile(obj):
           self.logPrint('    object file '+obj+' is missing')
           return True
       return False
     import graph
     for filename in self.sourceDatabase.topologicalSort(check):
       objects += self.buildFile(filename, objDir)
   else:
     walker = DirectoryTreeWalker(self.argDB, self.log, self.configInfo)
     if totalRebuild:
       dirs = map(lambda d: os.path.join(rootDir, 'src', d), ['inline', 'sys', 'vec', 'mat', 'dm', 'ksp', 'snes', 'ts', 'docs'])
     else:
       dirs = [rootDir]
     if parallel:
       futures = []
       for d in dirs:
         for root, files in walker.walk(d):
           futures += self.buildDirParallel(root, files, objDir)
       for future in futures:
         objects += future.finish()
     else:
       for d in dirs:
         for root, files in walker.walk(d):
           objects += self.buildDir(root, files, objDir)

   if len(objects):
     if self.argDB['buildArchive']:
       self.buildArchive(library, objects)
     self.buildSharedLibrary(libname, objects)
   return len(objects)

 def rebuildDependencies(self, libname, rootDir):
   self.logWrite('Rebuilding Dependencies\n', debugSection = 'screen', forceScroll = True)
   self.sourceDatabase = SourceDatabase(self.argDB, self.log)
   depBuilder          = DependencyBuilder(self.argDB, self.log, self.sourceManager, self.sourceDatabase, self.getObjDir(libname))
   walker              = DirectoryTreeWalker(self.argDB, self.log, self.configInfo)

   for root, files in walker.walk(rootDir):
     depBuilder.buildDependencies(root, files)
   if not len(self.sourceDatabase):
     self.logPrint('No dependency information found -- disabling dependency tracking')
     self.sourceDatabase = NullSourceDatabase()
   else:
     depBuilder.buildDependenciesF90()
   if self.verbose > 3:
     import graph
     print('Source database:')
     for filename in self.sourceDatabase.topologicalSort(lambda x: True):
       print('  ',filename)
   return

 def updateDependencies(self, libname, rootDir):
   '''Calculates build dependencies and stores them in a database
   - If --dependencies is False, ignore them
   '''
   self.operationFailed = False
   if self.argDB['dependencies']:
     if not self.argDB['rebuildDependencies'] and os.path.isfile(self.sourceDBFilename):
       self.logPrint('Loading Dependencies')
       import cPickle

       with file(self.sourceDBFilename, 'rb') as f:
         self.sourceDatabase = cPickle.load(f)
       self.sourceDatabase.verbose = self.verbose
     else:
       self.rebuildDependencies(libname, rootDir)
   else:
     self.logPrint('Disabling dependency tracking')
     self.sourceDatabase = NullSourceDatabase()
   return

 def cleanupTest(self, dirname, execname):
   # ${RM} $* *.o $*.mon.* gmon.out mon.out *.exe *.ilk *.pdb *.tds
   import re
   trash = re.compile('^('+execname+'(\.o|\.mon\.\w+|\.exe|\.ilk|\.pdb|\.tds)?|g?mon.out)$')
   for fname in os.listdir(dirname):
     if trash.match(fname):
       os.remove(fname)
   return

 def checkTestOutput(self, testDir, executable, output, testNum):
   from difflib import unified_diff
   outputName = os.path.join(testDir, 'output', os.path.basename(executable)+'_'+str(testNum)+'.out')
   ret        = 0
   if not os.path.isfile(outputName):
     self.logPrint("MISCONFIGURATION: Regression output file %s (test %d) is missing" % (outputName, testNum), debugSection='screen')
   else:
     with file(outputName) as f:
       validOutput = f.read()
       if not validOutput == output:
         self.logPrint("TEST ERROR: Regression output for %s (test %s) does not match" % (executable, str(testNum)))
         for line in unified_diff(output.split('\n'), validOutput.split('\n'), fromfile='Current Output', tofile='Saved Output'):
           self.logPrint(line)
         self.logPrintDivider()
         self.logPrint(validOutput, indent = 0)
         self.logPrintDivider()
         self.logPrint(output, indent = 0)
         ret = -1
       else:
         self.logPrint("TEST SUCCESS: Regression output for %s (test %s) matches" % (executable, str(testNum)))
   return ret

 def getTestCommand(self, executable, **params):
   numProcs = params.get('numProcs', 1)
   args     = params.get('args', '')
   return ' '.join([self.configInfo.mpi.mpiexec, '-n', str(numProcs), os.path.abspath(executable), args])

 def runTest(self, testDir, executable, testNum, **params):
   cmd = self.getTestCommand(executable, **params)
   self.logPrint('Running test for '+executable)
   if not self.dryRun:
     (output, error, status) = self.executeShellCommand(cmd, checkCommand = noCheckCommand, log=self.log)
     if status:
       self.logPrint("TEST ERROR: Failed to execute %s\n" % executable)
       self.logPrint(output+error, indent = 0)
       ret = -2
     else:
       ret = self.checkTestOutput(testDir, executable, output+error, testNum)
   return ret

 def regressionTestsDir(self, dirname, dummy):
   ''' This is run in a PETSc source directory'''
   self.logWrite('Entering '+dirname+'\n', debugSection = 'screen', forceScroll = True)
   os.chdir(dirname)
   sourceMap = self.sourceManager.sortSourceFiles(dirname)
   objects   = []
   if sourceMap['C']:
     self.logPrint('Compiling C files '+str(sourceMap['C']))
     self.compileC(sourceMap['C'])
   if sourceMap['Fortran']:
     if not self.fortrancpp.fortranDatatypes:
       self.logPrint('Compiling Fortran files '+str(sourceMap['Fortran']))
       self.compileF(sourceMap['Fortran'])
   if sourceMap['Objects']:
     packageNames = set([p.name for p in self.framework.packages])
     for obj in sourceMap['Objects']:
       # TESTEXAMPLES_C_X = ex3.PETSc runex3 ex3.rm
       # .PETSc: filters out messages from build
       # .rm: cleans up test
       executable = os.path.splitext(obj)[0]
       paramKey   = os.path.relpath(os.path.abspath(executable), self.petscDir)
       testNum    = 1
       if paramKey in regressionRequirements:
         if not regressionRequirements[paramKey].issubset(packageNames):
           continue
       self.logPrint('Linking object '+obj+' into '+executable)
       # TODO: Fix this hack
       if executable[-1] == 'f':
         self.link(executable, obj, 'FC')
       else:
         self.link(executable, obj, self.languages.clanguage)
       self.runTest(dirname, executable, testNum, **regressionParameters.get(paramKey, {}))
       testNum += 1
       while '%s_%d' % (paramKey, testNum) in regressionParameters:
         self.runTest(dirname, executable, testNum, **regressionParameters.get('%s_%d' % (paramKey, testNum), {}))
         testNum += 1
       self.cleanupTest(dirname, executable)
   return

 def regressionTests(self, rootDir):
   if not self.argDB['regressionTests']: return
   if not self.checkDir(rootDir, allowExamples = True):
     self.logPrint('Nothing to be done')
   self.operationFailed = False
   for root, dirs, files in os.walk(rootDir):
     self.logPrint('Processing '+root)
     if 'examples' in dirs:
       for exroot, exdirs, exfiles in os.walk(os.path.join(root, 'examples')):
         self.logPrint('Processing '+exroot)
         print('  Testing in root',root)
         self.regressionTestsDir(exroot, exfiles)
         for badDir in [d for d in exdirs if not self.checkDir(os.path.join(exroot, d), allowExamples = True)]:
           exdirs.remove(badDir)
     for badDir in [d for d in dirs if not self.checkDir(os.path.join(root, d))]:
       dirs.remove(badDir)
   return

 def buildEtags(self):
   oldPath = sys.path
   sys.path.append(os.path.join(self.petscDir, 'bin', 'maint'))
   from generateetags import main
   main()
   os.system('find config -type f -name "*.py" | xargs etags -o TAGS_PYTHON')
   sys.path = oldPath
   return

 def buildFortranStubs(self):
   self.logWrite('Building Fortran stubs\n', debugSection = 'screen', forceScroll = True)
   oldPath = sys.path
   sys.path.append(os.path.join(self.petscDir, 'bin', 'maint'))
   from generatefortranstubs import main, processf90interfaces
   for d in os.listdir(os.path.join(self.petscDir, 'include', 'finclude', 'ftn-auto')):
     if d.endswith('-tmpdir'): shutil.rmtree(os.path.join(self.petscDir, 'include', 'finclude', 'ftn-auto', d))
   main(self.petscDir, self.configInfo.sowing.bfort, os.getcwd(),0)
   processf90interfaces(self.petscDir,0)
   for d in os.listdir(os.path.join(self.petscDir, 'include', 'finclude', 'ftn-auto')):
     if d.endswith('-tmpdir'): shutil.rmtree(os.path.join(self.petscDir, 'include', 'finclude', 'ftn-auto', d))
   sys.path = oldPath
   return

 def check(self):
   self.logWrite('Checking build\n', debugSection = 'screen', forceScroll = True)
   return

 def clean(self, libname):
   self.logWrite('Cleaning build\n', debugSection = 'screen', forceScroll = True)
   if os.path.isfile(self.sourceDBFilename):
     os.remove(self.sourceDBFilename)
     self.logPrint('Removed '+self.sourceDBFilename)
   if os.path.exists(self.getObjDir(libname)):
     shutil.rmtree(self.getObjDir(libname))
   self.logPrint('Removed '+self.getObjDir(libname))
   return

 def run(self):
   self.setup()
   #self.buildFortranStubs()
   self.updateDependencies('libpetsc', self.rootDir)
   if self.buildLibraries('libpetsc', self.rootDir):
     # This is overkill, but right now it is cheap
     self.rebuildDependencies('libpetsc', self.rootDir)
   self.regressionTests(self.rootDir)
   self.cleanup()
   return

if __name__ == '__main__':
  PETScMaker().run()
