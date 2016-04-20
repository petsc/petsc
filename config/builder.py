#!/usr/bin/env python

from __future__ import with_statement  # For python-2.5

import os, sys
import shutil
import tempfile

sys.path.insert(0, os.path.join(os.environ['PETSC_DIR'], 'config'))
sys.path.insert(0, os.path.join(os.environ['PETSC_DIR'], 'config', 'BuildSystem'))

import logger, script

regressionParameters = {'src/dm/impls/patch/examples/tests/ex1': [{'numProcs': 1, 'args': '-patch_size 2 -grid_size 6'},
                                                                  {'numProcs': 4, 'args': '-patch_size 2 -grid_size 4'},
                                                                  {'numProcs': 4, 'args': '-patch_size 2 -grid_size 4 -comm_size 1'},
                                                                  {'numProcs': 4, 'args': '-patch_size 2 -grid_size 4 -comm_size 2'},
                                                                  {'numProcs': 16, 'args': '-patch_size 2 -grid_size 8 -comm_size 2'}],
                        'src/mat/examples/tests/ex170':          [{'numProcs': 1, 'args': ''},
                                                                  {'numProcs': 1, 'args': '-testnum 1'},
                                                                  {'numProcs': 2, 'args': '-testnum 1'},
                                                                  {'numProcs': 1, 'args': '-testnum 2'},
                                                                  {'numProcs': 2, 'args': '-testnum 2'}],
                        'src/dm/impls/plex/examples/tests/ex1': [# CTetGen tests 0-1
                                                                 {'numProcs': 1, 'args': '-dim 3 -ctetgen_verbose 4 -dm_view ::ascii_info_detail -info -info_exclude null'},
                                                                 {'numProcs': 1, 'args': '-dim 3 -ctetgen_verbose 4 -refinement_limit 0.0625 -dm_view ::ascii_info_detail -info -info_exclude null'},
                                                                 # Test 2D LaTex and ASCII output 2-9
                                                                 {'numProcs': 1, 'args': '-dim 2 -dm_view ::ascii_latex'},
                                                                 {'numProcs': 1, 'args': '-dim 2 -dm_refine 1 -interpolate 1 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 2, 'args': '-dim 2 -dm_refine 1 -interpolate 1 -test_partition -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 2, 'args': '-dim 2 -dm_refine 1 -interpolate 1 -test_partition -dm_view ::ascii_latex'},
                                                                 {'numProcs': 1, 'args': '-dim 2 -cell_simplex 0 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 1, 'args': '-dim 2 -cell_simplex 0 -dm_refine 1 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 2, 'args': '-dim 2 -cell_simplex 0 -dm_refine 1 -interpolate 1 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 2, 'args': '-dim 2 -cell_simplex 0 -dm_refine 1 -interpolate 1 -test_partition -dm_view ::ascii_latex'},
                                                                 # CGNS tests 10-11 (need to find smaller test meshes)
                                                                 {'numProcs': 1, 'args': '-filename %(meshes)s/tut21.cgns -interpolate 1 -dm_view', 'requires': ['cgns']},
                                                                 {'numProcs': 1, 'args': '-filename %(meshes)s/StaticMixer.cgns -interpolate 1 -dm_view', 'requires': ['cgns', 'Broken']},
                                                                 # Gmsh test 12
                                                                 {'numProcs': 1, 'args': '-filename %(meshes)s/doublet-tet.msh -interpolate 1 -dm_view'},
                                                                 # Parallel refinement tests with overlap 13-14
                                                                 {'numProcs': 2, 'args': '-dim 2 -cell_simplex 1 -dm_refine 1 -interpolate 1 -test_partition -overlap 1 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 8, 'args': '-dim 2 -cell_simplex 1 -dm_refine 1 -interpolate 1 -test_partition -overlap 1 -dm_view ::ascii_info_detail'},
                                                                 # Gmsh mesh reader tests 15-18
                                                                 {'numProcs': 1, 'args': '-filename %(meshes)s/square.msh -interpolate 1 -dm_view'},
                                                                 {'numProcs': 1, 'args': '-filename %(meshes)s/square_bin.msh -interpolate 1 -dm_view'},
                                                                 {'numProcs': 3, 'args': '-filename %(meshes)s/square.msh -interpolate 1 -dm_view'},
                                                                 {'numProcs': 3, 'args': '-filename %(meshes)s/square_bin.msh -interpolate 1 -dm_view'},
                                                                 # Parallel simple partitioner tests 19-20
                                                                 {'numProcs': 2, 'args': '-dim 2 -cell_simplex 1 -dm_refine 0 -interpolate 0 -petscpartitioner_type simple -partition_view -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 8, 'args': '-dim 2 -cell_simplex 1 -dm_refine 1 -interpolate 1 -petscpartitioner_type simple -partition_view -dm_view ::ascii_info_detail'},
                                                                 # Fluent mesh reader tests 21-24
                                                                 {'numProcs': 1, 'args': '-filename %(meshes)s/square.cas -interpolate 1 -dm_view'},
                                                                 {'numProcs': 3, 'args': '-filename %(meshes)s/square.cas -interpolate 1 -dm_view'},
                                                                 {'numProcs': 1, 'args': '-filename %(meshes)s/cube_5tets_ascii.cas -interpolate 1 -dm_view'},
                                                                 {'numProcs': 1, 'args': '-filename %(meshes)s/cube_5tets.cas -interpolate 1 -dm_view', 'requires': 'Broken'},
                                                                 ],
                        'src/dm/impls/plex/examples/tests/ex3': [# 2D P_1 on a triangle
                                                                 {'num': 'p1_2d_0', 'numProcs': 1, 'args': '-petscspace_order 1 -num_comp 2 -qorder 1 -convergence'},
                                                                 {'num': 'p1_2d_1', 'numProcs': 1, 'args': '-petscspace_order 1 -num_comp 2 -qorder 1 -porder 1'},
                                                                 {'num': 'p1_2d_2', 'numProcs': 1, 'args': '-petscspace_order 1 -num_comp 2 -qorder 1 -porder 2'},
                                                                 {'num': 'p1_2d_3', 'numProcs': 1, 'args': '-petscspace_order 1 -num_comp 2 -qorder 1 -convergence -conv_refine 0', 'requires': 'pragmatic'},
                                                                 {'num': 'p1_2d_4', 'numProcs': 1, 'args': '-petscspace_order 1 -num_comp 2 -qorder 1 -porder 1 -conv_refine 0', 'requires': 'pragmatic'},
                                                                 {'num': 'p1_2d_5', 'numProcs': 1, 'args': '-petscspace_order 1 -num_comp 2 -qorder 1 -porder 2 -conv_refine 0', 'requires': 'pragmatic'},
                                                                 # 3D P_1 on a tetrahedron
                                                                 {'num': 'p1_3d_0', 'numProcs': 1, 'args': '-dim 3 -petscspace_order 1 -num_comp 3 -qorder 1 -convergence'},
                                                                 {'num': 'p1_3d_1', 'numProcs': 1, 'args': '-dim 3 -petscspace_order 1 -num_comp 3 -qorder 1 -porder 1'},
                                                                 {'num': 'p1_3d_2', 'numProcs': 1, 'args': '-dim 3 -petscspace_order 1 -num_comp 3 -qorder 1 -porder 2'},
                                                                 {'num': 'p1_3d_3', 'numProcs': 1, 'args': '-dim 3 -petscspace_order 1 -num_comp 3 -qorder 1 -convergence -conv_refine 0', 'requires': 'pragmatic'},
                                                                 {'num': 'p1_3d_4', 'numProcs': 1, 'args': '-dim 3 -petscspace_order 1 -num_comp 3 -qorder 1 -porder 1 -conv_refine 0', 'requires': 'pragmatic'},
                                                                 {'num': 'p1_3d_5', 'numProcs': 1, 'args': '-dim 3 -petscspace_order 1 -num_comp 3 -qorder 1 -porder 2 -conv_refine 0', 'requires': 'pragmatic'},
                                                                 # 2D P_2 on a triangle
                                                                 {'num': 'p2_2d_0', 'numProcs': 1, 'args': '-petscspace_order 2 -num_comp 2 -qorder 2 -convergence'},
                                                                 {'num': 'p2_2d_1', 'numProcs': 1, 'args': '-petscspace_order 2 -num_comp 2 -qorder 2 -porder 1'},
                                                                 {'num': 'p2_2d_2', 'numProcs': 1, 'args': '-petscspace_order 2 -num_comp 2 -qorder 2 -porder 2'},
                                                                 {'num': 'p2_2d_3', 'numProcs': 1, 'args': '-petscspace_order 2 -num_comp 2 -qorder 2 -convergence -conv_refine 0', 'requires': 'pragmatic'},
                                                                 {'num': 'p2_2d_4', 'numProcs': 1, 'args': '-petscspace_order 2 -num_comp 2 -qorder 2 -porder 1 -conv_refine 0', 'requires': 'pragmatic'},
                                                                 {'num': 'p2_2d_5', 'numProcs': 1, 'args': '-petscspace_order 2 -num_comp 2 -qorder 2 -porder 2 -conv_refine 0', 'requires': 'pragmatic'},
                                                                 # 3D P_2 on a tetrahedron
                                                                 {'num': 'p2_3d_0', 'numProcs': 1, 'args': '-dim 3 -petscspace_order 2 -num_comp 3 -qorder 2 -convergence'},
                                                                 {'num': 'p2_3d_1', 'numProcs': 1, 'args': '-dim 3 -petscspace_order 2 -num_comp 3 -qorder 2 -porder 1'},
                                                                 {'num': 'p2_3d_2', 'numProcs': 1, 'args': '-dim 3 -petscspace_order 2 -num_comp 3 -qorder 2 -porder 2'},
                                                                 {'num': 'p2_3d_3', 'numProcs': 1, 'args': '-dim 3 -petscspace_order 2 -num_comp 3 -qorder 2 -convergence -conv_refine 0', 'requires': 'pragmatic'},
                                                                 {'num': 'p2_3d_4', 'numProcs': 1, 'args': '-dim 3 -petscspace_order 2 -num_comp 3 -qorder 2 -porder 1 -conv_refine 0', 'requires': 'pragmatic'},
                                                                 {'num': 'p2_3d_5', 'numProcs': 1, 'args': '-dim 3 -petscspace_order 2 -num_comp 3 -qorder 2 -porder 2 -conv_refine 0', 'requires': 'pragmatic'},
                                                                 # 2D Q_1 on a quadrilaterial
                                                                 {'num': 'q1_2d_0', 'numProcs': 1, 'args': '-simplex 0 -petscspace_order 1 -petscspace_poly_tensor 1 -num_comp 2 -qorder 1 -convergence'},
                                                                 {'num': 'q1_2d_1', 'numProcs': 1, 'args': '-simplex 0 -petscspace_order 1 -petscspace_poly_tensor 1 -num_comp 2 -qorder 1 -porder 1'},
                                                                 {'num': 'q1_2d_2', 'numProcs': 1, 'args': '-simplex 0 -petscspace_order 1 -petscspace_poly_tensor 1 -num_comp 2 -qorder 1 -porder 2'},
                                                                 # 2D Q_2 on a quadrilaterial
                                                                 {'num': 'q2_2d_0', 'numProcs': 1, 'args': '-simplex 0 -petscspace_order 2 -petscspace_poly_tensor 1 -num_comp 2 -qorder 2 -convergence'},
                                                                 {'num': 'q2_2d_1', 'numProcs': 1, 'args': '-simplex 0 -petscspace_order 2 -petscspace_poly_tensor 1 -num_comp 2 -qorder 2 -porder 1'},
                                                                 {'num': 'q2_2d_2', 'numProcs': 1, 'args': '-simplex 0 -petscspace_order 2 -petscspace_poly_tensor 1 -num_comp 2 -qorder 2 -porder 2'},
                                                                 # 2D P_3 on a triangle
                                                                 {'num': 'p3_2d_0', 'numProcs': 1, 'args': '-petscspace_order 3 -num_comp 2 -qorder 3 -convergence', 'requires': 'Broken'},
                                                                 {'num': 'p3_2d_1', 'numProcs': 1, 'args': '-petscspace_order 3 -num_comp 2 -qorder 3 -porder 1', 'requires': 'Broken'},
                                                                 {'num': 'p3_2d_2', 'numProcs': 1, 'args': '-petscspace_order 3 -num_comp 2 -qorder 3 -porder 2', 'requires': 'Broken'},
                                                                 {'num': 'p3_2d_3', 'numProcs': 1, 'args': '-petscspace_order 3 -num_comp 2 -qorder 3 -porder 3', 'requires': 'Broken'},
                                                                 # 2D P_1disc on a triangle/quadrilateral
                                                                 {'num': 'p1d_2d_0', 'numProcs': 1, 'args': '-petscspace_order 1 -petscdualspace_lagrange_continuity 0 -num_comp 2 -qorder 1 -convergence'},
                                                                 {'num': 'p1d_2d_1', 'numProcs': 1, 'args': '-petscspace_order 1 -petscdualspace_lagrange_continuity 0 -num_comp 2 -qorder 1 -porder 1'},
                                                                 {'num': 'p1d_2d_2', 'numProcs': 1, 'args': '-petscspace_order 1 -petscdualspace_lagrange_continuity 0 -num_comp 2 -qorder 1 -porder 2'},
                                                                 {'num': 'p1d_2d_3', 'numProcs': 1, 'args': '-simplex 0 -petscspace_order 1 -petscdualspace_lagrange_continuity 0 -num_comp 2 -qorder 1 -convergence'},
                                                                 {'num': 'p1d_2d_4', 'numProcs': 1, 'args': '-simplex 0 -petscspace_order 1 -petscdualspace_lagrange_continuity 0 -num_comp 2 -qorder 1 -porder 1'},
                                                                 {'num': 'p1d_2d_5', 'numProcs': 1, 'args': '-simplex 0 -petscspace_order 1 -petscdualspace_lagrange_continuity 0 -num_comp 2 -qorder 1 -porder 2'},
                                                                 # Test quadrature 2D P_1 on a triangle
                                                                 {'num': 'p1_quad_2', 'numProcs': 1, 'args': '-petscspace_order 1 -num_comp 2 -porder 1 -qorder 2'},
                                                                 {'num': 'p1_quad_5', 'numProcs': 1, 'args': '-petscspace_order 1 -num_comp 2 -porder 1 -qorder 5'},
                                                                 # Test quadrature 2D P_2 on a triangle
                                                                 {'num': 'p2_quad_3', 'numProcs': 1, 'args': '-petscspace_order 2 -num_comp 2 -porder 2 -qorder 3'},
                                                                 {'num': 'p2_quad_5', 'numProcs': 1, 'args': '-petscspace_order 2 -num_comp 2 -porder 2 -qorder 5'},
                                                                 # Test quadrature 2D Q_1 on a quadrilateral
                                                                 {'num': 'q1_quad_2', 'numProcs': 1, 'args': '-simplex 0 -petscspace_order 1 -petscspace_poly_tensor 1 -num_comp 2 -porder 1 -qorder 2'},
                                                                 {'num': 'q1_quad_5', 'numProcs': 1, 'args': '-simplex 0 -petscspace_order 1 -petscspace_poly_tensor 1 -num_comp 2 -porder 1 -qorder 5'},
                                                                 # Test quadrature 2D Q_2 on a quadrilateral
                                                                 {'num': 'q2_quad_3', 'numProcs': 1, 'args': '-simplex 0 -petscspace_order 2 -petscspace_poly_tensor 1 -num_comp 2 -porder 1 -qorder 3'},
                                                                 {'num': 'q2_quad_5', 'numProcs': 1, 'args': '-simplex 0 -petscspace_order 2 -petscspace_poly_tensor 1 -num_comp 2 -porder 1 -qorder 5'},
                                                                 ],
                        'src/dm/impls/plex/examples/tests/ex4': [# 2D Simplex 0-3
                                                                 {'numProcs': 1, 'args': '-dim 2 -cell_hybrid 0 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 1, 'args': '-dim 2 -cell_hybrid 0 -num_refinements 1 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 2, 'args': '-dim 2 -cell_hybrid 0 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 2, 'args': '-dim 2 -cell_hybrid 0 -num_refinements 1 -dm_view ::ascii_info_detail'},
                                                                 # 2D Hybrid Simplex 4-7
                                                                 {'numProcs': 1, 'args': '-dim 2 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 1, 'args': '-dim 2 -num_refinements 1 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 2, 'args': '-dim 2 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 2, 'args': '-dim 2 -num_refinements 1 -dm_view ::ascii_info_detail'},
                                                                 # 3D Simplex 8-11
                                                                 {'numProcs': 1, 'args': '-dim 3 -cell_hybrid 0 -num_refinements 1 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 2, 'args': '-dim 3 -cell_hybrid 0 -num_refinements 1 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 1, 'args': '-dim 3 -cell_hybrid 0 -test_num 1 -num_refinements 1 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 2, 'args': '-dim 3 -cell_hybrid 0 -test_num 1 -num_refinements 1 -dm_view ::ascii_info_detail'},
                                                                 # 2D Quad 12-13
                                                                 {'numProcs': 1, 'args': '-dim 2 -cell_simplex 0 -cell_hybrid 0 -num_refinements 1 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 2, 'args': '-dim 2 -cell_simplex 0 -cell_hybrid 0 -num_refinements 1 -dm_view ::ascii_info_detail'},
                                                                 # 3D Hex 14-15
                                                                 {'numProcs': 1, 'args': '-dim 3 -cell_simplex 0 -cell_hybrid 0 -num_refinements 1 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 2, 'args': '-dim 3 -cell_simplex 0 -cell_hybrid 0 -num_refinements 1 -dm_view ::ascii_info_detail'},
                                                                 # 3D Hybrid Simplex 16-19
                                                                 {'numProcs': 1, 'args': '-dim 3 -num_refinements 1 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 2, 'args': '-dim 3 -num_refinements 1 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 1, 'args': '-dim 3 -test_num 1 -num_refinements 1 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 2, 'args': '-dim 3 -test_num 1 -num_refinements 1 -dm_view ::ascii_info_detail'},
                                                                 # 3D Hybrid Hex 20-23
                                                                 {'numProcs': 1, 'args': '-dim 3 -cell_simplex 0 -num_refinements 1 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 2, 'args': '-dim 3 -cell_simplex 0 -num_refinements 1 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 1, 'args': '-dim 3 -cell_simplex 0 -test_num 1 -num_refinements 1 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 2, 'args': '-dim 3 -cell_simplex 0 -test_num 1 -num_refinements 1 -dm_view ::ascii_info_detail'},
                                                                 # 2D Hybrid Simplex 24-24
                                                                 {'numProcs': 1, 'args': '-dim 2 -test_num 1 -num_refinements 1 -dm_view ::ascii_info_detail'},
                                                                 # 3D Multiple Refinement 25-26
                                                                 {'numProcs': 1, 'args': '-dim 3 -cell_hybrid 0 -test_num 2 -num_refinements 2 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 1, 'args': '-dim 3 -cell_simplex 0 -cell_hybrid 0 -test_num 1 -num_refinements 2 -dm_view ::ascii_info_detail'},
                                                                 # 2D Hybrid Quad 27-28
                                                                 {'numProcs': 1, 'args': '-dim 2 -cell_simplex 0 -num_refinements 1 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 2, 'args': '-dim 2 -cell_simplex 0 -num_refinements 1 -dm_view ::ascii_info_detail'},
                                                                 # 1D Simplex 29-31
                                                                 {'numProcs': 1, 'args': '-dim 1 -cell_hybrid 0 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 1, 'args': '-dim 1 -cell_hybrid 0 -num_refinements 1 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 1, 'args': '-dim 1 -cell_hybrid 0 -num_refinements 5 -dm_view ::ascii_info_detail'},
                                                                 # 2D Simplex 32-34
                                                                 {'numProcs': 1, 'args': '-dim 2 -cell_hybrid 0 -num_refinements 1 -uninterpolate -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 2, 'args': '-dim 2 -cell_hybrid 0 -num_refinements 1 -uninterpolate -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 2, 'args': '-dim 2 -cell_hybrid 0 -num_refinements 3 -uninterpolate -dm_view ::ascii_info_detail'},
                                                                 ],
                        'src/dm/impls/plex/examples/tests/ex5': [# 2D Simplex 0-1
                                                                 {'numProcs': 1, 'args': '-dim 2 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 2, 'args': '-dim 2 -dm_view ::ascii_info_detail'},
                                                                 # 2D Quads 2-3
                                                                 {'numProcs': 1, 'args': '-dim 2 -cell_simplex 0 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 2, 'args': '-dim 2 -cell_simplex 0 -dm_view ::ascii_info_detail'},
                                                                 # 3D Simplex 4-5
                                                                 {'numProcs': 1, 'args': '-dim 3 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 2, 'args': '-dim 3 -dm_view ::ascii_info_detail'},
                                                                 # 3D Hex 6-7
                                                                 {'numProcs': 1, 'args': '-dim 3 -cell_simplex 0 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 2, 'args': '-dim 3 -cell_simplex 0 -dm_view ::ascii_info_detail'},
                                                                 # Examples from PyLith 8-12
                                                                 {'numProcs': 1, 'args': '-dim 2 -test_num 1 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 1, 'args': '-dim 3 -test_num 1 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 1, 'args': '-dim 3 -cell_simplex 0 -test_num 1 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 1, 'args': '-dim 2 -cell_simplex 0 -test_num 1 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 1, 'args': '-dim 3 -cell_simplex 0 -test_num 2 -dm_view ::ascii_info_detail'},],
                        'src/dm/impls/plex/examples/tests/ex6': [{'numProcs': 1, 'args': '-malloc_dump'},
                                                                 {'numProcs': 1, 'args': '-malloc_dump -pend 10000'},
                                                                 {'numProcs': 1, 'args': '-malloc_dump -pend 10000 -fill 0.05'},
                                                                 {'numProcs': 1, 'args': '-malloc_dump -pend 10000 -fill 0.25'}],
                        'src/dm/impls/plex/examples/tests/ex7': [# Two cell test meshes 0-7
                                                                 {'numProcs': 1, 'args': '-dim 2 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 2, 'args': '-dim 2 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 1, 'args': '-dim 2 -cell_simplex 0 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 2, 'args': '-dim 2 -cell_simplex 0 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 1, 'args': '-dim 3 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 2, 'args': '-dim 3 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 1, 'args': '-dim 3 -cell_simplex 0 -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 2, 'args': '-dim 3 -cell_simplex 0 -dm_view ::ascii_info_detail'},
                                                                 # 2D Hybrid Mesh 8
                                                                 {'numProcs': 1, 'args': '-dim 2 -cell_simplex 0 -testnum 1 -dm_view ::ascii_info_detail'},
                                                                 # TetGen meshes 9-10
                                                                 {'numProcs': 1, 'args': '-dim 2 -use_generator -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 1, 'args': '-dim 3 -use_generator -dm_view ::ascii_info_detail'},
                                                                 # Cubit meshes 11
                                                                 {'numProcs': 1, 'args': '-dim 3 -filename %(meshes)s/blockcylinder-50.exo -dm_view ::ascii_info_detail', 'requires': 'exodusii'},
                                                                 #{'numProcs': 1, 'args': '-dim 3 -filename /PETSc3/petsc/blockcylinder-50.exo -dm_view ::ascii_info_detail'},
                                                                 #{'numProcs': 1, 'args': '-dim 3 -filename /PETSc3/petsc/blockcylinder-20.exo'},
                                                                 ],
                        'src/dm/impls/plex/examples/tests/ex8': [{'numProcs': 1, 'args': '-dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 1, 'args': '-interpolate -dm_view ::ascii_info_detail'},
                                                                 {'numProcs': 1, 'args': '-transform'},
                                                                 {'numProcs': 1, 'args': '-interpolate -transform'},
                                                                 {'numProcs': 1, 'args': '-run_type file -filename %(meshes)s/simpleblock-100.exo -dm_view ::ascii_info_detail -v0 -1.5,-0.5,0.5,-0.5,-0.5,0.5,0.5,-0.5,0.5 -J 0.0,0.0,0.5,0.0,0.5,0.0,-0.5,0.0,0.0,0.0,0.0,0.5,0.0,0.5,0.0,-0.5,0.0,0.0,0.0,0.0,0.5,0.0,0.5,0.0,-0.5,0.0,0.0 -invJ 0.0,0.0,-2.0,0.0,2.0,0.0,2.0,0.0,0.0,0.0,0.0,-2.0,0.0,2.0,0.0,2.0,0.0,0.0,0.0,0.0,-2.0,0.0,2.0,0.0,2.0,0.0,0.0 -detJ 0.125,0.125,0.125', 'requires': ['exodusii']},
                                                                 {'numProcs': 1, 'args': '-interpolate -run_type file -filename %(meshes)s/simpleblock-100.exo -dm_view ::ascii_info_detail -v0 -1.5,-0.5,0.5,-0.5,-0.5,0.5,0.5,-0.5,0.5 -J 0.0,0.0,0.5,0.0,0.5,0.0,-0.5,0.0,0.0,0.0,0.0,0.5,0.0,0.5,0.0,-0.5,0.0,0.0,0.0,0.0,0.5,0.0,0.5,0.0,-0.5,0.0,0.0 -invJ 0.0,0.0,-2.0,0.0,2.0,0.0,2.0,0.0,0.0,0.0,0.0,-2.0,0.0,2.0,0.0,2.0,0.0,0.0,0.0,0.0,-2.0,0.0,2.0,0.0,2.0,0.0,0.0 -detJ 0.125,0.125,0.125 -centroid -1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0 -normal 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 -vol 1.0,1.0,1.0', 'requires': ['exodusii']}],
                        'src/dm/impls/plex/examples/tests/ex9': [# 2D Simplex P_1 scalar tests
                                                                 {'numProcs': 1, 'args': '-num_dof 1,0,0 -iterations 10000 \
                                                                  -max_cone_time 1.1e-8 -max_closure_time 1.3e-7 -max_vec_closure_time 3.6e-7'},
                                                                 {'numProcs': 1, 'args': '-refinement_limit 1.0e-5 -num_dof 1,0,0 -iterations 2 \
                                                                  -max_cone_time 2.1e-8 -max_closure_time 1.5e-7 -max_vec_closure_time 3.6e-7'},
                                                                 {'numProcs': 1, 'args': '-num_fields 1 -num_components 1 -num_dof 1,0,0 -iterations 10000 \
                                                                  -max_cone_time 1.1e-8 -max_closure_time 1.3e-7 -max_vec_closure_time 4.5e-7'},
                                                                 {'numProcs': 1, 'args': '-refinement_limit 1.0e-5 -num_fields 1 -num_components 1 -num_dof 1,0,0 -iterations 2 \
                                                                  -max_cone_time 2.1e-8 -max_closure_time 1.5e-7 -max_vec_closure_time 4.7e-7'},
                                                                 {'numProcs': 1, 'args': '-interpolate -num_dof 1,0,0 -iterations 10000 \
                                                                  -max_cone_time 1.1e-8 -max_closure_time 6.5e-7 -max_vec_closure_time 1.0e-6'},
                                                                 {'numProcs': 1, 'args': '-interpolate -refinement_limit 1.0e-4 -num_dof 1,0,0 -iterations 2 \
                                                                  -max_cone_time 2.1e-8 -max_closure_time 6.5e-7 -max_vec_closure_time 1.0e-6'},
                                                                 {'numProcs': 1, 'args': '-interpolate -num_fields 1 -num_components 1 -num_dof 1,0,0 -iterations 10000 \
                                                                  -max_cone_time 1.1e-8 -max_closure_time 6.5e-7 -max_vec_closure_time 1.1e-6'},
                                                                 {'numProcs': 1, 'args': '-interpolate -refinement_limit 1.0e-4 -num_fields 1 -num_components 1 -num_dof 1,0,0 -iterations 2 \
                                                                  -max_cone_time 2.1e-8 -max_closure_time 6.5e-7 -max_vec_closure_time 1.2e-6'},
                                                                 # 2D Simplex P_1 vector tests
                                                                 # 2D Simplex P_2 scalar tests
                                                                 # 2D Simplex P_2 vector tests
                                                                 # 2D Simplex P_2/P_1 vector/scalar tests
                                                                 # 2D Quad P_1 scalar tests
                                                                 # 2D Quad P_1 vector tests
                                                                 # 2D Quad P_2 scalar tests
                                                                 # 2D Quad P_2 vector tests
                                                                 # 3D Simplex P_1 scalar tests
                                                                 # 3D Simplex P_1 vector tests
                                                                 # 3D Simplex P_2 scalar tests
                                                                 # 3D Simplex P_2 vector tests
                                                                 # 3D Hex P_1 scalar tests
                                                                 # 3D Hex P_1 vector tests
                                                                 # 3D Hex P_2 scalar tests
                                                                 # 3D Hex P_2 vector tests
                                                                 ],
                        'src/dm/impls/plex/examples/tests/ex10': [# Two cell tests
                                                                  {'numProcs': 1, 'args': '-dim 2 -interpolate 1 -cell_simplex 1 -num_dof 1,0,0 -mat_view'},
                                                                  {'numProcs': 1, 'args': '-dim 2 -interpolate 1 -cell_simplex 0 -num_dof 1,0,0 -mat_view'},
                                                                  {'numProcs': 1, 'args': '-dim 3 -interpolate 1 -cell_simplex 1 -num_dof 1,0,0,0 -mat_view'},
                                                                  {'numProcs': 1, 'args': '-dim 3 -interpolate 1 -cell_simplex 0 -num_dof 1,0,0,0 -mat_view'},
                                                                  # Refined tests
                                                                  {'numProcs': 1, 'args': '-dim 2 -interpolate 1 -cell_simplex 1 -refinement_limit 0.00625 -num_dof 1,0,0'},
                                                                  {'numProcs': 1, 'args': '-dim 2 -interpolate 1 -cell_simplex 0 -refinement_uniform       -num_dof 1,0,0'},
                                                                  {'numProcs': 1, 'args': '-dim 3 -interpolate 1 -cell_simplex 1 -refinement_limit 0.00625 -num_dof 1,0,0,0'},
                                                                  {'numProcs': 1, 'args': '-dim 3 -interpolate 1 -cell_simplex 0 -refinement_uniform       -num_dof 1,0,0,0'},
                                                                  # Parallel tests
                                                                  # Grouping tests
                                                                  {'num': 'group_1', 'numProcs': 1, 'args': '-num_groups 1 -num_dof 1,0,0 -is_view -orig_mat_view -perm_mat_view'},
                                                                  {'num': 'group_2', 'numProcs': 1, 'args': '-num_groups 2 -num_dof 1,0,0 -is_view -perm_mat_view'},
                                                                  ],
                        'src/dm/impls/plex/examples/tests/ex11': [{'numProcs': 1, 'args': ''},
                                                                  {'numProcs': 2, 'args': '-filename %(meshes)s/2Dgrd.exo -overlap 1', 'requires': ['Chaco']}],
                        'src/dm/impls/plex/examples/tests/ex12': [{'numProcs': 1, 'args': '-dm_view ascii:mesh.tex:ascii_latex'},
                                                                  # Parallel, no overlap tests 1-2
                                                                  {'numProcs': 3, 'args': '-test_partition -dm_view ::ascii_info_detail'},
                                                                  {'numProcs': 8, 'args': '-test_partition -dm_view ::ascii_info_detail'},
                                                                  # Parallel, level-1 overlap tests 3-4
                                                                  {'numProcs': 3, 'args': '-test_partition -overlap 1 -dm_view ::ascii_info_detail'},
                                                                  {'numProcs': 8, 'args': '-test_partition -overlap 1 -dm_view ::ascii_info_detail'},
                                                                  # Parallel, level-2 overlap test 5
                                                                  {'numProcs': 8, 'args': '-test_partition -overlap 2 -dm_view ::ascii_info_detail'},
                                                                  # Parallel load balancing, test 6-7
                                                                  {'numProcs': 2, 'args': '-test_partition -overlap 1 -dm_view ::ascii_info_detail'},
                                                                  {'numProcs': 2, 'args': '-test_partition -overlap 1 -load_balance -dm_view ::ascii_info_detail'},
                                                                  # Parallel redundant copying, test 8
                                                                  {'numProcs': 2, 'args': '-test_redundant -dm_view ::ascii_info_detail'}],
                        'src/dm/impls/plex/examples/tests/ex13': [{'numProcs': 1, 'args': '-test_partition 0 -dm_view ascii::ascii_info_detail -oriented_dm_view ascii::ascii_info_detail -orientation_view'},
                                                                  {'numProcs': 2, 'args': '-dm_view ascii::ascii_info_detail -oriented_dm_view ascii::ascii_info_detail -orientation_view'},
                                                                  {'numProcs': 2, 'args': '-test_num 1 -dm_view ascii::ascii_info_detail -oriented_dm_view ascii::ascii_info_detail -orientation_view'},
                                                                  {'numProcs': 3, 'args': '-dm_view ascii::ascii_info_detail -oriented_dm_view ascii::ascii_info_detail -orientation_view'},
                                                                  ],
                        'src/dm/impls/plex/examples/tests/ex14': [{'numProcs': 1, 'args': '-dm_view -dm_refine 1 -dm_coarsen'},
                                                                  ],
                        'src/dm/impls/plex/examples/tests/ex15': [{'numProcs': 2, 'args': '-verbose -globaltonatural_sf_view'},
                                                                  {'numProcs': 2, 'args': '-verbose -global_vec_view hdf5:V.h5:native -test_read'},
                                                                  ],
                        'src/dm/impls/plex/examples/tests/ex16': [{'numProcs': 1, 'args': '-dm_view ascii::ascii_info_detail'},
                                                                  ],
                        'src/dm/impls/plex/examples/tests/ex17': [{'numProcs': 1, 'args': '-test_partition 0 -dm_view ascii::ascii_info_detail'},
                                                                  ],
                        'src/dm/impls/plex/examples/tests/ex1f90': [{'numProcs': 1, 'args': ''}],
                        'src/dm/impls/plex/examples/tests/ex2f90': [{'numProcs': 1, 'args': ''}],
                        'src/dm/impls/plex/examples/tutorials/ex1': [{'numProcs': 1, 'args': ''},
                                                                     {'numProcs': 1, 'args': '-dim 3'},],
                        'src/dm/impls/plex/examples/tutorials/ex1f90': [{'numProcs': 1, 'args': ''},
                                                                        {'numProcs': 1, 'args': '-dim 3'},],
                        'src/dm/impls/plex/examples/tutorials/ex2': [# CGNS meshes 0-1
                                                                     {'numProcs': 1, 'args': '-filename %(meshes)s/tut21.cgns -interpolate 1', 'requires': ['cgns', 'Broken']},
                                                                     {'numProcs': 1, 'args': '-filename %(meshes)s/grid_c.cgns -interpolate 1', 'requires': ['cgns', 'Broken']},
                                                                     # Gmsh meshes 2-4
                                                                     {'numProcs': 1, 'args': '-filename %(meshes)s/doublet-tet.msh -interpolate 1'},
                                                                     {'numProcs': 1, 'args': '-filename %(meshes)s/square.msh -interpolate 1'},
                                                                     {'numProcs': 1, 'args': '-filename %(meshes)s/square_bin.msh -interpolate 1'},
                                                                     # Exodus meshes 5-9
                                                                     {'numProcs': 1, 'args': '-filename %(meshes)s/sevenside-quad.exo -interpolate 1', 'requires': ['exodusii']},
                                                                     {'numProcs': 1, 'args': '-filename %(meshes)s/sevenside-quad-15.exo -interpolate 1', 'requires': ['exodusii']},
                                                                     {'numProcs': 1, 'args': '-filename %(meshes)s/squaremotor-30.exo -interpolate 1', 'requires': ['exodusii']},
                                                                     {'numProcs': 1, 'args': '-filename %(meshes)s/blockcylinder-50.exo -interpolate 1', 'requires': ['exodusii']},
                                                                     {'numProcs': 1, 'args': '-filename %(meshes)s/simpleblock-100.exo -interpolate 1', 'requires': ['exodusii']},
                                                                     ],
                        'src/dm/impls/plex/examples/tutorials/ex5': [# Idempotence of saving/loading
                                                                     {'numProcs': 1, 'args': '-filename %(meshes)s/Rect-tri3.exo -dm_view ::ascii_info_detail', 'requires': ['exodusii']},
                                                                     {'numProcs': 2, 'args': '-filename %(meshes)s/Rect-tri3.exo -dm_view ::ascii_info_detail', 'requires': ['exodusii']},
                                                                     ],
                        'src/snes/examples/tutorials/ex5':   [# MISSING ALL TESTS FROM MAKEFILE
                                                               # MSM tests
                                                               {'num': 'asm_0', 'numProcs': 1, 'args': './ex5 -mms 1 -par 0.0 -snes_monitor -snes_converged_reason -snes_view -ksp_rtol 1.0e-9 -ksp_monitor -ksp_type richardson -pc_type asm -pc_asm_blocks 2 -pc_asm_overlap 0 -pc_asm_local_type additive -sub_pc_type lu'},
                                                               {'num': 'msm_0', 'numProcs': 1, 'args': './ex5 -mms 1 -par 0.0 -snes_monitor -snes_converged_reason -snes_view -ksp_rtol 1.0e-9 -ksp_monitor -ksp_type richardson -pc_type asm -pc_asm_blocks 2 -pc_asm_overlap 0 -pc_asm_local_type multiplicative -sub_pc_type lu'},
                                                               {'num': 'asm_1', 'numProcs': 1, 'args': './ex5 -mms 1 -par 0.0 -snes_monitor -snes_converged_reason -snes_view -ksp_rtol 1.0e-9 -ksp_monitor -ksp_type richardson -pc_type asm -pc_asm_blocks 2 -pc_asm_overlap 0 -pc_asm_local_type additive -sub_pc_type lu -da_grid_x 8'},
                                                               {'num': 'msm_1', 'numProcs': 1, 'args': './ex5 -mms 1 -par 0.0 -snes_monitor -snes_converged_reason -snes_view -ksp_rtol 1.0e-9 -ksp_monitor -ksp_type richardson -pc_type asm -pc_asm_blocks 2 -pc_asm_overlap 0 -pc_asm_local_type multiplicative -sub_pc_type lu -da_grid_x 8'},
                                                               {'num': 'asm_2', 'numProcs': 1, 'args': './ex5 -mms 1 -par 0.0 -snes_monitor -snes_converged_reason -snes_view -ksp_rtol 1.0e-9 -ksp_monitor -ksp_type richardson -pc_type asm -pc_asm_blocks 3 -pc_asm_overlap 0 -pc_asm_local_type additive -sub_pc_type lu -da_grid_x 8'},
                                                               {'num': 'msm_2', 'numProcs': 1, 'args': './ex5 -mms 1 -par 0.0 -snes_monitor -snes_converged_reason -snes_view -ksp_rtol 1.0e-9 -ksp_monitor -ksp_type richardson -pc_type asm -pc_asm_blocks 3 -pc_asm_overlap 0 -pc_asm_local_type multiplicative -sub_pc_type lu -da_grid_x 8'},
                                                               {'num': 'asm_3', 'numProcs': 1, 'args': './ex5 -mms 1 -par 0.0 -snes_monitor -snes_converged_reason -snes_view -ksp_rtol 1.0e-9 -ksp_monitor -ksp_type richardson -pc_type asm -pc_asm_blocks 4 -pc_asm_overlap 0 -pc_asm_local_type additive -sub_pc_type lu -da_grid_x 8'},
                                                               {'num': 'msm_3', 'numProcs': 1, 'args': './ex5 -mms 1 -par 0.0 -snes_monitor -snes_converged_reason -snes_view -ksp_rtol 1.0e-9 -ksp_monitor -ksp_type richardson -pc_type asm -pc_asm_blocks 4 -pc_asm_overlap 0 -pc_asm_local_type multiplicative -sub_pc_type lu -da_grid_x 8'},
                                                               {'num': 'asm_4', 'numProcs': 2, 'args': './ex5 -mms 1 -par 0.0 -snes_monitor -snes_converged_reason -snes_view -ksp_rtol 1.0e-9 -ksp_monitor -ksp_type richardson -pc_type asm -pc_asm_blocks 2 -pc_asm_overlap 0 -pc_asm_local_type additive -sub_pc_type lu -da_grid_x 8'},
                                                               {'num': 'msm_4', 'numProcs': 2, 'args': './ex5 -mms 1 -par 0.0 -snes_monitor -snes_converged_reason -snes_view -ksp_rtol 1.0e-9 -ksp_monitor -ksp_type richardson -pc_type asm -pc_asm_blocks 2 -pc_asm_overlap 0 -pc_asm_local_type multiplicative -sub_pc_type lu -da_grid_x 8'},
                                                               {'num': 'asm_5', 'numProcs': 2, 'args': './ex5 -mms 1 -par 0.0 -snes_monitor -snes_converged_reason -snes_view -ksp_rtol 1.0e-9 -ksp_monitor -ksp_type richardson -pc_type asm -pc_asm_blocks 4 -pc_asm_overlap 0 -pc_asm_local_type additive -sub_pc_type lu -da_grid_x 8'},
                                                               {'num': 'msm_5', 'numProcs': 2, 'args': './ex5 -mms 1 -par 0.0 -snes_monitor -snes_converged_reason -snes_view -ksp_rtol 1.0e-9 -ksp_monitor -ksp_type richardson -pc_type asm -pc_asm_blocks 4 -pc_asm_overlap 0 -pc_asm_local_type multiplicative -sub_pc_type lu -da_grid_x 8'},
                                                               ],
                        'src/snes/examples/tutorials/ex12':   [# 2D serial P1 test 0-4
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0    -bc_type dirichlet -interpolate 0 -petscspace_order 1 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0    -bc_type dirichlet -interpolate 1 -petscspace_order 1 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -petscspace_order 1 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0    -bc_type neumann   -interpolate 1 -petscspace_order 1 -bd_petscspace_order 1 -show_initial -dm_plex_print_fem 1 -dm_view ::ascii_info_detail'},
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0625 -bc_type neumann   -interpolate 1 -petscspace_order 1 -bd_petscspace_order 1 -show_initial -dm_plex_print_fem 1'},
                                                               # 2D serial P2 test 5-8
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0    -bc_type dirichlet -interpolate 1 -petscspace_order 2 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -petscspace_order 2 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0    -bc_type neumann   -interpolate 1 -petscspace_order 2 -bd_petscspace_order 2 -show_initial -dm_plex_print_fem 1 -dm_view ::ascii_info_detail'},
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0625 -bc_type neumann   -interpolate 1 -petscspace_order 2 -bd_petscspace_order 2 -show_initial -dm_plex_print_fem 1 -dm_view ::ascii_info_detail'},
                                                               # 3D serial P1 test 9-12
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -refinement_limit 0.0    -bc_type dirichlet -interpolate 0 -petscspace_order 1 -show_initial -dm_plex_print_fem 1 -dm_view'},
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -refinement_limit 0.0    -bc_type dirichlet -interpolate 1 -petscspace_order 1 -show_initial -dm_plex_print_fem 1 -dm_view'},
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -refinement_limit 0.0125 -bc_type dirichlet -interpolate 1 -petscspace_order 1 -show_initial -dm_plex_print_fem 1 -dm_view'},
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -refinement_limit 0.0    -bc_type neumann   -interpolate 1 -petscspace_order 1 -bd_petscspace_order 1 -snes_fd -show_initial -dm_plex_print_fem 1 -dm_view'},
                                                               # Analytic variable coefficient 13-20
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0    -variable_coefficient analytic -interpolate 1 -petscspace_order 1 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0625 -variable_coefficient analytic -interpolate 1 -petscspace_order 1 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0    -variable_coefficient analytic -interpolate 1 -petscspace_order 2 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0625 -variable_coefficient analytic -interpolate 1 -petscspace_order 2 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -refinement_limit 0.0    -variable_coefficient analytic -interpolate 1 -petscspace_order 1 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -refinement_limit 0.0125 -variable_coefficient analytic -interpolate 1 -petscspace_order 1 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -refinement_limit 0.0    -variable_coefficient analytic -interpolate 1 -petscspace_order 2 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -refinement_limit 0.0125 -variable_coefficient analytic -interpolate 1 -petscspace_order 2 -show_initial -dm_plex_print_fem 1'},
                                                               # P1 variable coefficient 21-28
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0    -variable_coefficient field    -interpolate 1 -petscspace_order 1 -mat_petscspace_order 1 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0625 -variable_coefficient field    -interpolate 1 -petscspace_order 1 -mat_petscspace_order 1 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0    -variable_coefficient field    -interpolate 1 -petscspace_order 2 -mat_petscspace_order 1 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0625 -variable_coefficient field    -interpolate 1 -petscspace_order 2 -mat_petscspace_order 1 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -refinement_limit 0.0    -variable_coefficient field    -interpolate 1 -petscspace_order 1 -mat_petscspace_order 1 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -refinement_limit 0.0125 -variable_coefficient field    -interpolate 1 -petscspace_order 1 -mat_petscspace_order 1 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -refinement_limit 0.0    -variable_coefficient field    -interpolate 1 -petscspace_order 2 -mat_petscspace_order 1 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -refinement_limit 0.0125 -variable_coefficient field    -interpolate 1 -petscspace_order 2 -mat_petscspace_order 1 -show_initial -dm_plex_print_fem 1'},
                                                               # P0 variable coefficient 29-36
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0    -variable_coefficient field    -interpolate 1 -petscspace_order 1 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0625 -variable_coefficient field    -interpolate 1 -petscspace_order 1 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0    -variable_coefficient field    -interpolate 1 -petscspace_order 2 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0625 -variable_coefficient field    -interpolate 1 -petscspace_order 2 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -refinement_limit 0.0    -variable_coefficient field    -interpolate 1 -petscspace_order 1 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -refinement_limit 0.0125 -variable_coefficient field    -interpolate 1 -petscspace_order 1 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -refinement_limit 0.0    -variable_coefficient field    -interpolate 1 -petscspace_order 2 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -refinement_limit 0.0125 -variable_coefficient field    -interpolate 1 -petscspace_order 2 -show_initial -dm_plex_print_fem 1'},
                                                               # Using ExodusII mesh 37-38 BROKEN
                                                               {'numProcs': 1, 'args': '-run_type test -f %(meshes)s/sevenside.exo -refinement_limit 0.0    -bc_type dirichlet -interpolate 1 -show_initial -dm_plex_print_fem 1 -dm_view',
                                                                'requires': ['exodusii', 'Broken']},
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -f /Users/knepley/Downloads/kis_modell_tet2.exo -refinement_limit 0.0    -bc_type dirichlet -interpolate 1 -show_initial -dm_plex_print_fem 1 -dm_view',
                                                                'requires': ['exodusii', 'Broken']},
                                                               # Full solve 39-44
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.015625 -interpolate 1 -petscspace_order 2 -pc_type gamg -ksp_rtol 1.0e-10 -ksp_monitor_short -ksp_converged_reason -snes_monitor_short -snes_converged_reason', 'parser': 'Solver'},
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.015625 -variable_coefficient nonlinear -interpolate 1 -petscspace_order 2 -pc_type svd -ksp_rtol 1.0e-10 -snes_monitor_short -snes_converged_reason'},
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.03125 -variable_coefficient nonlinear -interpolate 1 -petscspace_order 1 -snes_type fas -snes_fas_levels 2 -pc_type svd -ksp_rtol 1.0e-10 -fas_coarse_pc_type svd -fas_coarse_ksp_rtol 1.0e-10 -fas_coarse_snes_monitor_short -snes_monitor_short -snes_linesearch_type basic -fas_coarse_snes_linesearch_type basic -snes_converged_reason -dm_refine_hierarchy 1 -snes_view -fas_levels_1_snes_type newtonls -fas_levels_1_pc_type svd -fas_levels_1_ksp_rtol 1.0e-10 -fas_levels_1_snes_monitor_short'},
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.0625 -variable_coefficient nonlinear -interpolate 1 -petscspace_order 1 -snes_type fas -snes_fas_levels 3 -pc_type svd -ksp_rtol 1.0e-10 -fas_coarse_pc_type svd -fas_coarse_ksp_rtol 1.0e-10 -fas_coarse_snes_monitor_short -snes_monitor_short -snes_linesearch_type basic -fas_coarse_snes_linesearch_type basic -snes_converged_reason -dm_refine_hierarchy 2 -dm_plex_print_fem 0 -snes_view -fas_levels_1_snes_type newtonls -fas_levels_1_pc_type svd -fas_levels_1_ksp_rtol 1.0e-10 -fas_levels_1_snes_monitor_short -fas_levels_2_snes_type newtonls -fas_levels_2_pc_type svd -fas_levels_2_ksp_rtol 1.0e-10 -fas_levels_2_snes_monitor_short'},
                                                               {'numProcs': 2, 'args': '-run_type full -refinement_limit 0.03125 -variable_coefficient nonlinear -interpolate 1 -petscspace_order 1 -snes_type fas -snes_fas_levels 2 -pc_type svd -ksp_rtol 1.0e-10 -fas_coarse_pc_type svd -fas_coarse_ksp_rtol 1.0e-10 -fas_coarse_snes_monitor_short -snes_monitor_short -snes_linesearch_type basic -fas_coarse_snes_linesearch_type basic -snes_converged_reason -dm_refine_hierarchy 1 -snes_view -fas_levels_1_snes_type newtonls -fas_levels_1_pc_type svd -fas_levels_1_ksp_rtol 1.0e-10 -fas_levels_1_snes_monitor_short'},
                                                               {'numProcs': 2, 'args': '-run_type full -refinement_limit 0.0625 -variable_coefficient nonlinear -interpolate 1 -petscspace_order 1 -snes_type fas -snes_fas_levels 3 -pc_type svd -ksp_rtol 1.0e-10 -fas_coarse_pc_type svd -fas_coarse_ksp_rtol 1.0e-10 -fas_coarse_snes_monitor_short -snes_monitor_short -snes_linesearch_type basic -fas_coarse_snes_linesearch_type basic -snes_converged_reason -dm_refine_hierarchy 2 -dm_plex_print_fem 0 -snes_view -fas_levels_1_snes_type newtonls -fas_levels_1_pc_type svd -fas_levels_1_ksp_rtol 1.0e-10 -fas_levels_1_snes_monitor_short -fas_levels_2_snes_type newtonls -fas_levels_2_pc_type svd -fas_levels_2_ksp_rtol 1.0e-10 -fas_levels_2_snes_monitor_short'},
                                                               # Restarting 0-1
                                                               {'num': 'restart_0', 'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0    -bc_type dirichlet -interpolate 1 -petscspace_order 1 -dm_view hdf5:sol.h5 -vec_view hdf5:sol.h5::append'},
                                                               {'num': 'restart_1', 'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0    -bc_type dirichlet -interpolate 1 -petscspace_order 1 -f sol.h5 -restart', 'requires': 'exodusii'},
                                                               # Periodicity 0
                                                               {'num': 'periodic_0', 'numProcs': 1, 'args': '-run_type full -refinement_limit 0.0    -bc_type dirichlet -interpolate 1 -petscspace_order 1'},
                                                               # FAS
                                                               {'num': 'fas_newton_0', 'numProcs': 1, 'args': '-run_type full -variable_coefficient nonlinear -interpolate 1 -petscspace_order 1 -snes_type fas -snes_fas_levels 2 -pc_type svd -ksp_rtol 1.0e-10 -fas_coarse_pc_type svd -fas_coarse_ksp_rtol 1.0e-10 -fas_coarse_snes_monitor_short -snes_monitor_short -snes_linesearch_type basic -fas_coarse_snes_linesearch_type basic -snes_converged_reason -dm_refine_hierarchy 1 -snes_view -fas_levels_1_snes_type newtonls -fas_levels_1_pc_type svd -fas_levels_1_ksp_rtol 1.0e-10 -fas_levels_1_snes_monitor_short'},
                                                               {'num': 'fas_newton_1', 'numProcs': 1, 'args': '-run_type full -dm_refine_hierarchy 3 -interpolate 1 -petscspace_order 1 -snes_type fas -snes_fas_levels 3 -ksp_rtol 1.0e-10 -fas_coarse_pc_type lu -fas_coarse_snes_monitor_short -snes_monitor_short -snes_linesearch_type basic -fas_coarse_snes_linesearch_type basic -snes_converged_reason -snes_view -fas_levels_snes_type newtonls -fas_levels_snes_linesearch_type basic -fas_levels_ksp_rtol 1.0e-10 -fas_levels_snes_monitor_short'},
                                                               {'num': 'fas_ngs_0', 'numProcs': 1, 'args': '-run_type full -variable_coefficient nonlinear -interpolate 1 -petscspace_order 1 -snes_type fas -snes_fas_levels 2 -pc_type svd -ksp_rtol 1.0e-10 -fas_coarse_pc_type svd -fas_coarse_ksp_rtol 1.0e-10 -fas_coarse_snes_monitor_short -snes_monitor_short -snes_linesearch_type basic -fas_coarse_snes_linesearch_type basic -snes_converged_reason -dm_refine_hierarchy 1 -snes_view -fas_levels_1_snes_type ngs -fas_levels_1_snes_monitor_short'},
                                                               {'num': 'fas_newton_coarse_0', 'numProcs': 1, 'args': '-run_type full -dm_refine 2 -variable_coefficient nonlinear -interpolate 1 -petscspace_order 1 -snes_type fas -snes_fas_levels 2 -pc_type svd -ksp_rtol 1.0e-10 -fas_coarse_pc_type svd -fas_coarse_ksp_rtol 1.0e-10 -fas_coarse_snes_monitor_short -snes_monitor_short -snes_linesearch_type basic -fas_coarse_snes_linesearch_type basic -snes_converged_reason -dm_coarsen_hierarchy 1 -snes_view -fas_levels_1_snes_type newtonls -fas_levels_1_pc_type svd -fas_levels_1_ksp_rtol 1.0e-10 -fas_levels_1_snes_monitor_short',
                                                                'requires': ['pragmatic']},
                                                               {'num': 'mg_newton_coarse_0', 'numProcs': 1, 'args': '-run_type full -dm_refine 3 -interpolate 1 -petscspace_order 1 -snes_monitor_short -ksp_monitor_true_residual -snes_linesearch_type basic -snes_converged_reason -dm_coarsen_hierarchy 3 -snes_view -dm_view -ksp_type richardson -pc_type mg  -pc_mg_levels 4 -snes_atol 1.0e-8 -ksp_atol 1.0e-8 -snes_rtol 0.0 -ksp_rtol 0.0 -ksp_norm_type unpreconditioned -mg_levels_ksp_type gmres -mg_levels_pc_type ilu -mg_levels_ksp_max_it 10',
                                                                'requires': ['pragmatic']},
                                                               # Full runs with simplices
                                                               #   ASM
                                                               {'num': 'tri_q2q1_asm_lu', 'numProcs': 1, 'args': '-run_type full -dm_refine 3 -bc_type dirichlet -interpolate 1 -petscspace_order 1 -ksp_type gmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type asm -pc_asm_type restrict -pc_asm_blocks 4 -sub_pc_type lu -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view -show_solution 0', 'parser': 'Solver'},
                                                               {'num': 'tri_q2q1_msm_lu', 'numProcs': 1, 'args': '-run_type full -dm_refine 3 -bc_type dirichlet -interpolate 1 -petscspace_order 1 -ksp_type gmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type asm -pc_asm_type restrict -pc_asm_local_type multiplicative -pc_asm_blocks 4 -sub_pc_type lu -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view -show_solution 0', 'parser': 'Solver'},
                                                               {'num': 'tri_q2q1_asm_sor', 'numProcs': 1, 'args': '-run_type full -dm_refine 3 -bc_type dirichlet -interpolate 1 -petscspace_order 1 -ksp_type gmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type asm -pc_asm_type restrict -pc_asm_blocks 4 -sub_pc_type sor -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view -show_solution 0', 'parser': 'Solver'},
                                                               {'num': 'tri_q2q1_msm_sor', 'numProcs': 1, 'args': '-run_type full -dm_refine 3 -bc_type dirichlet -interpolate 1 -petscspace_order 1 -ksp_type gmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type asm -pc_asm_type restrict -pc_asm_local_type multiplicative -pc_asm_blocks 4 -sub_pc_type sor -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view -show_solution 0', 'parser': 'Solver'},
                                                               #   Convergence
                                                               {'num': 'tet_conv_p1_r0', 'numProcs': 1, 'args': '-run_type full -dim 3 -dm_refine 0 -bc_type dirichlet -interpolate 1 -petscspace_order 1 -dm_view -snes_converged_reason -pc_type lu'},
                                                               {'num': 'tet_conv_p1_r2', 'numProcs': 1, 'args': '-run_type full -dim 3 -dm_refine 2 -bc_type dirichlet -interpolate 1 -petscspace_order 1 -dm_view -snes_converged_reason -pc_type lu'},
                                                               {'num': 'tet_conv_p1_r3', 'numProcs': 1, 'args': '-run_type full -dim 3 -dm_refine 3 -bc_type dirichlet -interpolate 1 -petscspace_order 1 -dm_view -snes_converged_reason -pc_type lu'},
                                                               {'num': 'tet_conv_p2_r0', 'numProcs': 1, 'args': '-run_type full -dim 3 -dm_refine 0 -bc_type dirichlet -interpolate 1 -petscspace_order 2 -dm_view -snes_converged_reason -pc_type lu'},
                                                               {'num': 'tet_conv_p2_r2', 'numProcs': 1, 'args': '-run_type full -dim 3 -dm_refine 2 -bc_type dirichlet -interpolate 1 -petscspace_order 2 -dm_view -snes_converged_reason -pc_type lu'},
                                                               # tensor
                                                               {'num': 'tensor_plex_2d', 'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0 -simplex 0 -bc_type dirichlet -petscspace_order 1 -petscspace_poly_tensor -dm_refine_hierarchy 2'},
                                                               {'num': 'tensor_p4est_2d', 'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0 -simplex 0 -bc_type dirichlet -petscspace_order 1 -petscspace_poly_tensor -dm_forest_initial_refinement 2 -dm_forest_minimum_refinement 0 -dm_plex_convert_type p4est', 'requires' : ['p4est']},
                                                               {'num': 'tensor_plex_3d', 'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0 -simplex 0 -bc_type dirichlet -petscspace_order 1 -petscspace_poly_tensor -dm_refine_hierarchy 1 -dim 3'},
                                                               {'num': 'tensor_p4est_3d', 'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0 -simplex 0 -bc_type dirichlet -petscspace_order 1 -petscspace_poly_tensor -dm_forest_initial_refinement 1 -dm_forest_minimum_refinement 0 -dim 3 -dm_plex_convert_type p8est', 'requires' : ['p4est']},
                                                               # AMR
                                                               {'num': 'amr_0', 'numProcs': 5, 'args': '-run_type test -refinement_limit 0.0 -simplex 0 -bc_type dirichlet -petscspace_order 1 -petscspace_poly_tensor -dm_refine 1'},
                                                               {'num': 'amr_1', 'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0 -simplex 0 -bc_type dirichlet -petscspace_order 1 -petscspace_poly_tensor -dm_plex_convert_type p4est -dm_p4est_refine_pattern center -dm_forest_maximum_refinement 5 -dm_view vtk:amr.vtu:vtk_vtu -vec_view vtk:amr.vtu:vtk_vtu:append', 'requires' : ['p4est']},
                                                               {'num': 'p4est_test_q2_conformal_serial', 'numProcs': 1, 'args': '-run_type test -interpolate 1 -petscspace_order 2 -simplex 0 -petscspace_poly_tensor -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2', 'requires': ['p4est']},
                                                               {'num': 'p4est_test_q2_conformal_parallel', 'numProcs': 7, 'args': '-run_type test -interpolate 1 -petscspace_order 2 -simplex 0 -petscspace_poly_tensor -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2', 'requires': ['p4est']},
                                                               {'num': 'p4est_test_q2_nonconformal_serial', 'numProcs': 1, 'args': '-run_type test -interpolate 1 -petscspace_order 2 -simplex 0 -petscspace_poly_tensor -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash', 'requires': ['p4est']},
                                                               {'num': 'p4est_test_q2_nonconformal_parallel', 'numProcs': 7, 'args': '-run_type test -interpolate 1 -petscspace_order 2 -simplex 0 -petscspace_poly_tensor -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash', 'requires': ['p4est']},
                                                               {'num': 'p4est_exact_q2_conformal_serial', 'numProcs': 1, 'args': '-run_type exact -interpolate 1 -petscspace_order 2 -snes_max_it 1 -snes_type fas -snes_fas_levels 3 -pc_type none -ksp_type preonly -fas_coarse_pc_type none -fas_coarse_ksp_type preonly -fas_coarse_snes_monitor_short -snes_monitor_short -snes_linesearch_type basic -fas_coarse_snes_linesearch_type basic -snes_converged_reason -snes_view -fas_levels_snes_type newtonls -fas_levels_pc_type none -fas_levels_ksp_type preonly -fas_levels_snes_monitor_short -simplex 0 -petscspace_poly_tensor -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2', 'requires': ['p4est']},
                                                               {'num': 'p4est_exact_q2_conformal_parallel', 'numProcs': 7, 'args': '-run_type exact -interpolate 1 -petscspace_order 2 -snes_max_it 1 -snes_type fas -snes_fas_levels 3 -pc_type none -ksp_type preonly -fas_coarse_pc_type none -fas_coarse_ksp_type preonly -fas_coarse_snes_monitor_short -snes_monitor_short -snes_linesearch_type basic -fas_coarse_snes_linesearch_type basic -snes_converged_reason -snes_view -fas_levels_snes_type newtonls -fas_levels_pc_type none -fas_levels_ksp_type preonly -fas_levels_snes_monitor_short -simplex 0 -petscspace_poly_tensor -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2', 'requires': ['p4est']},
                                                               {'num': 'p4est_exact_q2_nonconformal_serial', 'numProcs': 1, 'args': '-run_type exact -interpolate 1 -petscspace_order 2 -snes_max_it 1 -snes_type fas -snes_fas_levels 3 -pc_type none -ksp_type preonly -fas_coarse_pc_type none -fas_coarse_ksp_type preonly -fas_coarse_snes_monitor_short -snes_monitor_short -snes_linesearch_type basic -fas_coarse_snes_linesearch_type basic -snes_converged_reason -snes_view -fas_levels_snes_type newtonls -fas_levels_pc_type none -fas_levels_ksp_type preonly -fas_levels_snes_monitor_short -simplex 0 -petscspace_poly_tensor -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash', 'requires': ['p4est']},
                                                               {'num': 'p4est_exact_q2_nonconformal_parallel', 'numProcs': 7, 'args': '-run_type exact -interpolate 1 -petscspace_order 2 -snes_max_it 1 -snes_type fas -snes_fas_levels 3 -pc_type none -ksp_type preonly -fas_coarse_pc_type none -fas_coarse_ksp_type preonly -fas_coarse_snes_monitor_short -snes_monitor_short -snes_linesearch_type basic -fas_coarse_snes_linesearch_type basic -snes_converged_reason -snes_view -fas_levels_snes_type newtonls -fas_levels_pc_type none -fas_levels_ksp_type preonly -fas_levels_snes_monitor_short -simplex 0 -petscspace_poly_tensor -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash', 'requires': ['p4est']},
                                                               {'num': 'p4est_full_q2_nonconformal_serial', 'numProcs': 1, 'args': '-run_type full -interpolate 1 -petscspace_order 2 -snes_max_it 20 -snes_type fas -snes_fas_levels 3 -pc_type jacobi -ksp_type cg -fas_coarse_pc_type jacobi -fas_coarse_ksp_type cg -fas_coarse_snes_monitor_short -snes_monitor_short -snes_linesearch_type basic -fas_coarse_snes_linesearch_type basic -snes_converged_reason -snes_view -fas_levels_snes_type newtonls -fas_levels_pc_type jacobi -fas_levels_ksp_type cg -fas_levels_snes_monitor_short -simplex 0 -petscspace_poly_tensor -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash', 'requires': ['p4est']},
                                                               {'num': 'p4est_full_q2_nonconformal_parallel', 'numProcs': 7, 'args': '-run_type full -interpolate 1 -petscspace_order 2 -snes_max_it 20 -snes_type fas -snes_fas_levels 3 -pc_type jacobi -ksp_type cg -fas_coarse_pc_type jacobi -fas_coarse_ksp_type cg -fas_coarse_snes_monitor_short -snes_monitor_short -snes_linesearch_type basic -fas_coarse_snes_linesearch_type basic -snes_converged_reason -snes_view -fas_levels_snes_type newtonls -fas_levels_pc_type jacobi -fas_levels_ksp_type cg -fas_levels_snes_monitor_short -simplex 0 -petscspace_poly_tensor -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash', 'requires': ['p4est']},
                                                               {'num': 'p4est_fas_q2_conformal_serial', 'numProcs': 1, 'args': '-run_type full -variable_coefficient nonlinear -interpolate 1 -petscspace_order 2 -snes_max_it 20 -snes_type fas -snes_fas_levels 3 -pc_type jacobi -ksp_type gmres -fas_coarse_pc_type svd -fas_coarse_ksp_type gmres -fas_coarse_snes_monitor_short -snes_monitor_short -snes_linesearch_type basic -fas_coarse_snes_linesearch_type basic -snes_converged_reason -snes_view -fas_levels_snes_type newtonls -fas_levels_pc_type svd -fas_levels_ksp_type gmres -fas_levels_snes_monitor_short -simplex 0 -petscspace_poly_tensor -dm_refine_hierarchy 3', 'requires': ['p4est']},
                                                               {'num': 'p4est_fas_q2_nonconformal_serial', 'numProcs': 1, 'args': '-run_type full -variable_coefficient nonlinear -interpolate 1 -petscspace_order 2 -snes_max_it 20 -snes_type fas -snes_fas_levels 3 -pc_type jacobi -ksp_type gmres -fas_coarse_pc_type jacobi -fas_coarse_ksp_type gmres -fas_coarse_ksp_monitor_true_residual -fas_coarse_snes_monitor_short -snes_monitor_short -snes_linesearch_type basic -fas_coarse_snes_linesearch_type basic -snes_converged_reason -snes_view -fas_levels_snes_type newtonls -fas_levels_pc_type jacobi -fas_levels_ksp_type gmres -fas_levels_snes_monitor_short -simplex 0 -petscspace_poly_tensor -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash', 'requires': ['p4est']},
                                                               {'num': 'fas_newton_0_p4est', 'numProcs': 1, 'args': '-run_type full -variable_coefficient nonlinear -interpolate 1 -petscspace_order 1 -snes_type fas -snes_fas_levels 2 -pc_type svd -ksp_rtol 1.0e-10 -fas_coarse_pc_type svd -fas_coarse_ksp_rtol 1.0e-10 -fas_coarse_snes_monitor_short -snes_monitor_short -snes_linesearch_type basic -fas_coarse_snes_linesearch_type basic -snes_converged_reason -snes_view -fas_levels_1_snes_type newtonls -fas_levels_1_pc_type svd -fas_levels_1_ksp_rtol 1.0e-10 -fas_levels_1_snes_monitor_short -simplex 0 -petscspace_poly_tensor -dm_plex_convert_type p4est -dm_forest_minimum_refinement 0 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash', 'requires' : ['p4est']},
                                                               ],
                        'src/snes/examples/tutorials/ex33':   [{'numProcs': 1, 'args': '-snes_converged_reason -snes_monitor_short'}],
                        'src/snes/examples/tutorials/ex36':   [# 2D serial P2/P1 tests 0-1
                                                               {'numProcs': 1, 'args': ''}],
                        'src/snes/examples/tutorials/ex52':   [# 2D Laplacian 0-3
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0 -petscspace_order 1 -compute_function', 'requires': ['cuda']},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0 -petscspace_order 1 -compute_function -batch', 'requires': ['cuda']},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0 -petscspace_order 1 -compute_function -batch -gpu', 'requires': ['cuda']},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0 -petscspace_order 1 -compute_function -batch -gpu -gpu_batches 2', 'requires': ['cuda']},
                                                               # 2D Laplacian refined 4-8
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0625 -petscspace_order 1 -compute_function', 'requires': ['cuda']},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0625 -petscspace_order 1 -compute_function -batch', 'requires': ['cuda']},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0625 -petscspace_order 1 -compute_function -batch -gpu', 'requires': ['cuda']},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0625 -petscspace_order 1 -compute_function -batch -gpu -gpu_batches 2', 'requires': ['cuda']},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0625 -petscspace_order 1 -compute_function -batch -gpu -gpu_batches 4', 'requires': ['cuda']},
                                                               # 2D Elasticity 9-12
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0 -petscspace_order 1 -compute_function -op_type elasticity', 'requires': ['cuda']},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0 -petscspace_order 1 -compute_function -op_type elasticity -batch', 'requires': ['cuda']},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0 -petscspace_order 1 -compute_function -op_type elasticity -batch -gpu', 'requires': ['cuda']},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0 -petscspace_order 1 -compute_function -op_type elasticity -batch -gpu -gpu_batches 2', 'requires': ['cuda']},
                                                               # 2D Elasticity refined 13-17
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0625 -petscspace_order 1 -compute_function -op_type elasticity', 'requires': ['cuda']},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0625 -petscspace_order 1 -compute_function -op_type elasticity -batch', 'requires': ['cuda']},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0625 -petscspace_order 1 -compute_function -op_type elasticity -batch -gpu', 'requires': ['cuda']},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0625 -petscspace_order 1 -compute_function -op_type elasticity -batch -gpu -gpu_batches 2', 'requires': ['cuda']},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0625 -petscspace_order 1 -compute_function -op_type elasticity -batch -gpu -gpu_batches 4', 'requires': ['cuda']},
                                                               # 3D Laplacian 18-20
                                                               {'numProcs': 1, 'args': '-dim 3 -dm_view -refinement_limit 0.0 -petscspace_order 1 -compute_function', 'requires': ['cuda']},
                                                               {'numProcs': 1, 'args': '-dim 3 -dm_view -refinement_limit 0.0 -petscspace_order 1 -compute_function -batch', 'requires': ['cuda']},
                                                               {'numProcs': 1, 'args': '-dim 3 -dm_view -refinement_limit 0.0 -petscspace_order 1 -compute_function -batch -gpu', 'requires': ['cuda']},
                                                               # 3D Laplacian refined 21-24
                                                               {'numProcs': 1, 'args': '-dim 3 -dm_view -refinement_limit 0.0125 -petscspace_order 1 -compute_function', 'requires': ['cuda']},
                                                               {'numProcs': 1, 'args': '-dim 3 -dm_view -refinement_limit 0.0125 -petscspace_order 1 -compute_function -batch', 'requires': ['cuda']},
                                                               {'numProcs': 1, 'args': '-dim 3 -dm_view -refinement_limit 0.0125 -petscspace_order 1 -compute_function -batch -gpu', 'requires': ['cuda']},
                                                               {'numProcs': 1, 'args': '-dim 3 -dm_view -refinement_limit 0.0125 -petscspace_order 1 -compute_function -batch -gpu -gpu_batches 2', 'requires': ['cuda']},
                                                               # 3D Elasticity 25-27
                                                               {'numProcs': 1, 'args': '-dim 3 -dm_view -refinement_limit 0.0 -petscspace_order 1 -compute_function -op_type elasticity',
                                                                'requires': ['cuda']},
                                                               {'numProcs': 1, 'args': '-dim 3 -dm_view -refinement_limit 0.0 -petscspace_order 1 -compute_function -op_type elasticity -batch', 'requires': ['cuda']},
                                                               {'numProcs': 1, 'args': '-dim 3 -dm_view -refinement_limit 0.0 -petscspace_order 1 -compute_function -op_type elasticity -batch -gpu', 'requires': ['cuda']},
                                                               # 3D Elasticity refined 28-31
                                                               {'numProcs': 1, 'args': '-dim 3 -dm_view -refinement_limit 0.0125 -petscspace_order 1 -compute_function -op_type elasticity', 'requires': ['cuda']},
                                                               {'numProcs': 1, 'args': '-dim 3 -dm_view -refinement_limit 0.0125 -petscspace_order 1 -compute_function -op_type elasticity -batch', 'requires': ['cuda']},
                                                               {'numProcs': 1, 'args': '-dim 3 -dm_view -refinement_limit 0.0125 -petscspace_order 1 -compute_function -op_type elasticity -batch -gpu', 'requires': ['cuda']},
                                                               {'numProcs': 1, 'args': '-dim 3 -dm_view -refinement_limit 0.0125 -petscspace_order 1 -compute_function -op_type elasticity -batch -gpu -gpu_batches 2', 'requires': ['cuda']},
                                                               # 2D Laplacian OpenCL 32-35
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0 -compute_function -petscspace_order 1 -petscfe_type basic -dm_plex_print_fem 1',
                                                                'requires': ['opencl']},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0 -compute_function -petscspace_order 1 -petscfe_type opencl -dm_plex_print_fem 1 -dm_plex_print_tol 1.0e-06',
                                                                'requires': ['opencl']},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0 -compute_function -petscspace_order 1 -petscfe_type opencl -petscfe_num_blocks 2 -dm_plex_print_fem 1 -dm_plex_print_tol 1.0e-06',
                                                                'requires': ['opencl']},
                                                               {'numProcs': 1, 'args': '-dm_view -refinement_limit 0.0 -compute_function -petscspace_order 1 -petscfe_type opencl -petscfe_num_blocks 2 -petscfe_num_batches 2 -dm_plex_print_fem 1 -dm_plex_print_tol 1.0e-06',
                                                                'requires': ['opencl']},
                                                               # 2D Laplacian Parallel Refinement 36-37
                                                               {'numProcs': 2, 'args': '-dm_view -interpolate -refinement_limit 0.0625 -petscspace_order 1 -refinement_uniform -compute_function -batch -gpu -gpu_batches 2', 'requires': ['opencl']},
                                                               {'numProcs': 2, 'args': '-dm_view -interpolate -refinement_limit 0.0625 -petscspace_order 1 -refinement_uniform -refinement_rounds 3 -compute_function -batch -gpu -gpu_batches 2', 'requires': ['opencl']},
                                                               ],
                        'src/snes/examples/tutorials/ex62':   [# 2D serial P1 tests 0-3
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0    -bc_type dirichlet -interpolate 0 -vel_petscspace_order 1 -pres_petscspace_order 1 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0    -bc_type dirichlet -interpolate 1 -vel_petscspace_order 1 -pres_petscspace_order 1 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0625 -bc_type dirichlet -interpolate 0 -vel_petscspace_order 1 -pres_petscspace_order 1 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 1 -pres_petscspace_order 1 -show_initial -dm_plex_print_fem 1'},
                                                               # 2D serial P2 tests 4-5
                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0    -bc_type dirichlet -interpolate 1 -vel_petscspace_order 2 -pres_petscspace_order 1 -show_initial -dm_plex_print_fem 1'},

                                                               {'numProcs': 1, 'args': '-run_type test -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 2 -pres_petscspace_order 1 -show_initial -dm_plex_print_fem 1'},
                                                               # Parallel tests 6-17
                                                               {'numProcs': 2, 'args': '-run_type test -refinement_limit 0.0    -test_partition -bc_type dirichlet -interpolate 0 -vel_petscspace_order 1 -pres_petscspace_order 1 -dm_plex_print_fem 1'},
                                                               {'numProcs': 3, 'args': '-run_type test -refinement_limit 0.0    -test_partition -bc_type dirichlet -interpolate 0 -vel_petscspace_order 1 -pres_petscspace_order 1 -dm_plex_print_fem 1'},
                                                               {'numProcs': 5, 'args': '-run_type test -refinement_limit 0.0    -test_partition -bc_type dirichlet -interpolate 0 -vel_petscspace_order 1 -pres_petscspace_order 1 -dm_plex_print_fem 1'},
                                                               {'numProcs': 2, 'args': '-run_type test -refinement_limit 0.0    -test_partition -bc_type dirichlet -interpolate 1 -vel_petscspace_order 1 -pres_petscspace_order 1 -dm_plex_print_fem 1'},
                                                               {'numProcs': 3, 'args': '-run_type test -refinement_limit 0.0    -test_partition -bc_type dirichlet -interpolate 1 -vel_petscspace_order 1 -pres_petscspace_order 1 -dm_plex_print_fem 1'},
                                                               {'numProcs': 5, 'args': '-run_type test -refinement_limit 0.0    -test_partition -bc_type dirichlet -interpolate 1 -vel_petscspace_order 1 -pres_petscspace_order 1 -dm_plex_print_fem 1'},
                                                               {'numProcs': 2, 'args': '-run_type test -refinement_limit 0.0625 -test_partition -bc_type dirichlet -interpolate 0 -vel_petscspace_order 1 -pres_petscspace_order 1 -dm_plex_print_fem 1'},
                                                               {'numProcs': 3, 'args': '-run_type test -refinement_limit 0.0625 -test_partition -bc_type dirichlet -interpolate 0 -vel_petscspace_order 1 -pres_petscspace_order 1 -dm_plex_print_fem 1'},
                                                               {'numProcs': 5, 'args': '-run_type test -refinement_limit 0.0625 -test_partition -bc_type dirichlet -interpolate 0 -vel_petscspace_order 1 -pres_petscspace_order 1 -dm_plex_print_fem 1'},
                                                               {'numProcs': 2, 'args': '-run_type test -refinement_limit 0.0625 -test_partition -bc_type dirichlet -interpolate 1 -vel_petscspace_order 1 -pres_petscspace_order 1 -dm_plex_print_fem 1'},
                                                               {'numProcs': 3, 'args': '-run_type test -refinement_limit 0.0625 -test_partition -bc_type dirichlet -interpolate 1 -vel_petscspace_order 1 -pres_petscspace_order 1 -dm_plex_print_fem 1'},
                                                               {'numProcs': 5, 'args': '-run_type test -refinement_limit 0.0625 -test_partition -bc_type dirichlet -interpolate 1 -vel_petscspace_order 1 -pres_petscspace_order 1 -dm_plex_print_fem 1'},
                                                               # Full solutions 18-29
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -interpolate 0 -vel_petscspace_order 1 -pres_petscspace_order 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view', 'parser': 'Solver'},
                                                               {'numProcs': 2, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -interpolate 0 -vel_petscspace_order 1 -pres_petscspace_order 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view', 'parser': 'Solver'},
                                                               {'numProcs': 3, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -interpolate 0 -vel_petscspace_order 1 -pres_petscspace_order 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_monitor_short -snes_converged_reason -snes_view', 'parser': 'Solver'},
                                                               {'numProcs': 5, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -interpolate 0 -vel_petscspace_order 1 -pres_petscspace_order 1 -pc_type jacobi -ksp_rtol 1.0e-10 -snes_monitor_short -snes_converged_reason -snes_view', 'parser': 'Solver'},
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 1 -pres_petscspace_order 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view', 'parser': 'Solver'},
                                                               {'numProcs': 2, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 1 -pres_petscspace_order 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view', 'parser': 'Solver'},
                                                               {'numProcs': 3, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 1 -pres_petscspace_order 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_monitor_short -snes_converged_reason -snes_view', 'parser': 'Solver'},
                                                               {'numProcs': 5, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 1 -pres_petscspace_order 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_monitor_short -snes_converged_reason -snes_view', 'parser': 'Solver'},
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 2 -pres_petscspace_order 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view', 'parser': 'Solver'},
                                                               {'numProcs': 2, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 2 -pres_petscspace_order 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view', 'parser': 'Solver'},
                                                               {'numProcs': 3, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 2 -pres_petscspace_order 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view', 'parser': 'Solver'},
                                                               {'numProcs': 5, 'args': '-run_type full -refinement_limit 0.0625 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 2 -pres_petscspace_order 1 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view', 'parser': 'Solver'},
                                                               # Stokes preconditioners 30-36
                                                               #   Jacobi
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 2 -pres_petscspace_order 1 -ksp_gmres_restart 100 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view -show_solution 0', 'parser': 'Solver'},
                                                               #  Block diagonal \begin{pmatrix} A & 0 \\ 0 & I \end{pmatrix}
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 2 -pres_petscspace_order 1 -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type additive -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view -show_solution 0', 'parser': 'Solver'},
                                                               #  Block triangular \begin{pmatrix} A & B \\ 0 & I \end{pmatrix}
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 2 -pres_petscspace_order 1 -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type multiplicative -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view -show_solution 0', 'parser': 'Solver'},
                                                               #  Diagonal Schur complement \begin{pmatrix} A & 0 \\ 0 & S \end{pmatrix}
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 2 -pres_petscspace_order 1 -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type diag -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view -show_solution 0', 'parser': 'Solver'},
                                                               #  Upper triangular Schur complement \begin{pmatrix} A & B \\ 0 & S \end{pmatrix}
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 2 -pres_petscspace_order 1 -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type upper -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view -show_solution 0', 'parser': 'Solver'},
                                                               #  Lower triangular Schur complement \begin{pmatrix} A & B \\ 0 & S \end{pmatrix}
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 2 -pres_petscspace_order 1 -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type lower -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view -show_solution 0', 'parser': 'Solver'},
                                                               #  Full Schur complement \begin{pmatrix} I & 0 \\ B^T A^{-1} & I \end{pmatrix} \begin{pmatrix} A & 0 \\ 0 & S \end{pmatrix} \begin{pmatrix} I & A^{-1} B \\ 0 & I \end{pmatrix}
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 2 -pres_petscspace_order 1 -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view -show_solution 0', 'parser': 'Solver'},
                                                               #  SIMPLE \begin{pmatrix} I & 0 \\ B^T A^{-1} & I \end{pmatrix} \begin{pmatrix} A & 0 \\ 0 & B^T diag(A)^{-1} B \end{pmatrix} \begin{pmatrix} I & diag(A)^{-1} B \\ 0 & I \end{pmatrix}
                                                               #{'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 2 -pres_petscspace_order 1 -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -fieldsplit_pressure_inner_ksp_type preonly -fieldsplit_pressure_inner_pc_type jacobi -fieldsplit_pressure_upper_ksp_type preonly -fieldsplit_pressure_upper_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view -show_solution 0', 'parser': 'Solver'},
                                                               #  SIMPLEC \begin{pmatrix} I & 0 \\ B^T A^{-1} & I \end{pmatrix} \begin{pmatrix} A & 0 \\ 0 & B^T rowsum(A)^{-1} B \end{pmatrix} \begin{pmatrix} I & rowsum(A)^{-1} B \\ 0 & I \end{pmatrix}
                                                               #{'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 2 -pres_petscspace_order 1 -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -fieldsplit_pressure_inner_ksp_type preonly -fieldsplit_pressure_inner_pc_type jacobi -fieldsplit_pressure_inner_pc_jacobi_type rowsum -fieldsplit_pressure_upper_ksp_type preonly -fieldsplit_pressure_upper_pc_type jacobi -fieldsplit_pressure_upper_pc_jacobi_type rowsum -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view -show_solution 0', 'parser': 'Solver'},
                                                               # Stokes preconditioners with MF Jacobian action 37-42
                                                               #   Jacobi
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 2 -pres_petscspace_order 1 -jacobian_mf -ksp_gmres_restart 100 -pc_type jacobi -ksp_rtol 1.0e-9 -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view -show_solution 0', 'parser': 'Solver', 'requires': 'Broken'},
                                                               #  Block diagonal \begin{pmatrix} A & 0 \\ 0 & I \end{pmatrix}
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 2 -pres_petscspace_order 1 -jacobian_mf -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type additive -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view -show_solution 0', 'parser': 'Solver', 'requires': 'Broken'},
                                                               #  Block triangular \begin{pmatrix} A & B \\ 0 & I \end{pmatrix}
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 2 -pres_petscspace_order 1 -jacobian_mf -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type multiplicative -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view -show_solution 0', 'parser': 'Solver', 'requires': 'Broken'},
                                                               #  Diagonal Schur complement \begin{pmatrix} A & 0 \\ 0 & S \end{pmatrix}
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 2 -pres_petscspace_order 1 -jacobian_mf -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type diag -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view -show_solution 0', 'parser': 'Solver', 'requires': 'Broken'},
                                                               #  Upper triangular Schur complement \begin{pmatrix} A & B \\ 0 & S \end{pmatrix}
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 2 -pres_petscspace_order 1 -jacobian_mf -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type upper -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view -show_solution 0', 'parser': 'Solver', 'requires': 'Broken'},
                                                               #  Lower triangular Schur complement \begin{pmatrix} A & B \\ 0 & S \end{pmatrix}
                                                               #{'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 2 -pres_petscspace_order 1 -jacobian_mf -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type lower -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view -show_solution 0', 'parser': 'Solver', 'requires': 'Broken'},
                                                               #  Full Schur complement \begin{pmatrix} A & B \\ B^T & S \end{pmatrix}
                                                               {'numProcs': 1, 'args': '-run_type full -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 2 -pres_petscspace_order 1 -jacobian_mf -ksp_type fgmres -ksp_gmres_restart 100 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view -show_solution 0', 'parser': 'Solver', 'requires': 'Broken'},
                                                               # 3D serial P1 tests 43-46
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -refinement_limit 0.0    -bc_type dirichlet -interpolate 0 -vel_petscspace_order 1 -pres_petscspace_order 1 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -refinement_limit 0.0    -bc_type dirichlet -interpolate 1 -vel_petscspace_order 1 -pres_petscspace_order 1 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -refinement_limit 0.0125 -bc_type dirichlet -interpolate 0 -vel_petscspace_order 1 -pres_petscspace_order 1 -show_initial -dm_plex_print_fem 1'},
                                                               {'numProcs': 1, 'args': '-run_type test -dim 3 -refinement_limit 0.0125 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 1 -pres_petscspace_order 1 -show_initial -dm_plex_print_fem 1'},
                                                               # Full runs with quads
                                                               #   FULL Schur with LU/Jacobi
                                                               {'num': 'quad_q2q1_full', 'numProcs': 1, 'args': '-run_type full -simplex 0 -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 2 -vel_petscspace_poly_tensor -pres_petscspace_order 1 -pres_petscspace_poly_tensor -ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view -show_solution 0', 'parser': 'Solver'},
                                                               {'num': 'quad_q2p1_full', 'numProcs': 1, 'args': '-run_type full -simplex 0 -refinement_limit 0.00625 -bc_type dirichlet -interpolate 1 -vel_petscspace_order 2 -vel_petscspace_poly_tensor -pres_petscspace_order 1 -pres_petscdualspace_lagrange_continuity 0 -ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_velocity_ksp_type gmres -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view -show_solution 0', 'parser': 'Solver'},
                                                               ],
                        'src/snes/examples/tutorials/ex69':   [# 2D serial P2/P1 tests 0-1
                                                               {'numProcs': 1, 'args': '-dm_plex_separate_marker -vel_petscspace_order 2 -pres_petscspace_order 1 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition full -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type svd -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view -dm_view -dmsnes_check -show_solution', 'parser': 'Solver'},
                                                               {'numProcs': 1, 'args': '-dm_plex_separate_marker -dm_refine 1 -vel_petscspace_order 2 -pres_petscspace_order 1 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition full -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_pc_type svd -snes_monitor_short -ksp_monitor_short -snes_converged_reason -ksp_converged_reason -snes_view -dm_view -dmsnes_check -show_solution', 'parser': 'Solver'},
                                                               # 2D serial discretization tests
                                                               {'num': 'p2p1', 'numProcs': 1, 'args': '-dm_plex_separate_marker -vel_petscspace_order 2 -pres_petscspace_order 1 -ksp_rtol 1e-12 -ksp_atol 1e-12 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition selfp -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-12  -fieldsplit_pressure_ksp_atol 5e-9 -fieldsplit_pressure_ksp_gmres_restart 200 -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -snes_converged_reason -snes_view -ksp_monitor_short -ksp_converged_reason -dm_view', 'parser': 'Solver'},
                                                               {'num': 'p2p1ref', 'numProcs': 1, 'args': '-dm_plex_separate_marker -dm_refine 2 -vel_petscspace_order 2 -pres_petscspace_order 1 -ksp_rtol 1e-12 -ksp_atol 1e-12 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition selfp -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-12  -fieldsplit_pressure_ksp_atol 5e-9 -fieldsplit_pressure_ksp_gmres_restart 200 -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -snes_converged_reason -snes_view -ksp_monitor_short -ksp_converged_reason -dm_view', 'parser': 'Solver'},
                                                               {'num': 'q2q1', 'numProcs': 1, 'args': '-dm_plex_separate_marker -simplex 0 -vel_petscspace_order 2 -vel_petscspace_poly_tensor -pres_petscspace_order 1 -pres_petscspace_poly_tensor -ksp_rtol 1e-12 -ksp_atol 1e-12 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition selfp -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-12  -fieldsplit_pressure_ksp_atol 5e-9 -fieldsplit_pressure_ksp_gmres_restart 200 -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -snes_converged_reason -snes_view -ksp_monitor_short -ksp_converged_reason -dm_view', 'parser': 'Solver'},
                                                               {'num': 'q2q1ref', 'numProcs': 1, 'args': '-dm_plex_separate_marker -simplex 0 -dm_refine 2 -vel_petscspace_order 2 -vel_petscspace_poly_tensor -pres_petscspace_order 1 -pres_petscspace_poly_tensor -ksp_rtol 1e-12 -ksp_atol 1e-12 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition selfp -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-12  -fieldsplit_pressure_ksp_atol 5e-9 -fieldsplit_pressure_ksp_gmres_restart 200 -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -snes_converged_reason -snes_view -ksp_monitor_short -ksp_converged_reason -dm_view', 'parser': 'Solver'},
                                                               {'num': 'q1p0', 'numProcs': 1, 'args': '-dm_plex_separate_marker -simplex 0 -vel_petscspace_order 1 -vel_petscspace_poly_tensor -pres_petscspace_order 0 -pres_petscspace_poly_tensor -ksp_rtol 1e-12 -ksp_atol 1e-12 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition selfp -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-12  -fieldsplit_pressure_ksp_atol 5e-9 -fieldsplit_pressure_ksp_gmres_restart 200 -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -snes_converged_reason -snes_view -ksp_monitor_short -ksp_converged_reason -dm_view', 'parser': 'Solver'},
                                                               {'num': 'q1p0ref', 'numProcs': 1, 'args': '-dm_plex_separate_marker -simplex 0 -dm_refine 2 -vel_petscspace_order 1 -vel_petscspace_poly_tensor -pres_petscspace_order 0 -pres_petscspace_poly_tensor -ksp_rtol 1e-12 -ksp_atol 1e-12 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition selfp -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-12  -fieldsplit_pressure_ksp_atol 5e-9 -fieldsplit_pressure_ksp_gmres_restart 200 -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -snes_converged_reason -snes_view -ksp_monitor_short -ksp_converged_reason -dm_view', 'parser': 'Solver'},
                                                               {'num': 'q2p1', 'numProcs': 1, 'args': '-dm_plex_separate_marker -simplex 0 -vel_petscspace_order 2 -vel_petscspace_poly_tensor -pres_petscspace_order 1 -pres_petscdualspace_lagrange_continuity -ksp_rtol 1e-12 -ksp_atol 1e-12 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition selfp -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-12  -fieldsplit_pressure_ksp_atol 5e-9 -fieldsplit_pressure_ksp_gmres_restart 200 -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -snes_converged_reason -snes_view -ksp_monitor_short -ksp_converged_reason -dm_view', 'parser': 'Solver'},
                                                               {'num': 'q2p1ref', 'numProcs': 1, 'args': '-dm_plex_separate_marker -simplex 0 -dm_refine 2 -vel_petscspace_order 2 -vel_petscspace_poly_tensor -pres_petscspace_order 1 -pres_petscdualspace_lagrange_continuity -ksp_rtol 1e-12 -ksp_atol 1e-12 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition selfp -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-12  -fieldsplit_pressure_ksp_atol 5e-9 -fieldsplit_pressure_ksp_gmres_restart 200 -fieldsplit_pressure_pc_type jacobi -snes_monitor_short -snes_converged_reason -snes_view -ksp_monitor_short -ksp_converged_reason -dm_view', 'parser': 'Solver'}
                                                               ],
                        'src/snes/examples/tutorials/ex75':   [# 2D serial P2/P1 tests 0-1
                                                               {'numProcs': 1, 'args': ''},
                                                               {'numProcs': 1, 'args': '-fem'}],
                        'src/snes/examples/tutorials/ex77':   [# Test from Sander
                                                               {'numProcs': 1, 'args': '-run_type full -dim 3 -dm_refine 3 -interpolate 1 -def_petscspace_order 2 -pres_petscspace_order 1 -elastMat_petscspace_order 0 -bd_def_petscspace_order 2 -bd_pres_petscspace_order 1 -snes_rtol 1e-05 -ksp_type fgmres -ksp_rtol 1e-10 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type upper -fieldsplit_deformation_ksp_type preonly -fieldsplit_deformation_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi -snes_monitor -ksp_monitor -snes_converged_reason -ksp_converged_reason -show_solution 0', 'parser': 'Solver'}],
                        'src/ts/examples/tutorials/ex11':      [# 2D Advection 0-10
                                                                {'numProcs': 1, 'args': '-ufv_vtk_interval 0 -f %(meshes)s/sevenside.exo','requires': ['exodusii']},
                                                                {'numProcs': 1, 'args': '-ufv_vtk_interval 0 -f %(meshes)s/sevenside-quad-15.exo','requires': ['exodusii']},
                                                                {'numProcs': 2, 'args': '-ufv_vtk_interval 0 -f %(meshes)s/sevenside.exo','requires': ['exodusii']},
                                                                {'numProcs': 2, 'args': '-ufv_vtk_interval 0 -f %(meshes)s/sevenside-quad-15.exo','requires': ['exodusii']},
                                                                {'numProcs': 8, 'args': '-ufv_vtk_interval 0 -f %(meshes)s/sevenside-quad.exo','requires': ['exodusii']},
                                                                {'numProcs': 1, 'args': '-ufv_vtk_interval 0 -f %(meshes)s/sevenside.exo -ts_type rosw','requires': ['exodusii']},
                                                                {'numProcs': 1, 'args': '-ufv_vtk_interval 0 -f %(meshes)s/squaremotor-30.exo -ufv_split_faces','requires': ['exodusii']},
                                                                {'numProcs': 1, 'args': '-ufv_vtk_interval 0 -f %(meshes)s/sevenside-quad-15.exo -dm_refine 1','requires': ['exodusii']},
                                                                {'numProcs': 2, 'args': '-ufv_vtk_interval 0 -f %(meshes)s/sevenside-quad-15.exo -dm_refine 2','requires': ['exodusii']},
                                                                {'numProcs': 8, 'args': '-ufv_vtk_interval 0 -f %(meshes)s/sevenside-quad-15.exo -dm_refine 2','requires': ['exodusii']},
                                                                {'numProcs': 1, 'args': '-ufv_vtk_interval 0 -f %(meshes)s/sevenside-quad.exo','requires': ['exodusii']},
                                                                # 2D Shallow water 11
                                                                {'num': 'sw_0', 'numProcs': 1, 'args': ' -ufv_vtk_interval 0 -f %(meshes)s/annulus-20.exo -bc_wall 100,101 -physics sw -ufv_cfl 5 -petscfv_type leastsquares -petsclimiter_type sin -ts_final_time 1 -ts_ssp_type rks2 -ts_ssp_nstages 10 -monitor height,energy','requires': ['exodusii']},
                                                                # 3D Advection 12
                                                                {'num': 'adv_0', 'numProcs': 1, 'args': '-ufv_vtk_interval 0 -f %(meshes)s/blockcylinder-50.exo -bc_inflow 100,101,200 -bc_outflow 201', 'requires': 'Broken'},
                                                                # 2D p4est advection
                                                                {'num': 'p4est_advec_2d', 'numProcs': 1,'args': '-ufv_vtk_interval 0 -f -dm_plex_convert_type p4est -dm_forest_minimum_refinement 1 -dm_forest_initial_refinement 2 -dm_p4est_refine_pattern hash -dm_forest_maximum_refinement 5','requires':['p4est']},
                                                                ],
                        'src/ts/examples/tutorials/ex18':      [# 2D harmonic velocity, no porosity 0-7
                                                                {'num': 'p1p1', 'numProcs': 1, 'args': '-x_bd_type none -y_bd_type none -velocity_petscspace_order 1 -velocity_petscspace_poly_tensor -porosity_petscspace_order 1 -porosity_petscspace_poly_tensor -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -pc_factor_shift_type nonzero -ts_monitor -snes_monitor_short -ksp_monitor_short -dmts_check'},
                                                                {'num': 'p2p1', 'numProcs': 1, 'args': '-x_bd_type none -y_bd_type none -velocity_petscspace_order 2 -velocity_petscspace_poly_tensor -porosity_petscspace_order 1 -porosity_petscspace_poly_tensor -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -ts_monitor -snes_monitor_short -ksp_monitor_short -dmts_check'},
                                                                {'num': 'p1p1_xper', 'numProcs': 1, 'args': '-dm_refine 1 -y_bd_type none -velocity_petscspace_order 1 -velocity_petscspace_poly_tensor -porosity_petscspace_order 1 -porosity_petscspace_poly_tensor -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -pc_type lu -pc_factor_shift_type nonzero -ksp_rtol 1.0e-8 -ts_monitor -snes_monitor_short -ksp_monitor_short -dmts_check'},
                                                                {'num': 'p1p1_xper_ref', 'numProcs': 1, 'args': '-dm_refine 3 -y_bd_type none -velocity_petscspace_order 1 -velocity_petscspace_poly_tensor -porosity_petscspace_order 1 -porosity_petscspace_poly_tensor -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -pc_type lu -pc_factor_shift_type nonzero -ksp_rtol 1.0e-8 -ts_monitor -snes_monitor_short -ksp_monitor_short -dmts_check'},
                                                                {'num': 'p1p1_xyper', 'numProcs': 1, 'args': '-dm_refine 1 -velocity_petscspace_order 1 -velocity_petscspace_poly_tensor -porosity_petscspace_order 1 -porosity_petscspace_poly_tensor -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -pc_type lu -pc_factor_shift_type nonzero -ksp_rtol 1.0e-8 -ts_monitor -snes_monitor_short -ksp_monitor_short -dmts_check'},
                                                                {'num': 'p1p1_xyper_ref', 'numProcs': 1, 'args': '-dm_refine 3 -velocity_petscspace_order 1 -velocity_petscspace_poly_tensor -porosity_petscspace_order 1 -porosity_petscspace_poly_tensor -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -pc_type lu -pc_factor_shift_type nonzero -ksp_rtol 1.0e-8 -ts_monitor -snes_monitor_short -ksp_monitor_short -dmts_check'},
                                                                {'num': 'p2p1_xyper', 'numProcs': 1, 'args': '-dm_refine 1 -velocity_petscspace_order 2 -velocity_petscspace_poly_tensor -porosity_petscspace_order 1 -porosity_petscspace_poly_tensor -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -pc_type lu -pc_factor_shift_type nonzero -ksp_rtol 1.0e-8 -ts_monitor -snes_monitor_short -ksp_monitor_short -dmts_check'},
                                                                {'num': 'p2p1_xyper_ref', 'numProcs': 1, 'args': '-dm_refine 3 -velocity_petscspace_order 2 -velocity_petscspace_poly_tensor -porosity_petscspace_order 1 -porosity_petscspace_poly_tensor -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -pc_type lu -pc_factor_shift_type nonzero -ksp_rtol 1.0e-8 -ts_monitor -snes_monitor_short -ksp_monitor_short -dmts_check'},
                                                                #   Must check that FV BCs propagate to coarse meshes
                                                                #   Must check that FV BC ids propagate to coarse meshes
                                                                #   Must check that FE+FV BCs work at the same time
                                                                # 2D Advection, matching wind in ex11 8-11
                                                                #   NOTE implicit solves are limited by accuracy of FD Jacobian
                                                                {'num': 'adv_0',    'numProcs': 1, 'args': '-f %(meshes)s/sevenside-quad.exo -x_bd_type none -y_bd_type none -use_fv -velocity_dist zero -porosity_dist tilted -ts_type ssp -ts_final_time 2.0 -ts_max_steps 1000 -ts_dt 0.993392 -ts_view -dm_view','requires': ['exodusii']},
                                                                {'num': 'adv_0_im', 'numProcs': 1, 'args': '-f %(meshes)s/sevenside-quad.exo -x_bd_type none -y_bd_type none -use_fv -use_implicit -velocity_dist zero -porosity_dist tilted -ts_type beuler -ts_final_time 2.0 -ts_max_steps 1000 -ts_dt 0.993392 -ts_view -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -pc_type lu','requires': ['exodusii']},
                                                                {'num': 'adv_0_im_2', 'numProcs': 1, 'args': '-f %(meshes)s/sevenside-quad.exo -x_bd_type none -y_bd_type none -use_fv -use_implicit -velocity_dist constant -porosity_dist tilted -ts_type beuler -ts_final_time 2.0 -ts_max_steps 1000 -ts_dt 0.993392 -ts_view -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -pc_type lu -snes_rtol 1.0e-7','requires': ['exodusii']},
                                                                {'num': 'adv_0_im_3', 'numProcs': 1, 'args': '-f %(meshes)s/sevenside-quad.exo -x_bd_type none -y_bd_type none -use_fv -use_implicit -velocity_petscspace_order 1 -velocity_petscspace_poly_tensor -velocity_dist constant -porosity_dist tilted -ts_type beuler -ts_final_time 2.0 -ts_max_steps 1000 -ts_dt 0.993392 -ts_view -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -pc_type svd -snes_rtol 1.0e-7','requires': ['exodusii']},
                                                                {'num': 'adv_0_im_4', 'numProcs': 1, 'args': '-f %(meshes)s/sevenside-quad.exo -x_bd_type none -y_bd_type none -use_fv -use_implicit -velocity_petscspace_order 2 -velocity_petscspace_poly_tensor -velocity_dist constant -porosity_dist tilted -ts_type beuler -ts_final_time 2.0 -ts_max_steps 1000 -ts_dt 0.993392 -ts_view -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -pc_type svd -snes_rtol 1.0e-7','requires': ['exodusii']},
                                                                # 2D Advection, misc
                                                                {'num': 'adv_1', 'numProcs': 1, 'args': '-x_bd_type none -y_bd_type none -use_fv -velocity_dist zero -porosity_dist tilted -ts_type ssp -ts_final_time 2.0 -ts_max_steps 1000 -ts_dt 0.993392 -bc_inflow 1,2,4 -bc_outflow 3 -ts_view -dm_view'},
                                                                {'num': 'adv_2', 'numProcs': 1, 'args': '-x_bd_type none -y_bd_type none -use_fv -velocity_dist zero -porosity_dist tilted -ts_type beuler -ts_final_time 2.0 -ts_max_steps 1000 -ts_dt 0.993392 -bc_inflow 3,4 -bc_outflow 1,2 -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -ksp_max_it 100 -ts_view -dm_view -snes_converged_reason -ksp_converged_reason'},
                                                                {'num': 'adv_3', 'numProcs': 1, 'args': '-y_bd_type none -use_fv -velocity_dist zero -porosity_dist tilted -ts_type beuler -ts_final_time 2.0 -ts_max_steps 1000 -ts_dt 0.993392 -bc_inflow 3 -bc_outflow 1 -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -ksp_max_it 100 -ts_view -dm_view -snes_converged_reason'},
                                                                {'num': 'adv_3_ex', 'numProcs': 1, 'args': '-y_bd_type none -use_fv -velocity_dist zero -porosity_dist tilted -ts_type ssp -ts_final_time 2.0 -ts_max_steps 1000 -ts_dt 0.1 -bc_inflow 3 -bc_outflow 1 -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -ksp_max_it 100 -ts_view -dm_view -snes_converged_reason'},
                                                                {'num': 'adv_4', 'numProcs': 1, 'args': '-x_bd_type none -y_bd_type none -use_fv -velocity_dist zero -porosity_dist tilted -ts_type beuler -ts_final_time 2.0 -ts_max_steps 1000 -ts_dt 0.993392 -bc_inflow 3 -bc_outflow 1 -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -ksp_max_it 100 -ts_view -dm_view -snes_converged_reason'},
                                                                # 2D Advection, box, delta
                                                                {'num': 'adv_delta_yper_0', 'numProcs': 1, 'args': '-x_bd_type none -use_fv -velocity_dist constant -porosity_dist delta -inflow_state 0.0 -ts_type euler -ts_final_time 5.0 -ts_max_steps 20 -ts_dt 0.333333 -bc_inflow 2 -bc_outflow 4 -ts_view -dm_view -monitor Error'},
                                                                {'num': 'adv_delta_yper_1', 'numProcs': 1, 'args': '-x_bd_type none -use_fv -velocity_dist constant -porosity_dist delta -inflow_state 0.0 -ts_type euler -ts_final_time 5.0 -ts_max_steps 40 -ts_dt 0.166666 -bc_inflow 2 -bc_outflow 4 -ts_view -dm_view -monitor Error -dm_refine 1 -source_loc 0.416666,0.416666'},
                                                                {'num': 'adv_delta_yper_2', 'numProcs': 1, 'args': '-x_bd_type none -use_fv -velocity_dist constant -porosity_dist delta -inflow_state 0.0 -ts_type euler -ts_final_time 5.0 -ts_max_steps 80 -ts_dt 0.083333 -bc_inflow 2 -bc_outflow 4 -ts_view -dm_view -monitor Error -dm_refine 2 -source_loc 0.458333,0.458333'},
                                                                {'num': 'adv_delta_yper_fim_0', 'numProcs': 1, 'args': '-x_bd_type none -use_fv -use_implicit -velocity_petscspace_order 0 -velocity_petscspace_poly_tensor -velocity_dist constant -porosity_dist delta -inflow_state 0.0 -ts_type mimex -ts_final_time 5.0 -ts_max_steps 20 -ts_dt 0.333333 -bc_inflow 2 -bc_outflow 4 -ts_view -monitor Error -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -mat_coloring_greedy_symmetric 0 -snes_rtol 1.0e-7 -pc_type lu -snes_converged_reason'},
                                                                {'num': 'adv_delta_yper_fim_1', 'numProcs': 1, 'args': '-x_bd_type none -use_fv -use_implicit -velocity_petscspace_order 1 -velocity_petscspace_poly_tensor -velocity_dist constant -porosity_dist delta -inflow_state 0.0 -ts_type mimex -ts_final_time 5.0 -ts_max_steps 20 -ts_dt 0.333333 -bc_inflow 2 -bc_outflow 4 -ts_view -monitor Error -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -mat_coloring_greedy_symmetric 0 -snes_rtol 1.0e-7 -pc_type lu -snes_converged_reason -snes_linesearch_type basic'},
                                                                {'num': 'adv_delta_yper_fim_2', 'numProcs': 1, 'args': '-x_bd_type none -use_fv -use_implicit -velocity_petscspace_order 2 -velocity_petscspace_poly_tensor -velocity_dist constant -porosity_dist delta -inflow_state 0.0 -ts_type mimex -ts_final_time 5.0 -ts_max_steps 20 -ts_dt 0.333333 -bc_inflow 2 -bc_outflow 4 -ts_view -monitor Error -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -mat_coloring_greedy_symmetric 0 -snes_rtol 1.0e-7 -pc_type lu -snes_converged_reason -snes_linesearch_type basic'},
                                                                {'num': 'adv_delta_yper_im_0', 'numProcs': 1, 'args': '-x_bd_type none -use_fv -use_implicit -velocity_petscspace_order 0 -velocity_petscspace_poly_tensor -velocity_dist constant -porosity_dist delta -inflow_state 0.0 -ts_type mimex -ts_mimex_version 0 -ts_final_time 5.0 -ts_max_steps 20 -ts_dt 0.333333 -bc_inflow 2 -bc_outflow 4 -ts_view -monitor Error -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -snes_rtol 1.0e-7 -pc_type lu -snes_converged_reason'},
                                                                {'num': 'adv_delta_yper_im_1', 'numProcs': 1, 'args': '-x_bd_type none -use_fv -use_implicit -velocity_petscspace_order 0 -velocity_petscspace_poly_tensor -velocity_dist constant -porosity_dist delta -inflow_state 0.0 -ts_type mimex -ts_mimex_version 0 -ts_final_time 5.0 -ts_max_steps 40 -ts_dt 0.166666 -bc_inflow 2 -bc_outflow 4 -ts_view -monitor Error -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -snes_rtol 1.0e-7 -pc_type lu -snes_converged_reason -dm_refine 1 -source_loc 0.416666,0.416666'},
                                                                {'num': 'adv_delta_yper_im_2', 'numProcs': 1, 'args': '-x_bd_type none -use_fv -use_implicit -velocity_petscspace_order 0 -velocity_petscspace_poly_tensor -velocity_dist constant -porosity_dist delta -inflow_state 0.0 -ts_type mimex -ts_mimex_version 0 -ts_final_time 5.0 -ts_max_steps 80 -ts_dt 0.083333 -bc_inflow 2 -bc_outflow 4 -ts_view -monitor Error -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -snes_rtol 1.0e-7 -pc_type lu -snes_converged_reason -dm_refine 2 -source_loc 0.458333,0.458333'},
                                                                {'num': 'adv_delta_yper_im_3', 'numProcs': 1, 'args': '-x_bd_type none -use_fv -use_implicit -velocity_petscspace_order 1 -velocity_petscspace_poly_tensor -velocity_dist constant -porosity_dist delta -inflow_state 0.0 -ts_type mimex -ts_mimex_version 0 -ts_final_time 5.0 -ts_max_steps 20 -ts_dt 0.333333 -bc_inflow 2 -bc_outflow 4 -ts_view -monitor Error -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -snes_rtol 1.0e-7 -pc_type lu -snes_converged_reason'},
                                                                #    I believe the nullspace is sin(pi y)
                                                                {'num': 'adv_delta_yper_im_4', 'numProcs': 1, 'args': '-x_bd_type none -use_fv -use_implicit -velocity_petscspace_order 1 -velocity_petscspace_poly_tensor -velocity_dist constant -porosity_dist delta -inflow_state 0.0 -ts_type mimex -ts_mimex_version 0 -ts_final_time 5.0 -ts_max_steps 40 -ts_dt 0.166666 -bc_inflow 2 -bc_outflow 4 -ts_view -monitor Error -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -snes_rtol 1.0e-7 -pc_type lu -snes_converged_reason -dm_refine 1 -source_loc 0.416666,0.416666'},
                                                                {'num': 'adv_delta_yper_im_5', 'numProcs': 1, 'args': '-x_bd_type none -use_fv -use_implicit -velocity_petscspace_order 1 -velocity_petscspace_poly_tensor -velocity_dist constant -porosity_dist delta -inflow_state 0.0 -ts_type mimex -ts_mimex_version 0 -ts_final_time 5.0 -ts_max_steps 80 -ts_dt 0.083333 -bc_inflow 2 -bc_outflow 4 -ts_view -monitor Error -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -snes_rtol 1.0e-7 -pc_type lu -snes_converged_reason -dm_refine 2 -source_loc 0.458333,0.458333'},
                                                                {'num': 'adv_delta_yper_im_6', 'numProcs': 1, 'args': '-x_bd_type none -use_fv -use_implicit -velocity_petscspace_order 2 -velocity_petscspace_poly_tensor -velocity_dist constant -porosity_dist delta -inflow_state 0.0 -ts_type mimex -ts_final_time 5.0 -ts_max_steps 20 -ts_dt 0.333333 -bc_inflow 2 -bc_outflow 4 -ts_view -monitor Error -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -snes_rtol 1.0e-7 -pc_type svd -snes_converged_reason'},
                                                                # 2D Advection, magma benchmark 1
                                                                {'num': 'adv_delta_shear_im_0', 'numProcs': 1, 'args': '-y_bd_type none -dm_refine 2 -use_fv -use_implicit -velocity_petscspace_order 1 -velocity_petscspace_poly_tensor -velocity_dist shear -porosity_dist delta -inflow_state 0.0 -ts_type mimex -ts_final_time 5.0 -ts_max_steps 20 -ts_dt 0.333333 -bc_inflow 1,3 -ts_view -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -snes_rtol 1.0e-7 -pc_type lu -snes_converged_reason -source_loc 0.458333,0.708333'},
                                                                # 2D Advection, box, gaussian
                                                                {'num': 'adv_gauss', 'numProcs': 1, 'args': '-x_bd_type none -y_bd_type none -use_fv -velocity_dist constant -porosity_dist gaussian -inflow_state 0.0 -ts_type ssp -ts_final_time 2.0 -ts_max_steps 100 -ts_dt 0.01 -bc_inflow 1 -bc_outflow 3 -ts_view -dm_view'},
                                                                {'num': 'adv_gauss_im', 'numProcs': 1, 'args': '-x_bd_type none -y_bd_type none -use_fv -use_implicit -velocity_dist constant -porosity_dist gaussian -inflow_state 0.0 -ts_type beuler -ts_final_time 2.0 -ts_max_steps 100 -ts_dt 0.01 -bc_inflow 1 -bc_outflow 3 -ts_view -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -snes_rtol 1.0e-7'},
                                                                {'num': 'adv_gauss_im_1', 'numProcs': 1, 'args': '-x_bd_type none -y_bd_type none -use_fv -use_implicit -velocity_petscspace_order 1 -velocity_petscspace_poly_tensor -velocity_dist constant -porosity_dist gaussian -inflow_state 0.0 -ts_type beuler -ts_final_time 2.0 -ts_max_steps 100 -ts_dt 0.01 -bc_inflow 1 -bc_outflow 3 -ts_view -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -snes_rtol 1.0e-7'},
                                                                {'num': 'adv_gauss_im_2', 'numProcs': 1, 'args': '-x_bd_type none -y_bd_type none -use_fv -use_implicit -velocity_petscspace_order 2 -velocity_petscspace_poly_tensor -velocity_dist constant -porosity_dist gaussian -inflow_state 0.0 -ts_type beuler -ts_final_time 2.0 -ts_max_steps 100 -ts_dt 0.01 -bc_inflow 1 -bc_outflow 3 -ts_view -dm_view -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -snes_rtol 1.0e-7'},
                                                                {'num': 'adv_gauss_corner', 'numProcs': 1, 'args': '-x_bd_type none -y_bd_type none -use_fv -velocity_dist constant -porosity_dist gaussian -inflow_state 0.0 -ts_type ssp -ts_final_time 2.0 -ts_max_steps 100 -ts_dt 0.01 -bc_inflow 1 -bc_outflow 2 -ts_view -dm_view'},
                                                                # 2D Advection+Harmonic 12-
                                                                {'num': 'adv_harm_0', 'numProcs': 1, 'args': '-x_bd_type none -y_bd_type none -velocity_petscspace_order 2 -velocity_petscspace_poly_tensor -use_fv -velocity_dist harmonic -porosity_dist gaussian -ts_type beuler -ts_final_time 2.0 -ts_max_steps 1000 -ts_dt 0.993392 -bc_inflow 1,2,4 -bc_outflow 3 -use_implicit -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -ksp_max_it 100 -ts_view -dm_view -snes_converged_reason -ksp_converged_reason -snes_monitor -dmts_check'},
                                                                ],
                        'src/tao/examples/tutorials/ex1':      [# 2D 0-1
                                                                {'numProcs': 1, 'args': '-run_type test -dmsnes_check -potential_petscspace_order 2 -conductivity_petscspace_order 1 -multiplier_petscspace_order 2'},
                                                                {'numProcs': 1, 'args': '-potential_petscspace_order 2 -conductivity_petscspace_order 1 -multiplier_petscspace_order 2 -snes_monitor -pc_type fieldsplit -pc_fieldsplit_0_fields 0,1 -pc_fieldsplit_1_fields 2 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition selfp -fieldsplit_0_pc_type lu -fieldsplit_1_ksp_rtol 1.0e-10 -fieldsplit_1_pc_type lu -sol_vec_view'}],
                        'src/tao/examples/tutorials/ex2':      [# 2D 0-1 Dual solution
                                                                {'numProcs': 1, 'args': '-run_type test -dmsnes_check -potential_petscspace_order 2 -charge_petscspace_order 1 -multiplier_petscspace_order 1'},
                                                                {'numProcs': 1, 'args': '-potential_petscspace_order 2 -charge_petscspace_order 1 -multiplier_petscspace_order 1 -snes_monitor -snes_converged_reason -pc_type fieldsplit -pc_fieldsplit_0_fields 0,1 -pc_fieldsplit_1_fields 2 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition selfp -fieldsplit_0_pc_type lu -fieldsplit_1_ksp_rtol 1.0e-10 -fieldsplit_1_pc_type lu -sol_vec_view'},
                                                                {'numProcs': 1, 'args': '-potential_petscspace_order 2 -charge_petscspace_order 1 -multiplier_petscspace_order 1 -snes_monitor -snes_converged_reason -snes_fd -pc_type fieldsplit -pc_fieldsplit_0_fields 0,1 -pc_fieldsplit_1_fields 2 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition selfp -fieldsplit_0_pc_type lu -fieldsplit_1_ksp_rtol 1.0e-10 -fieldsplit_1_pc_type lu -sol_vec_view'}
                                                                ],
                        }

def noCheckCommand(command, status, output, error):
  ''' Do no check result'''
  return

class IdentityParser(object):
  def __init__(self):
    return

  def parse(self, text):
    return text, ''

class KSP(object):
  def __init__(self, atol = 1.0e-12, rtol = 1.0e-8):
    self.res  = []
    self.atol = atol
    self.rtol = rtol
    return

  def addResidual(self, n, res):
    if not len(self.res) == n: raise RuntimeError('Invalid KSP residual '+str(res)+' at iterate '+str(n))
    self.res.append(res)
    return

  def __eq__(self, s):
    return all([abs(a-b) < self.atol or abs((a-b)/a) < self.rtol for a, b in zip(self.res, s.res)])

  def __str__(self):
    return 'SNES:\n'+str(self.res)

class SNES(object):
  def __init__(self, atol = 1.0e-12, rtol = 1.0e-8):
    self.res = []
    self.atol = atol
    self.rtol = rtol
    return

  def addResidual(self, n, res):
    if not len(self.res) == n: raise RuntimeError('Invalid SNES residual at iterate '+str(n))
    self.res.append(res)
    return

  def __eq__(self, s):
    return all([abs(a-b) < self.atol or abs((a-b)/a) < self.rtol for a, b in zip(self.res, s.res)])

  def __str__(self):
    return 'SNES:\n'+str(self.res)

class Error(object):
  def __init__(self, atol = 1.0e-12, rtol = 1.0e-8):
    self.res  = 0.0
    self.atol = atol
    self.rtol = rtol
    return

  def setNorm(self, res):
    self.res = res
    return

  def __eq__(self, s):
    return all([abs(a-b) < self.atol or abs((a-b)/a) < self.rtol for a, b in [(self.res, s.res)]])

  def __str__(self):
    return 'L_2 Error:\n'+str(self.res)

class SolverParser(object):
  def __init__(self, atol = 1.0e-12):
    import re

    self.atol    = atol
    self.reSNES  = re.compile(r'\s*(?P<it>\d+) SNES Function norm (?P<norm>\d+\.\d+(e(\+|-)\d+)?)')
    self.reKSP   = re.compile(r'\s*(?P<it>\d+) KSP Residual norm (?P<norm>\d+\.\d+(e(\+|-)\d+)?)')
    self.reError = re.compile(r'L_2 Error: (?P<norm>\d+\.\d+(e(\+|-)\d+)?)')
    return

  def parse(self, text):
    lines  = text.split('\n')
    objs   = []
    stack  = []
    excess = []
    for line in lines:
      mSNES  = self.reSNES.match(line)
      mKSP   = self.reKSP.match(line)
      mError = self.reError.match(line)
      if mSNES:
        it = int(mSNES.group('it'))
        if it == 0: stack.append(SNES(atol = self.atol))
        stack[-1].addResidual(it, float(mSNES.group('norm')))
      elif mKSP:
        it = int(mKSP.group('it'))
        if it == 0: stack.append(KSP(atol = self.atol))
        stack[-1].addResidual(it, float(mKSP.group('norm')))
      elif mError:
        o = Error(atol = self.atol)
        o.setNorm(float(mError.group('norm')))
        objs.append(o)
      elif line.strip().startswith('Nonlinear solve converged'):
        objs.append(stack.pop())
      elif line.strip().startswith('Linear solve converged'):
        objs.append(stack.pop())
      else:
        excess.append(line)
    return objs, '\n'.join(excess)

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
    m = re.match('-@\$\{MPIEXEC\} -n (?P<numProcs>\d+) ./(?P<ex>ex\w+)(?P<args>[\'%-.,\w ]+)>', lines[0])
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
      runs   = [e for e in testTargets if e == 'run'+base or e.startswith('run'+base+'_')]
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
    self.defines.update(self.configInfo.indexType.defines)
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
        target, deps = f.read().split(':', 1)
      except ValueError as e:
        self.logPrint('ERROR in dependency file %s: %s' % (depFile, str(e)))
    target = target.split()[0]
    if (target != self.sourceManager.getObjectName(source)): print target, self.sourceManager.getObjectName(source)
    assert(target == self.sourceManager.getObjectName(source))
    deps = deps.split('\n\n')[0]
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
    self.arch            = self.framework.require('PETSc.options.arch',        None)
    self.petscdir        = self.framework.require('PETSc.options.petscdir',    None)
    self.languages       = self.framework.require('PETSc.options.languages',   None)
    self.debugging       = self.framework.require('PETSc.options.debugging',   None)
    self.debuggers       = self.framework.require('config.utilities.debuggers',   None)
    self.compilers       = self.framework.require('config.compilers',            None)
    self.types           = self.framework.require('config.types',                None)
    self.headers         = self.framework.require('config.headers',              None)
    self.functions       = self.framework.require('config.functions',            None)
    self.libraries       = self.framework.require('config.libraries',            None)
    self.scalarType      = self.framework.require('PETSc.options.scalarTypes', None)
    self.indexType       = self.framework.require('PETSc.options.indexTypes', None)
    self.memAlign        = self.framework.require('PETSc.options.memAlign',    None)
    self.libraryOptions  = self.framework.require('PETSc.options.libraryOptions', None)
    self.fortrancpp      = self.framework.require('PETSc.options.fortranCPP', None)
    self.sharedLibraries = self.framework.require('PETSc.options.sharedLibraries', None)
    self.sowing          = self.framework.require('config.packages.sowing', None)
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

 def __init__(self, logname = 'make.log'):
   import RDict
   import os

   argDB = RDict.RDict(None, None, 0, 0, readonly = True)
   self.petscDir = os.environ['PETSC_DIR']
   arch  = self.findArch()
   argDB.saveFilename = os.path.join(self.petscDir, arch, 'lib','petsc','conf', 'RDict.db')
   argDB.load()
   script.Script.__init__(self, argDB = argDB)
   self.logName = logname
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
   self.petscConfDir = os.path.join(self.petscDir, self.petscArch, 'lib','petsc','conf')
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
   flags.extend([self.configInfo.setCompilers.CPPFLAGS]) # Add CPP_FLAGS
   if self.configInfo.compilers.generateDependencies[language]:
     flags.append(self.configInfo.compilers.dependenciesGenerationFlag[language])
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
   flags.extend([self.configInfo.setCompilers.CPPFLAGS]) # Add CPP_FLAGS
   if self.configInfo.compilers.generateDependencies[language]:
     flags.append(self.configInfo.compilers.dependenciesGenerationFlag[language])
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
   if os.path.samefile(self.rootDir, self.petscDir):
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
   self.logPrint('Making shared libraries for OS %s using language %s' % (osName, self.configInfo.setCompilers.language[-1]))
   # PCC_LINKER PCC_LINKER_FLAGS
   linker      = self.configInfo.setCompilers.getSharedLinker()
   linkerFlags = self.configInfo.setCompilers.getLinkerFlags()
   packageIncludes, packageLibs = self.getPackageInfo()
   extraLibs = self.configInfo.libraries.toStringNoDupes(self.configInfo.compilers.flibs+self.configInfo.compilers.cxxlibs+self.configInfo.compilers.LIBS.split(' '))
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
       cmd += linker+' -g  -dynamiclib -single_module -multiply_defined suppress -undefined dynamic_lookup '+flags+' -o '+sharedLib+' *.o -L'+libDir+' '+packageLibs+' '+sysLib+' '+extraLibs+' -lm -lc'
     elif osName == 'cygwin':
       cmd = linker+' '+linkerFlags+' -shared -o '+sharedLib+' *.o '+externalLib
     else:
       raise RuntimeError('Do not know how to make shared library for your '+osName+' OS')
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
   if self.argDB.get('with-single-library') == 0:
       libpetsc = ' -lpetscts -lpetscsnes -lpetscksp -lpetscdm -lpetscmat -lpetscvec -lpetscsys '
   else:
       libpetsc = ' -lpetsc '
   cmd = self.configInfo.compilers.getFullLinkerCmd(' '.join(objects)+' -L'+self.petscLibDir+libpetsc+packageLibs+' -L/usr/local/cuda/lib', executable)
   if not self.dryRun:
     (output, error, status) = self.executeShellCommand(cmd, checkCommand = noCheckCommand, log=self.log)
     if status:
       self.logPrint("ERROR IN LINK ******************************", debugSection='screen')
       self.logPrint(output+error, debugSection='screen')
     # TODO: Move dsymutil stuff from PETSc.options.debuggers to config.compilers
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
   totalRebuild = os.path.samefile(rootDir, self.petscDir) and not len(self.sourceDatabase)
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

 def checkTestOutputGeneric(self, parser, testDir, executable, cmd, output, testNum):
   from difflib import unified_diff
   outputName = os.path.join(testDir, 'output', os.path.basename(executable)+'_'+testNum+'.out')
   ret        = 0
   with file(outputName) as f:
     output                  = output.strip()
     parse, excess           = parser.parse(output)
     validOutput             = f.read().strip().replace('\r', '') # Jed is now stripping output it appears
     validParse, validExcess = parser.parse(validOutput)
     if not validParse == parse or not validExcess == excess:
       self.logPrint("TEST ERROR: Regression output for %s (test %s) does not match\n" % (executable, testNum), debugSection = 'screen')
       for linum,line in enumerate(unified_diff(validOutput.split('\n'), output.split('\n'), fromfile=outputName, tofile=cmd)):
         end = '' if linum < 3 else '\n' # Control lines have their own end-lines
         self.logWrite(line+end, debugSection = 'screen', forceScroll = True)
       self.logPrint('Reference output from %s\n' % outputName)
       self.logPrint(validOutput, indent = 0)
       self.logPrint('Current output from %s' % cmd)
       self.logPrint(output, indent = 0)
       ret = -1
     else:
       self.logPrint("TEST SUCCESS: Regression output for %s (test %s) matches" % (executable, testNum))
   return ret

 def checkTestOutput(self, numProcs, testDir, executable, cmd, output, testNum):
   return self.checkTestOutputGeneric(IdentityParser(), testDir, executable, cmd, output, testNum)

 def checkTestOutputSolver(self, numProcs, testDir, executable, cmd, output, testNum):
   if numProcs > 1: parser = SolverParser(atol = 1.0e-9)
   else:            parser = SolverParser()
   return self.checkTestOutputGeneric(parser, testDir, executable, cmd, output, testNum)

 def getTestCommand(self, executable, **params):
   numProcs = params.get('numProcs', 1)
   try:
     args   = params.get('args', '') % dict(meshes=os.path.join(self.petscDir,'share','petsc','datafiles','meshes'))
   except ValueError:
     args   = params.get('args', '')
   hosts    = ','.join(['localhost']*int(numProcs))
   return ' '.join([self.configInfo.mpi.mpiexec, '-host', hosts, '-n', str(numProcs), os.path.abspath(executable), args])

 def runTest(self, testDir, executable, testNum, replace, **params):
   '''testNum can be any string'''
   num = str(testNum)
   cmd = self.getTestCommand(executable, **params)
   numProcs = params.get('numProcs', 1)
   self.logPrint('Running #%s: %s' % (num, cmd), debugSection='screen')
   if not self.dryRun:
     (output, error, status) = self.executeShellCommand(cmd, checkCommand = noCheckCommand, log=self.log)
     outputName = os.path.join(testDir, 'output', os.path.basename(executable)+'_'+num+'.out')
     if status:
       self.logPrint("TEST ERROR: Failed to execute %s\n" % executable, debugSection='screen')
       self.logPrint(output+error, indent = 0, debugSection='screen')
       ret = -2
     elif replace:
       outputName = os.path.join(testDir, 'output', os.path.basename(executable)+'_'+str(testNum)+'.out')
       with file(outputName, 'w') as f:
         f.write(output+error)
       self.logPrint("REPLACED: Regression output file %s (test %s) was stored" % (outputName, str(testNum)), debugSection='screen')
       ret = 0
     elif not os.path.isfile(outputName):
       self.logPrint("MISCONFIGURATION: Regression output file %s (test %s) is missing" % (outputName, testNum), debugSection='screen')
       ret = 0
     else:
       ret = getattr(self, 'checkTestOutput'+params.get('parser', ''))(numProcs, testDir, executable, cmd, output+error, num)
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
       self.runTest(dirname, executable, testNum, False, **regressionParameters.get(paramKey, {}))
       testNum += 1
       while '%s_%d' % (paramKey, testNum) in regressionParameters:
         self.runTest(dirname, executable, testNum, False, **regressionParameters.get('%s_%d' % (paramKey, testNum), {}))
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
   for d in os.listdir(os.path.join(self.petscDir, 'include', 'petsc','finclude', 'ftn-auto')):
     if d.endswith('-tmpdir'): shutil.rmtree(os.path.join(self.petscDir, 'include', 'petsc','finclude', 'ftn-auto', d))
   main(self.petscDir, self.configInfo.sowing.bfort, os.getcwd(),0)
   processf90interfaces(self.petscDir,0)
   for d in os.listdir(os.path.join(self.petscDir, 'include', 'petsc','finclude', 'ftn-auto')):
     if d.endswith('-tmpdir'): shutil.rmtree(os.path.join(self.petscDir, 'include', 'petsc','finclude', 'ftn-auto', d))
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
