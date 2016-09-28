
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/dm/impls/plex/examples/tests/
EXAMPLESC       = ex1.c ex3.c ex9.c ex15.c
EXAMPLESF       = ex1f90.F ex2f90.F
MANSEC          = DM

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_DM_LIB}
	${RM} -f ex1.o

ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o ${PETSC_DM_LIB}
	${RM} -f ex4.o

ex5: ex5.o  chkopts
	-${CLINKER} -o ex5 ex5.o ${PETSC_DM_LIB}
	${RM} -f ex5.o

ex6: ex6.o  chkopts
	-${CLINKER} -o ex6 ex6.o ${PETSC_DM_LIB}
	${RM} -f ex6.o

ex1f90: ex1f90.o  chkopts
	-${FLINKER} -o ex1f90 ex1f90.o  ${PETSC_DM_LIB}
	${RM} -f ex1f90.o

ex2f90: ex2f90.o  chkopts
	-${FLINKER} -o ex2f90 ex2f90.o  ${PETSC_DM_LIB}
	${RM} -f ex2f90.o

ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o ${PETSC_SNES_LIB}
	${RM} -f ex3.o

ex9: ex9.o  chkopts
	-${CLINKER} -o ex9 ex9.o ${PETSC_SNES_LIB}
	${RM} -f ex9.o

ex14: ex14.o chkopts
	-${CLINKER} -o ex14 ex14.o ${PETSC_DM_LIB}
	${RM} -f ex14.o

ex15: ex15.o  chkopts
	-${CLINKER} -o ex15 ex15.o ${PETSC_DM_LIB}
	${RM} -f ex15.o

ex19: ex19.o  chkopts
	-${CLINKER} -o ex19 ex19.o ${PETSC_DM_LIB}
	${RM} -f ex19.o

ex20: ex20.o  chkopts
	-${CLINKER} -o ex20 ex20.o ${PETSC_DM_LIB}
	${RM} -f ex20.o


#--------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1 -dim 3 -ctetgen_verbose 4 -dm_view ascii::ascii_info_detail -info -info_exclude null > ex1_0.tmp 2>&1;\
	   if (${DIFF} output/ex1_0.out ex1_0.tmp) then true ;  \
	   else printf "${PWD}\nPossible problem with runex1, diffs above\n=========================================\n"; fi ;\
	   ${RM} -f ex1_0.tmp
runex1_2:
	-@${MPIEXEC} -n 1 ./ex1 -dim 3 -ctetgen_verbose 4 -refinement_limit 0.0625 -dm_view ascii::ascii_info_detail -info -info_exclude null > ex1_1.tmp 2>&1;\
	   if (${DIFF} output/ex1_1.out ex1_1.tmp) then true ;  \
	   else printf "${PWD}\nPossible problem with runex1_2, diffs above\n=========================================\n"; fi ;\
	   ${RM} -f ex1_1.tmp

runex1_test_shape:
	-@${MPIEXEC} -n 1 ./ex1 -dim 3 -interpolate -dm_refine_hierarchy 3 -test_shape > ex1_test_shape.tmp 2>&1;\
	   if (${DIFF} output/ex1_test_shape.out ex1_test_shape.tmp) then true ;  \
	   else printf "${PWD}\nPossible problem with runex1_test_shape, diffs above\n=================================\n"; fi ;\
	   ${RM} -f ex1_test_shape.tmp


runex1f90:
	-@${MPIEXEC} -n 1 ./ex1f90 > ex1f90_0.tmp 2>&1;\
	   if (${DIFF} output/ex1f90_0.out ex1f90_0.tmp) then true; \
	   else printf "${PWD}\nPossible problem with runex1f90, diffs above\n=========================================\n"; fi;\
	   ${RM} -f ex1f90_0.tmp

runex2f90:
	-@${MPIEXEC} -n 1 ./ex2f90 |grep -v DM_ > ex2f90_0.tmp 2>&1;\
	   if (${DIFF} output/ex2f90_0.out ex2f90_0.tmp) then true; \
	   else printf "${PWD}\nPossible problem with runex2f90, diffs above\n=========================================\n"; fi;\
	   ${RM} -f ex2f90_0.tmp

runex3:
	-@${MPIEXEC} -n 1 ./ex3 > ex3_1.tmp 2>&1;\
	   if (${DIFF} output/ex3_1.out ex3_1.tmp) then true ;  \
	   else printf "${PWD}\nPossible problem with runex3, diffs above\n=========================================\n"; fi ;\
	   ${RM} -f ex3_1.tmp

runex3_constraints:
	-@${MPIEXEC} -n 1 ./ex3 -simplex 0 -num_comp 2 -petscspace_poly_tensor -petscspace_order 1 -qorder 1 -constraints > ex3_constraints.tmp 2>&1;\
	   if (${DIFF} output/ex3_constraints.out ex3_constraints.tmp) then true ;  \
	   else printf "${PWD}\nPossible problem with runex3_constraints, diffs above\n=========================================\n"; fi ;\
	   ${RM} -f ex3_constraints.tmp
runex3_nonconforming_simplex_2:
	-@${MPIEXEC} -n 4 ./ex3 -test_fe_jacobian -test_injector -petscpartitioner_type simple -tree -simplex 1 -dim 2 -num_comp 2 -dm_plex_max_projection_height 1 -petscspace_order 2 -qorder 2 -dm_view ascii::ASCII_INFO_DETAIL > ex3_nonconforming_simplex_2.tmp 2>&1;\
	   if (${DIFF} output/ex3_nonconforming_simplex_2.out ex3_nonconforming_simplex_2.tmp) then true ;\
	   else printf "${PWD}\nPossible problem with runex3_nonconforming_simplex_2, diffs above\n=========================================\n"; fi ;\
	   ${RM} -f ex3_nonconforming_simplex_2.tmp
runex3_nonconforming_simplex_2_hi:
	-@${MPIEXEC} -n 4 ./ex3 -test_fe_jacobian -petscpartitioner_type simple -tree -simplex 1 -dim 2 -num_comp 2 -dm_plex_max_projection_height 1 -petscspace_order 4 -qorder 4 > ex3_nonconforming_simplex_2_hi.tmp 2>&1;\
	   if (${DIFF} output/ex3_nonconforming_simplex_2_hi.out ex3_nonconforming_simplex_2_hi.tmp) then true ;\
	   else printf "${PWD}\nPossible problem with runex3_nonconforming_simplex_2_hi, diffs above\n=========================================\n"; fi ;\
	   ${RM} -f ex3_nonconforming_simplex_2_hi.tmp
runex3_nonconforming_simplex_3:
	-@${MPIEXEC} -n 4 ./ex3 -test_fe_jacobian -test_injector -petscpartitioner_type simple -tree -simplex 1 -dim 3 -num_comp 3 -dm_plex_max_projection_height 2 -petscspace_order 2 -qorder 2 -dm_view ascii::ASCII_INFO_DETAIL > ex3_nonconforming_simplex_3.tmp 2>&1;\
	   if (${DIFF} output/ex3_nonconforming_simplex_3.out ex3_nonconforming_simplex_3.tmp) then true ;\
	   else printf "${PWD}\nPossible problem with runex3_nonconforming_simplex_3, diffs above\n=========================================\n"; fi ;\
	   ${RM} -f ex3_nonconforming_simplex_3.tmp
runex3_nonconforming_simplex_3_hi:
	-@${MPIEXEC} -n 4 ./ex3 -test_fe_jacobian -petscpartitioner_type simple -tree -simplex 1 -dim 3 -num_comp 3 -dm_plex_max_projection_height 2 -petscspace_order 4 -qorder 4 > ex3_nonconforming_simplex_3_hi.tmp 2>&1;\
	   if (${DIFF} output/ex3_nonconforming_simplex_3_hi.out ex3_nonconforming_simplex_3_hi.tmp) then true ;\
	   else printf "${PWD}\nPossible problem with runex3_nonconforming_simplex_3_hi, diffs above\n=========================================\n"; fi ;\
	   ${RM} -f ex3_nonconforming_simplex_3_hi.tmp
runex3_nonconforming_tensor_2:
	-@${MPIEXEC} -n 4 ./ex3 -test_fe_jacobian -test_injector -petscpartitioner_type simple -tree -simplex 0 -dim 2 -num_comp 2 -dm_plex_max_projection_height 1 -petscspace_poly_tensor -petscspace_order 2 -qorder 2 -dm_view ascii::ASCII_INFO_DETAIL > ex3_nonconforming_tensor_2.tmp 2>&1;\
	   if (${DIFF} output/ex3_nonconforming_tensor_2.out ex3_nonconforming_tensor_2.tmp) then true ;\
	   else printf "${PWD}\nPossible problem with runex3_nonconforming_tensor_2, diffs above\n=========================================\n"; fi ;\
	   ${RM} -f ex3_nonconforming_tensor_2.tmp
runex3_nonconforming_tensor_2_hi:
	-@${MPIEXEC} -n 4 ./ex3 -test_fe_jacobian -petscpartitioner_type simple -tree -simplex 0 -dim 2 -num_comp 2 -dm_plex_max_projection_height 1 -petscspace_poly_tensor -petscspace_order 4 -qorder 4 > ex3_nonconforming_tensor_2_hi.tmp 2>&1;\
	   if (${DIFF} output/ex3_nonconforming_tensor_2_hi.out ex3_nonconforming_tensor_2_hi.tmp) then true ;\
	   else printf "${PWD}\nPossible problem with runex3_nonconforming_tensor_2_hi, diffs above\n=========================================\n"; fi ;\
	   ${RM} -f ex3_nonconforming_tensor_2_hi.tmp
runex3_nonconforming_tensor_3:
	-@${MPIEXEC} -n 4 ./ex3 -test_fe_jacobian -petscpartitioner_type simple -tree -simplex 0 -dim 3 -num_comp 3 -dm_plex_max_projection_height 2 -petscspace_poly_tensor -petscspace_order 1 -qorder 1 -dm_view ascii::ASCII_INFO_DETAIL > ex3_nonconforming_tensor_3.tmp 2>&1;\
	   if (${DIFF} output/ex3_nonconforming_tensor_3.out ex3_nonconforming_tensor_3.tmp) then true ;\
	   else printf "${PWD}\nPossible problem with runex3_nonconforming_tensor_3, diffs above\n=========================================\n"; fi ;\
	   ${RM} -f ex3_nonconforming_tensor_3.tmp
runex3_nonconforming_tensor_3_hi:
	-@${MPIEXEC} -n 4 ./ex3 -test_fe_jacobian -petscpartitioner_type simple -tree -simplex 0 -dim 3 -num_comp 3 -dm_plex_max_projection_height 2 -petscspace_poly_tensor -petscspace_order 4 -qorder 4 > ex3_nonconforming_tensor_3_hi.tmp 2>&1;\
	   if (${DIFF} output/ex3_nonconforming_tensor_3_hi.out ex3_nonconforming_tensor_3_hi.tmp) then true ;\
	   else printf "${PWD}\nPossible problem with runex3_nonconforming_tensor_3_hi, diffs above\n=========================================\n"; fi ;\
	   ${RM} -f ex3_nonconforming_tensor_3_hi.tmp
runex3_nonconforming_simplex_2_fv:
	-@${MPIEXEC} -n 4 ./ex3 -test_fv_grad -test_injector -petsclimiter_type none -petscpartitioner_type simple -tree -simplex 1 -dim 2 -num_comp 2 > ex3_nonconforming_simplex_2_fv.tmp 2>&1;\
	   if (${DIFF} output/ex3_nonconforming_simplex_2_fv.out ex3_nonconforming_simplex_2_fv.tmp) then true ;\
	   else printf "${PWD}\nPossible problem with runex3_nonconforming_simplex_2_fv, diffs above\n=========================================\n"; fi ;\
	   ${RM} -f ex3_nonconforming_simplex_2_fv.tmp
runex3_nonconforming_simplex_3_fv:
	-@${MPIEXEC} -n 4 ./ex3 -test_fv_grad -test_injector -petsclimiter_type none -petscpartitioner_type simple -tree -simplex 1 -dim 3 -num_comp 3 > ex3_nonconforming_simplex_3_fv.tmp 2>&1;\
	   if (${DIFF} output/ex3_nonconforming_simplex_3_fv.out ex3_nonconforming_simplex_3_fv.tmp) then true ;\
	   else printf "${PWD}\nPossible problem with runex3_nonconforming_simplex_3_fv, diffs above\n=========================================\n"; fi ;\
	   ${RM} -f ex3_nonconforming_simplex_3_fv.tmp
runex3_nonconforming_tensor_2_fv:
	-@${MPIEXEC} -n 4 ./ex3 -test_fv_grad -test_injector -petsclimiter_type none -petscpartitioner_type simple -tree -simplex 0 -dim 2 -num_comp 2 > ex3_nonconforming_tensor_2_fv.tmp 2>&1;\
	   if (${DIFF} output/ex3_nonconforming_tensor_2_fv.out ex3_nonconforming_tensor_2_fv.tmp) then true ;\
	   else printf "${PWD}\nPossible problem with runex3_nonconforming_tensor_2_fv, diffs above\n=========================================\n"; fi ;\
	   ${RM} -f ex3_nonconforming_tensor_2_fv.tmp
runex3_nonconforming_tensor_3_fv:
	-@${MPIEXEC} -n 4 ./ex3 -test_fv_grad -test_injector -petsclimiter_type none -petscpartitioner_type simple -tree -simplex 0 -dim 3 -num_comp 3 > ex3_nonconforming_tensor_3_fv.tmp 2>&1;\
	   if (${DIFF} output/ex3_nonconforming_tensor_3_fv.out ex3_nonconforming_tensor_3_fv.tmp) then true ;\
	   else printf "${PWD}\nPossible problem with runex3_nonconforming_tensor_3_fv, diffs above\n=========================================\n"; fi ;\
	   ${RM} -f ex3_nonconforming_tensor_3_fv.tmp
runex3_nonconforming_simplex_2_view:
	-@${MPIEXEC} -n 4 ./ex3 -petscpartitioner_type simple -tree -num_comp 2 -simplex 1 -dim 2 -dm_view vtk:nonconformal_simplex_2.vtu:vtk_vtu
runex3_nonconforming_tensor_2_view:
	-@${MPIEXEC} -n 4 ./ex3 -petscpartitioner_type simple -tree -num_comp 2 -simplex 0 -dim 2 -dm_view vtk:nonconformal_tensor_2.vtu:vtk_vtu
runex3_nonconforming_simplex_3_view:
	-@${MPIEXEC} -n 4 ./ex3 -petscpartitioner_type simple -tree -num_comp 3 -simplex 1 -dim 3 -dm_view vtk:nonconformal_simplex_3.vtu:vtk_vtu
runex3_nonconforming_tensor_3_view:
	-@${MPIEXEC} -n 4 ./ex3 -petscpartitioner_type simple -tree -num_comp 3 -simplex 0 -dim 3 -dm_view vtk:nonconformal_tensor_3.vtu:vtk_vtu

runex6:
	-@${MPIEXEC} -n 1 ./ex6 > ex6.tmp 2>&1;\
	   if (${DIFF} output/ex6.out ex6.tmp) then true ;  \
	   else printf "${PWD}\nPossible problem with runex6, diffs above\n=========================================\n"; fi ;\
	   ${RM} -f ex6.tmp

runex15_0:
	-@${MPIEXEC} -n 2 ./ex15 -verbose -globaltonatural_sf_view > ex15_0.tmp 2>&1;\
          if (${DIFF} output/ex15_0.out ex15_0.tmp > /dev/null 2>&1) then x='good'; fi ;\
           if (${DIFF} output/ex15_0_alt.out ex15_0.tmp > /dev/null 2>&1) then x='good'; fi; \
           if [ "$$x" = "bad" ]; then ${DIFF} output/ex15_0.out ex15_0.tmp ; ${DIFF} output/ex15_0_alt.out ex15_0.tmp ; printf "${PWD}\nPossible problem with ex15_0, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex15_0.tmp

runex20_2d:
	-@${MPIEXEC} -n 1 ./ex20 -pre_adapt_dm_view ::ascii_info_detail -post_adapt_dm_view ::ascii_info_detail -dim 2 > ex20_2d.tmp 2>&1;\
		if (${DIFF} output/ex20_2d.out ex20_2d.tmp) then true ;\
		else printf "${PWD}\nPossible problem with runex20_2d, diffs above\n================================================================\n"; fi ;\
		${RM} -f ex20_2d.tmp

runex20_3d:
	-@${MPIEXEC} -n 1 ./ex20 -pre_adapt_dm_view ::ascii_info_detail -post_adapt_dm_view ::ascii_info_detail -dim 3 > ex20_3d.tmp 2>&1;\
		if (${DIFF} output/ex20_3d.out ex20_3d.tmp) then true ;\
		else printf "${PWD}\nPossible problem with runex20_3d, diffs above\n================================================================\n"; fi ;\
		${RM} -f ex20_3d.tmp

TESTEXAMPLES_C           = ex3.PETSc runex3_constraints runex3_nonconforming_tensor_2 runex3_nonconforming_tensor_3 runex3_nonconforming_tensor_2_fv runex3_nonconforming_tensor_3_fv ex3.rm ex6.PETSc runex6 ex6.rm
TESTEXAMPLES_C_NOTSINGLE = ex3.PETSc runex3_nonconforming_tensor_2_hi ex3.rm
TESTEXAMPLES_HDF5        = ex15.PETSc runex15_0 ex15.rm
TESTEXAMPLES_TRIANGLE    = ex3.PETSc runex3_nonconforming_simplex_2 runex3_nonconforming_simplex_2_hi runex3_nonconforming_simplex_2_fv ex3.rm ex20.PETSc runex20_2d ex20.rm
TESTEXAMPLES_CTETGEN     = ex1.PETSc runex1 runex1_2 runex1_test_shape ex1.rm ex3.PETSc runex3 runex3_nonconforming_simplex_3 runex3_nonconforming_simplex_3_hi runex3_nonconforming_simplex_3_fv ex3.rm ex20.PETSc runex20_3d ex20.rm
TESTEXAMPLES_F90         = ex1f90.PETSc runex1f90 ex1f90.rm ex2f90.PETSc runex2f90 ex2f90.rm

include ${PETSC_DIR}/lib/petsc/conf/test
