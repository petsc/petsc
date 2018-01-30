
# This directory contains SNES example programs for solving systems of
# nonlinear equations.

#CPPFLAGS	 = -I/PETSc3/geodynamics/PetscSimulationsViewers/src
CFLAGS           =
FFLAGS		 =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = src/snes/examples/tutorials/
MANSEC           = SNES
EXAMPLESC	 = ex1.c ex2.c ex3.c  ex5.c ex5s.c ex9.c \
                ex12.c ex14.c ex15.c ex18.c ex19.c ex20.c ex21.c ex22.c \
                ex25.c ex28.c ex30.c ex33.c \
                ex35.c ex42.c ex46.c ex48.c \
                ex56.c ex58.c ex59.c \
                ex62.c ex70.c ex77.c ex78.c \
                ex47cu.cu
EXAMPLESF	 = ex1f.F90 ex5f.F90 ex5f90.F90 ex5f90t.F90 ex5fs.F ex40f90.F90 ex73f90t.F90
EXAMPLESCH	 =
EXAMPLESFH       = ex5f.h
EXAMPLESMATLAB   = ex5m.m  ex61genm.m ex61m.m
DIRS		 = ex10d network

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_SNES_LIB}
	${RM} ex1.o
ex1f: ex1f.o  chkopts
	-${FLINKER} -o ex1f ex1f.o ${PETSC_SNES_LIB}
	${RM} ex1f.o
ex1f90: ex1f90.o  chkopts
	-${FLINKER} -o ex1f90 ex1f90.o ${PETSC_SNES_LIB}
	${RM} ex1f90.o
ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_SNES_LIB}
	${RM} ex2.o
ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o ${PETSC_SNES_LIB}
	${RM} ex3.o
ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o ${PETSC_SNES_LIB}
	${RM} ex4.o
ex5: ex5.o chkopts
	-${CLINKER} -o ex5 ex5.o ${PETSC_SNES_LIB}
	${RM} ex5.o
ex5f: ex5f.o  chkopts
	-${FLINKER} -o ex5f ex5f.o ${PETSC_SNES_LIB}
	${RM} ex5f.o
#
#  The SGI parallelizing compiler generates incorrect code by treating
#  the math functions (such as sqrt and exp) as local variables. The
#  sed below patches this.
#
ex5s: chkopts
	@if [ "${PETSC_ARCH}" != "IRIX64" ]; then echo "Only for PETSC_ARCH of IRIX64"; false ; fi
	-${CC} -pca keep  -WK,-lo=l ${FCONF} ${CFLAGS} -c ex5s.c
	sed "s/, sqrt/ /g"   ex5s.M | sed "s/, exp/ /g"  > ex5s_tmp.c
	-${CC} -mp ${PCC_FLAGS} ${CFLAGS} ${CCPPFLAGS} -c ex5s_tmp.c
	-${FC} -pfa keep -mp -64 ${FC_FLAGS} ${FFLAGS} ${FCPPFLAGS} -c ex5fs.F
	-${CLINKER} -mp -o ex5s ex5s_tmp.o ex5fs.o ${PETSC_SNES_LIB}
	${RM} ex5s.o
ex5f90: ex5f90.o  chkopts
	-${FLINKER} -o ex5f90 ex5f90.o ${PETSC_SNES_LIB}
	${RM} ex5f90.o
ex5f90t: ex5f90t.o  chkopts
	-${FLINKER} -o ex5f90t ex5f90t.o ${PETSC_SNES_LIB}
	${RM} ex5f90t.o
ex6: ex6.o  chkopts
	-${CLINKER} -o ex6 ex6.o ${PETSC_SNES_LIB}
	${RM} ex6.o
ex8: ex8.o chkopts
	-${CLINKER} -o ex8 ex8.o  ${PETSC_SNES_LIB}
	${RM} ex8.o
ex9: ex9.o chkopts
	-${CLINKER} -o ex9 ex9.o ${PETSC_SNES_LIB}
	${RM} ex9.o
ex10: ex10.o chkopts
	-${CLINKER} -o ex10 ex10.o ${PETSC_SNES_LIB}
	${RM} ex10.o
ex12: ex12.o chkopts
	-${CLINKER} -o ex12 ex12.o ${PETSC_SNES_LIB}
	${RM} ex12.o
ex13: ex13.o chkopts
	-${CLINKER} -o ex13 ex13.o ${PETSC_SNES_LIB}
	${DSYMUTIL} ex13
	${RM} ex13.o
ex14: ex14.o chkopts
	-${CLINKER} -o ex14 ex14.o ${PETSC_SNES_LIB}
	${RM} ex14.o
ex15: ex15.o chkopts
	-${CLINKER} -o ex15 ex15.o ${PETSC_SNES_LIB}
	${RM} ex15.o
ex16: ex16.o chkopts
	-${CLINKER} -o ex16 ex16.o ${PETSC_SNES_LIB}
	${RM} ex16.o
ex17: ex17.o chkopts
	-${CLINKER} -o ex17 ex17.o ${PETSC_SNES_LIB}
	${RM} ex17.o
ex18: ex18.o chkopts
	-${CLINKER} -o ex18 ex18.o ${PETSC_SNES_LIB}
	${RM} ex18.o
ex19:  ex19.o chkopts
	-${CLINKER} -o ex19  ex19.o ${PETSC_SNES_LIB}
	${RM} ex19.o
ex19tu:  ex19tu.o chkopts
	-${CLINKER} -o ex19tu  ex19tu.o ${PETSC_SNES_LIB}
	${RM} ex19tu.o
ex20: ex20.o chkopts
	-${CLINKER} -o ex20 ex20.o ${PETSC_SNES_LIB}
	${RM} ex20.o
ex21: ex21.o chkopts
	-${CLINKER} -o ex21 ex21.o ${PETSC_SNES_LIB}
	${RM} ex21.o
ex22: ex22.o chkopts
	-${CLINKER} -o ex22 ex22.o ${PETSC_SNES_LIB}
	${RM} ex22.o
ex23: ex23.o chkopts
	-${CLINKER} -o ex23 ex23.o ${PETSC_SNES_LIB}
	${RM} ex23.o
ex24: ex24.o chkopts
	-${CLINKER} -o ex24 ex24.o ${PETSC_SNES_LIB}
	${RM} ex24.o
ex25: ex25.o chkopts
	-${CLINKER} -o ex25 ex25.o ${PETSC_SNES_LIB}
	${RM} ex25.o
ex26: ex26.o chkopts
	-${CLINKER} -o ex26 ex26.o ${PETSC_SNES_LIB}
	${RM} ex26.o
ex27: ex27.o chkopts
	-${CLINKER} -o ex27 ex27.o ${PETSC_SNES_LIB}
	${RM} ex27.o
ex28: ex28.o chkopts
	-${CLINKER} -o ex28 ex28.o ${PETSC_SNES_LIB}
	${RM} ex28.o
ex29: ex29.o chkopts
	-${CLINKER} -o ex29 ex29.o ${PETSC_SNES_LIB}
	${RM} ex29.o
ex30: ex30.o chkopts
	-${CLINKER} -o ex30 ex30.o ${PETSC_SNES_LIB}
	${RM} ex30.o
ex32: ex32.o chkopts
	-${CLINKER} -o ex32 ex32.o ${PETSC_SNES_LIB}
	${RM} ex32.o
ex33: ex33.o chkopts
	-${CLINKER} -o ex33 ex33.o ${PETSC_SNES_LIB}
	${RM} ex33.o
ex35: ex35.o chkopts
	-${CLINKER} -o ex35 ex35.o ${PETSC_SNES_LIB}
	${RM} ex35.o
ex38: ex38.o  chkopts
	-${CLINKER} -o ex38 ex38.o ${PETSC_SNES_LIB}
	${RM} ex38.o
ex39f90: ex39f90.o  chkopts
	-${FLINKER} -o ex39f90 ex39f90.o ${PETSC_SNES_LIB}
	${RM} ex39f90.o
ex40f90: ex40f90.o  chkopts
	-${FLINKER} -o ex40f90 ex40f90.o ${PETSC_SNES_LIB}
	${RM} ex40f90.o
ex41: ex41.o  chkopts
	-${CLINKER} -o ex41 ex41.o ${PETSC_SNES_LIB}
	${RM} ex41.o
ex42: ex42.o  chkopts
	-${CLINKER} -o ex42 ex42.o ${PETSC_SNES_LIB}
	${RM} ex42.o
ex43: ex43.o  chkopts
	-${CLINKER} -o ex43 ex43.o ${PETSC_SNES_LIB}
	${RM} ex43.o
ex44: ex44.o  chkopts
	-${CLINKER} -o ex44 ex44.o ${PETSC_SNES_LIB}
	${RM} ex44.o
ex45: ex45.o  chkopts
	-${CLINKER} -o ex45 ex45.o ${PETSC_SNES_LIB}
	${RM} ex45.o
ex46: ex46.o  chkopts
	-${CLINKER} -o ex46 ex46.o ${PETSC_SNES_LIB}
	${RM} ex46.o
ex47cu: ex47cu.o  chkopts
	-${CLINKER} -o ex47cu ex47cu.o ${PETSC_SNES_LIB}
	${RM} ex47cu.o
ex48: ex48.o  chkopts
	-${CLINKER} -o ex48 ex48.o ${PETSC_SNES_LIB}
	${RM} ex48.o
ex53: ex53.o  chkopts
	-${CLINKER} -o ex53 ex53.o ${PETSC_SNES_LIB}
	${RM} ex53.o
ex54: ex54.o  chkopts
	-${CLINKER} -o ex54 ex54.o ${PETSC_SNES_LIB}
	${RM} ex54.o
ex55: ex55.o  chkopts
	-${CLINKER} -o ex55 ex55.o ${PETSC_SNES_LIB}
	${RM} ex55.o
ex56: ex56.o  chkopts
	-${CLINKER} -o ex56 ex56.o ${PETSC_SNES_LIB}
	${RM} ex56.o
ex57: ex57.o  chkopts
	-${CLINKER} -o ex57 ex57.o ${PETSC_SNES_LIB}
	${RM} ex57.o
ex58: ex58.o  chkopts
	-${CLINKER} -o ex58 ex58.o ${PETSC_SNES_LIB}
	${RM} ex58.o
ex59: ex59.o  chkopts
	-${CLINKER} -o ex59 ex59.o ${PETSC_SNES_LIB}
	${RM} ex59.o
ex60: ex60.o  chkopts
	-${CLINKER} -o ex60 ex60.o ${PETSC_SNES_LIB}
	${RM} ex60.o
ex61: ex61.o  chkopts
	-${CLINKER} -o ex61 ex61.o ${PETSC_SNES_LIB}!
	${RM} ex61.o
ex61gen: ex61gen.o  chkopts
	-${CLINKER} -o ex61gen ex61gen.o ${PETSC_SNES_LIB}
	${RM} ex61gen.o
ex61view: ex61view.o  chkopts
	-${CLINKER} -o ex61view ex61view.o ${PETSC_SNES_LIB}
	${RM} ex61view.o
ex62: ex62.o  chkopts
	-${CLINKER} -o ex62 ex62.o ${PETSC_SNES_LIB}
	${RM} ex62.o
ex63: ex63.o  chkopts
	-${CLINKER} -o ex63 ex63.o ${PETSC_SNES_LIB}
	${RM} ex63.o
ex64: ex64.o  chkopts
	-${CLINKER} -o ex64 ex64.o ${PETSC_SNES_LIB}
	${RM} ex64.o
ex65: ex65.o  chkopts
	-${CLINKER} -o ex65 ex65.o ${PETSC_SNES_LIB}
	${RM} ex65.o
ex69: ex69.o  chkopts
	-${CLINKER} -o ex69 ex69.o ${PETSC_SNES_LIB}
	${RM} ex69.o
ex70: ex70.o  chkopts
	-${CLINKER} -o ex70 ex70.o ${PETSC_SNES_LIB}
	${RM} ex70.o
ex73f90t: ex73f90t.o  chkopts
	-${FLINKER} -o ex73f90t ex73f90t.o ${PETSC_SNES_LIB}
	${RM} ex73f90t.o
ex77: ex77.o  chkopts
	-${CLINKER} -o ex77 ex77.o ${PETSC_SNES_LIB}
	${RM} ex77.o
ex78: ex78.o  chkopts
	-${CLINKER} -o ex78 ex78.o ${PETSC_SNES_LIB}
	${RM} ex78.o
#-------------------------------------------------------------------------

#  these tests are used by the makefile in PETSC_DIR for basic tests of the install and should not be removed
testex5f: ex5f.PETSc
	@if [ "${PETSC_WITH_BATCH}" != "" ]; then \
           echo "Running with batch filesystem; to test run src/snes/examples/tutorials/ex5f with" ; \
           echo "your systems batch system"; \
        elif [ "${MPIEXEC}" = "/bin/false" ]; then \
           echo "*mpiexec not found*. Please run src/snes/examples/tutorials/ex5f manually"; \
        elif [ -f ex5f ]; then \
	   ${MPIEXEC} -n 1 ./ex5f > ex5f_1.tmp 2>&1; \
	   if (${DIFF} output/ex5f_1.testout ex5f_1.tmp > /dev/null 2>&1) then \
           echo "Fortran example src/snes/examples/tutorials/ex5f run successfully with 1 MPI process"; \
	   else echo "Possible error running Fortran example src/snes/examples/tutorials/ex5f with 1 MPI process"; \
           echo "See http://www.mcs.anl.gov/petsc/documentation/faq.html";\
           cat ex5f_1.tmp; fi;  \
         ${RM} -f ex5f_1.tmp ;\
         ${MAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} ex5f.rm; fi

testex19: ex19.PETSc
	-@if [ "${PETSC_WITH_BATCH}" != "" ]; then \
           echo "Running with batch filesystem; to test run src/snes/examples/tutorials/ex19 with" ; \
           echo "your systems batch system"; \
        elif [ "${MPIEXEC}" = "/bin/false" ]; then \
           echo "*mpiexec not found*. Please run src/snes/examples/tutorials/ex19 manually"; \
	elif [ -f ex19 ]; then \
           ${MPIEXEC} -n 1 ./ex19 -da_refine 3 -pc_type mg -ksp_type fgmres  > ex19_1.tmp 2>&1; \
	   if (${DIFF} output/ex19_1.testout ex19_1.tmp > /dev/null 2>&1) then \
           echo "C/C++ example src/snes/examples/tutorials/ex19 run successfully with 1 MPI process"; \
	   else echo "Possible error running C/C++ src/snes/examples/tutorials/ex19 with 1 MPI process"; \
           echo "See http://www.mcs.anl.gov/petsc/documentation/faq.html";\
           cat ex19_1.tmp; fi; \
	if [ ! "${MPI_IS_MPIUNI}" ]; then \
           ${MPIEXEC} -n 2 ./ex19 -da_refine 3 -pc_type mg -ksp_type fgmres  > ex19_1.tmp 2>&1; \
	   if (${DIFF} output/ex19_1.testout ex19_1.tmp > /dev/null 2>&1) then \
           echo "C/C++ example src/snes/examples/tutorials/ex19 run successfully with 2 MPI processes"; \
	   else echo "Possible error running C/C++ src/snes/examples/tutorials/ex19 with 2 MPI processes"; \
           echo "See http://www.mcs.anl.gov/petsc/documentation/faq.html";\
           cat ex19_1.tmp; fi; fi; \
        ${RM} -f ex19_1.tmp; fi
runex19_hypre:
	-@${MPIEXEC} -n 2 ./ex19 -da_refine 3 -snes_monitor_short -pc_type hypre > ex19_1.tmp 2>&1; \
	   if (${DIFF} output/ex19_hypre.out ex19_1.tmp) then true; \
	   else  printf "${PWD}\nPossible problem with ex19_hypre, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex19_1.tmp
runex19_ml:
	-@${MPIEXEC} -n 2 ./ex19 -da_refine 3 -snes_monitor_short -pc_type ml > ex19_1.tmp 2>&1; \
	   if (${DIFF} output/ex19_ml.out ex19_1.tmp) then true; \
	   else  printf "${PWD}\nPossible problem with ex19_ml, diffs above\n=========================================\n"; fi; \
           ${RM} -f ex19_1.tmp
runex19_fieldsplit_mumps:
	-@${MPIEXEC} -n 2 ./ex19 -pc_type fieldsplit -pc_fieldsplit_block_size 4 -pc_fieldsplit_type SCHUR -pc_fieldsplit_0_fields 0,1,2 -pc_fieldsplit_1_fields 3 -fieldsplit_0_pc_type lu -fieldsplit_1_pc_type lu -snes_monitor_short -ksp_monitor_short  -fieldsplit_0_pc_factor_mat_solver_type mumps -fieldsplit_1_pc_factor_mat_solver_type mumps > ex19_6.tmp 2>&1; \
	   if (${DIFF} output/ex19_fieldsplit_5.out ex19_6.tmp) then true; \
	   else  printf "${PWD}\nPossible problem with ex19_fieldsplit_fieldsplit_mumps, diffs above\n=========================================\n"; fi; \
           ${RM} -f ex19_6.tmp
runex19_superlu_dist:
	-@${MPIEXEC} -n 1 ./ex19 -da_grid_x 20 -da_grid_y 20 -pc_type lu -pc_factor_mat_solver_type superlu_dist > ex19.tmp 2>&1; \
	   if (${DIFF} output/ex19_superlu.out ex19.tmp) then true; \
	   else  printf "${PWD}\nPossible problem with ex19_superlu_dist, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex19.tmp

#-------------------------------------------------------------------------

runex12_p4est_convergence_test:
	-@echo "=================="
	-@echo "Refinement level 2"
	-@echo "=================="
	-@${MPIEXEC} -n 4 ./ex12 -quiet -run_type test -interpolate 1 -petscspace_order 1 -simplex 0 -petscspace_poly_tensor -dm_plex_convert_type p4est -dm_forest_minimum_refinement 2 -dm_forest_initial_refinement 2 -dm_forest_maximum_refinement 4 -dm_p4est_refine_pattern hash
	-@echo "=================="
	-@echo "Refinement level 3"
	-@echo "=================="
	-@${MPIEXEC} -n 4 ./ex12 -quiet -run_type test -interpolate 1 -petscspace_order 1 -simplex 0 -petscspace_poly_tensor -dm_plex_convert_type p4est -dm_forest_minimum_refinement 3 -dm_forest_initial_refinement 3 -dm_forest_maximum_refinement 5 -dm_p4est_refine_pattern hash
	-@echo "=================="
	-@echo "Refinement level 4"
	-@echo "=================="
	-@${MPIEXEC} -n 4 ./ex12 -quiet -run_type test -interpolate 1 -petscspace_order 1 -simplex 0 -petscspace_poly_tensor -dm_plex_convert_type p4est -dm_forest_minimum_refinement 4 -dm_forest_initial_refinement 4 -dm_forest_maximum_refinement 6 -dm_p4est_refine_pattern hash
	-@echo "=================="
	-@echo "Refinement level 5"
	-@echo "=================="
	-@${MPIEXEC} -n 4 ./ex12 -quiet -run_type test -interpolate 1 -petscspace_order 1 -simplex 0 -petscspace_poly_tensor -dm_plex_convert_type p4est -dm_forest_minimum_refinement 5 -dm_forest_initial_refinement 5 -dm_forest_maximum_refinement 7 -dm_p4est_refine_pattern hash

include ${PETSC_DIR}/lib/petsc/conf/test
