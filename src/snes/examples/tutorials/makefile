
# This directory contains SNES example programs for solving systems of
# nonlinear equations.  

#CFLAGS		 = -I/PETSc3/geodynamics/PetscSimulationsViewers/src
CFLAGS       =
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = src/snes/examples/tutorials/
MANSEC           = SNES
EXAMPLESC	 = ex1.c ex2.c ex3.c ex5.c ex5s.c ex6.c \
                   ex14.c ex18.c ex19.c ex20.c ex21.c ex22.c \
                   ex23.c ex24.c ex25.c ex26.c ex27.c ex29.c ex30.c ex31.c ex38.c ex42.c ex43.c \
                   ex44.c ex45.c ex47.c ex48.c
EXAMPLESF	 = ex1f.F ex5f.F ex5f90.F ex5fs.F ex33f.F mex34f90.F ex34f90.F \
		   mex35f90.F90 ex35f90.F90 mex36f90.F90 ex36f90.F90 mex37f90.F90 ex37f90.F90 ex39f90.F
EXAMPLESCH	 = ex43-44.h
EXAMPLESFH       = ex5f.h
EXAMPLESMATLAB   = ex5m.m
DIRS		 = ex10d

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

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
ex4tu: ex4tu.o  chkopts
	-${CLINKER} -o ex4tu ex4tu.o ${PETSC_SNES_LIB}
	${RM} ex4tu.o
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
ex6: ex6.o  chkopts
	-${CLINKER} -o ex6 ex6.o ${PETSC_SNES_LIB}
	${RM} ex6.o
ex7: ex7.o  chkopts
	-${CLINKER} -o ex7 ex7.o -lsimviewers ${PETSC_SNES_LIB}
	${RM} ex7.o
ex8: ex8.o chkopts
	-${CLINKER} -o ex8 ex8.o -lsimviewers ${PETSC_SNES_LIB}
	${RM} ex8.o
ex9: ex9.o chkopts
	-${CLINKER} -o ex9 ex9.o ${PETSC_SNES_LIB}
	${RM} ex9.o
ex12: ex12.o chkopts
	-${CLINKER} -o ex12 ex12.o ${PETSC_SNES_LIB}
	${RM} ex12.o
ex13: ex13.o chkopts
	-${CLINKER} -o ex13 ex13.o ${PETSC_SNES_LIB}
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
ex31: ex31.o chkopts
	-${CLINKER} -o ex31 ex31.o ${PETSC_SNES_LIB}
	${RM} ex31.o
ex32: ex32.o chkopts
	-${CLINKER} -o ex32 ex32.o ${PETSC_SNES_LIB}
	${RM} ex32.o
ex33f: ex33f.o  chkopts
	-${FLINKER} -o ex33f ex33f.o ${PETSC_SNES_LIB}
	${RM} ex33f.o
ex34f90: mex34f90.o ex34f90.o  chkopts
	-${FLINKER} -o ex34f90 ex34f90.o ${PETSC_SNES_LIB}
	${RM} ex34f90.o mex34f90.mod mex34f90.o mex34finterfaces.mod
ex35f90: mex35f90.o ex35f90.o  chkopts
	-${FLINKER} -o ex35f90 ex35f90.o ${PETSC_SNES_LIB}
	${RM} ex35f90.o mex35f90.mod mex35f90.o mex35finterfaces.mod
ex36f90: mex36f90.o ex36f90.o  chkopts
	-${FLINKER} -o ex36f90 ex36f90.o ${PETSC_SNES_LIB}
	${RM} ex36f90.o mex36f90.mod mex36f90.o mex36finterfaces.mod
ex37f90: mex37f90.o ex37f90.o  chkopts
	-${FLINKER} -o ex37f90 ex37f90.o ${PETSC_SNES_LIB}
	${RM} ex37f90.o mex37f90.mod mex37f90.o mex37finterfaces.mod
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
ex47: ex47.o  chkopts
	-${CLINKER} -o ex47 ex47.o ${PETSC_SNES_LIB}
	${RM} ex47.o
ex48: ex48.o  chkopts
	-${CLINKER} -o ex48 ex48.o ${PETSC_SNES_LIB}
	${RM} ex48.o

#--------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1 -ksp_gmres_cgs_refinement_type refine_always -snes_monitor_short > ex1_1.tmp 2>&1;   \
	   if (${DIFF} output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   ${RM} -f ex1_1.tmp
runex1_2:
	-@${MPIEXEC} -n 2 ./ex1 -snes_mf_operator -snes_fd -snes_monitor_short > ex1_2.tmp 2>&1;   \
	   if (${DIFF} output/ex1_2.out ex1_2.tmp) then true; \
	   else echo "Possible problem with ex1_2, diffs above"; fi; \
	   ${RM} -f ex1_2.tmp
runex1f:
	-@${MPIEXEC} -n 1 ./ex1f -ksp_gmres_cgs_refinement_type refine_always -snes_monitor_short > ex1f_1.tmp 2>&1;   \
	   if (${DIFF} output/ex1f_1.out ex1f_1.tmp) then true; \
	   else echo "Possible problem with ex1f_1, diffs above"; fi; \
	   ${RM} -f ex1f_1.tmp
runex2:
	-@${MPIEXEC} -n 1 ./ex2 -nox -snes_monitor_cancel -snes_monitor_short \
		 -snes_view -pc_type jacobi -ksp_gmres_cgs_refinement_type refine_always >ex2_1.tmp 2>&1;	\
	   if (${DIFF} output/ex2_1.out ex2_1.tmp) then true; \
	   else echo "Possible problem with ex2_1, diffs above"; fi; \
	   ${RM} -f ex2_1.tmp
runex2_2:
	-@${MPIEXEC} -n 1 ./ex2 -nox -snes_monitor_cancel -snes_monitor_short \
		-snes_type tr -snes_view \
		-pc_type jacobi -ksp_gmres_cgs_refinement_type refine_always > ex2_2.tmp 2>&1; \
	   if (${DIFF} output/ex2_2.out ex2_2.tmp) then true; \
	   else echo "Possible problem with ex2_2, diffs above"; fi; \
	   ${RM} -f ex2_2.tmp
runex2_3:
	-@${MPIEXEC} -n 1 ./ex2 -nox -snes_monitor_cancel -snes_monitor_short -malloc no \
		 -snes_view -pc_type jacobi -ksp_gmres_cgs_refinement_type refine_always >ex2_3.tmp 2>&1;	\
	   if (${DIFF} output/ex2_1.out ex2_3.tmp) then true; \
	   else echo "Possible problem with ex2_3, diffs above"; fi; \
	   ${RM} -f ex2_3.tmp
runex3:
	-@${MPIEXEC} -n 1 ./ex3 -nox -snes_monitor_cancel -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex3_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex3_1.out ex3_1.tmp) then true; \
	   else echo "Possible problem with ex3_1, diffs above"; fi; \
	   ${RM} -f ex3_1.tmp
runex3_2:
	-@${MPIEXEC} -n 3 ./ex3 -nox -pc_type asm -mat_type mpiaij  \
	   -snes_monitor_cancel -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex3_2.tmp 2>&1; \
	   if (${DIFF} output/ex3_2.out ex3_2.tmp) then true; \
	   else echo "Possible problem with ex3_2, diffs above"; fi; \
	   ${RM} -f ex3_2.tmp
runex3_3:
	-@${MPIEXEC} -n 2 ./ex3 -nox -snes_monitor_cancel -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex3_3.tmp 2>&1; \
	   if (${DIFF} output/ex3_3.out ex3_3.tmp) then true; \
	   else echo "Possible problem with ex3_3, diffs above"; fi; \
	   ${RM} -f ex3_3.tmp
runex3_4:
	-@${MPIEXEC} -n 1 ./ex3 -nox -pre_check_iterates -post_check_iterates > ex3_4.tmp 2>&1; \
	   if (${DIFF} output/ex3_4.out ex3_4.tmp) then true; \
	   else echo "Possible problem with ex3_4, diffs above"; fi; \
	   ${RM} -f ex3_4.tmp
runex5:
	-@${MPIEXEC} -n 4 ./ex5 -snes_mf -da_processors_x 4 -da_processors_y 1 -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex5_1.tmp 2>&1; \
	   if (${DIFF} output/ex5_1.out ex5_1.tmp) then true; \
	   else echo "Possible problem with ex5_1, diffs above"; fi; \
	   ${RM} -f ex5_1.tmp
runex5_2:
	-@${MPIEXEC} -n 4 ./ex5 -da_processors_x 2 -da_processors_y 2 -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex5_2.tmp 2>&1; \
	   if (${DIFF} output/ex5_2.out ex5_2.tmp) then true; \
	   else echo "Possible problem with ex5_2, diffs above"; fi; \
	   ${RM} -f ex5_2.tmp
runex5_3:
	-@${MPIEXEC} -n 4 ./ex5 -snes_mf -da_processors_x 2 -da_processors_y 2 -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex5_3.tmp 2>&1;\
	   if (${DIFF} output/ex5_3.out ex5_3.tmp) then true; \
	   else echo "Possible problem with ex5_3, diffs above"; fi; \
	   ${RM} -f ex5_3.tmp
runex5_4:
	-@${MPIEXEC} -n 4 ./ex5 -da_processors_x 2 -da_processors_y 2 -snes_monitor_short -pc_type asm \
	   -pc_asm_blocks 4 -pc_asm_overlap 1 -ksp_gmres_cgs_refinement_type refine_always > ex5_4.tmp 2>&1; \
	   if (${DIFF} output/ex5_4.out ex5_4.tmp) then true; \
	   else echo "Possible problem with ex5_4, diffs above"; fi; \
	   ${RM} -f ex5_4.tmp
runex5_5:
	-@${MPIEXEC} -n 4 ./ex5 -da_processors_x 2 -da_processors_y 2 -snes_monitor_short -pc_type bjacobi \
	   -snes_view -ksp_gmres_cgs_refinement_type refine_always > ex5_5.tmp 2>&1; \
	   if (${DIFF} output/ex5_5.out ex5_5.tmp) then true; \
	   else echo "Possible problem with ex5_5, diffs above"; fi; \
	   ${RM} -f ex5_5.tmp
runex5_6:
	-@${MPIEXEC} -n 4 ./ex5 -snes_fd -da_processors_x 2 -da_processors_y 2 -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex5_6.tmp 2>&1; \
	   if (${DIFF} output/ex5_6.out ex5_6.tmp) then true; \
	   else echo "Possible problem with ex5_6, diffs above"; fi; \
	   ${RM} -f ex5_6.tmp
runex5_7:
	-@${MPIEXEC} -n 3 ./ex5 -snes_fd  -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex5_7.tmp 2>&1;\
	   if (${DIFF} output/ex5_7.out ex5_7.tmp) then true; \
	   else echo "Possible problem with ex5_7, diffs above"; fi; \
	   ${RM} -f ex5_7.tmp
runex5_8:
	-@${MPIEXEC} -n 2 ./ex5 -snes_monitor_solution -ksp_gmres_cgs_refinement_type refine_always > /dev/null 2>&1
runex5_9:
	-@${MPIEXEC} -n 4 ./ex5 -snes_mf -snes_monitor_short>ex5_9.tmp 2>&1;\
	   if (${DIFF} output/ex5_9.out ex5_9.tmp) then true; \
	   else echo "Possible problem with ex5_9, diffs above"; fi; \
	   ${RM} -f ex5_9.tmp
runex5_10:
	-@${MPIEXEC} -n 2 ./ex5 -ksp_diagonal_scale -ksp_monitor_short -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex5_10.tmp 2>&1;\
	   if (${DIFF} output/ex5_10.out ex5_10.tmp) then true; \
	   else echo "Possible problem with ex5_10, diffs above"; fi; \
	   ${RM} -f ex5_10.tmp
runex5_11:
	-@${MPIEXEC} -n 4 ./ex5 -snes_mf -da_processors_x 4 -da_processors_y 1 -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always -matlab_function > ex5_11.tmp 2>&1; \
	   if (${DIFF} output/ex5_1.out ex5_11.tmp) then true; \
	   else echo "Possible problem with ex5_11, diffs above"; fi; \
	   ${RM} -f ex5_11.tmp
runex5_13:
	-@for F in " " -matlab_function ; do \
            for J in " " -fd_jacobian -adic_jacobian; do \
              ${MPIEXEC} -n 2 ./ex5 -snes_monitor_short $$F $$J > ex5_13.tmp 2>&1; \
              if (${DIFF} output/ex5_13.out ex5_13.tmp) then true; \
	      else echo "Possible problem with ex5_13, diffs above " $F $J; fi; \
              ${RM} -f ex5_13.tmp; \
              ${MPIEXEC} -n 2 ./ex5 -snes_monitor_short $$F $$J -adicmf_jacobian > ex5_13.tmp 2>&1; \
              if (${DIFF} output/ex5_13.out ex5_13.tmp) then true; \
	      else echo "Possible problem with ex5_13, diffs above " $F $J; fi; \
              ${RM} -f ex5_13.tmp; \
            done; \
          done
runex5_14:
	-@${MPIEXEC} -n 4 ./ex5 -snes_monitor_short -fd_jacobian_ghosted > ex5_14.tmp 2>&1; \
	   if (${DIFF} output/ex5_13.out ex5_14.tmp) then true; \
	   else echo "Possible problem with ex5_14, diffs above "; fi; \
           ${RM} -f ex5_14.tmp; 
runex5f:
	-@${MPIEXEC} -n 4 ./ex5f -snes_mf -da_processors_x 4 -da_processors_y 1 -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex5f_1.tmp 2>&1; \
	   if (${DIFF} output/ex5f_1.out ex5f_1.tmp) then true; \
	   else echo "Possible problem with ex5f_1, diffs above"; fi; \
	   ${RM} -f ex5f_1.tmp
runex5f_2:
	-@${MPIEXEC} -n 4 ./ex5f  -da_processors_x 2 -da_processors_y 2 -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex5f_2.tmp 2>&1; \
	   if (${DIFF} output/ex5f_2.out ex5f_2.tmp) then true; \
	   else echo "Possible problem with ex5f_2, diffs above"; fi; \
	   ${RM} -f ex5f_2.tmp
runex5f_3:
	-@${MPIEXEC} -n 3 ./ex5f -snes_fd  -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex5f_3.tmp 2>&1;\
	   if (${DIFF} output/ex5f_3.out ex5f_3.tmp) then true; \
	   else echo "Possible problem with ex5f_3, diffs above"; fi; \
	   ${RM} -f ex5f_3.tmp
runex5f_4:
	-@${MPIEXEC} -n 2 ./ex5f -adifor_jacobian -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex5f_4.tmp 2>&1;\
	   if (${DIFF} output/ex5f_4.out ex5f_4.tmp) then true; \
	   else echo "Possible problem with ex5f_4, diffs above"; fi; \
	   ${RM} -f ex5f_4.tmp
runex5f_5:
	-@${MPIEXEC} -n 2 ./ex5f -adiformf_jacobian  -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex5f_5.tmp 2>&1;\
	   if (${DIFF} output/ex5f_5.out ex5f_5.tmp) then true; \
	   else echo "Possible problem with ex5f_5, diffs above"; fi; \
	   ${RM} -f ex5f_5.tmp
testex5f: ex5f.PETSc
	@if [ "${PETSC_WITH_BATCH}" != "" ]; then \
           echo "Running with batch filesystem; to test run src/snes/examples/tutorials/ex5f with" ; \
           echo "your systems batch system"; \
        elif [ -f ex5f ]; then \
	   ${MPIEXEC} -n 1 ./ex5f > ex5f_1.tmp 2>&1; \
	   if (${DIFF} output/ex5f_1.testout ex5f_1.tmp > /dev/null 2>&1) then \
           echo "Fortran example src/snes/examples/tutorials/ex5f run successfully with 1 MPI process"; \
	   else echo "Error running Fortran example src/snes/examples/tutorials/ex5f with 1 MPI process"; \
           echo "See http://www.mcs.anl.gov/petsc/petsc-as/documentation/troubleshooting.html";\
           cat ex5f_1.tmp; fi;  \
         ${RM} -f ex5f_1.tmp ;\
         ${MAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} ex5f.rm; fi
runex5f90:
	-@${MPIEXEC} -n 4 ./ex5f90 -snes_mf -da_processors_x 4 -da_processors_y 1 -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex5f90_1.tmp 2>&1; \
	   if (${DIFF} output/ex5f90_1.out ex5f90_1.tmp) then true; \
	   else echo "Possible problem with ex5f90_1, diffs above"; fi; \
	   ${RM} -f ex5f90_1.tmp
runex5f90_2:
	-@${MPIEXEC} -n 4 ./ex5f90 -da_processors_x 2 -da_processors_y 2 -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex5f90_2.tmp 2>&1; \
	   if (${DIFF} output/ex5f90_2.out ex5f90_2.tmp) then true; \
	   else echo "Possible problem with ex5f90_2, diffs above"; fi; \
	   ${RM} -f ex5f90_2.tmp
runex5f90_3:
	-@${MPIEXEC} -n 3 ./ex5f90 -snes_fd  -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex5f90_3.tmp 2>&1;\
	   if (${DIFF} output/ex5f90_3.out ex5f90_3.tmp) then true; \
	   else echo "Possible problem with ex5f90_3, diffs above"; fi; \
	   ${RM} -f ex5f90_3.tmp
runex5f90_4:
	-@${MPIEXEC} -n 3 ./ex5f90 -snes_mf_operator  -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex5f90_4.tmp 2>&1;\
	   if (${DIFF} output/ex5f90_4.out ex5f90_4.tmp) then true; \
	   else echo "Possible problem with ex5f90_4, diffs above"; fi; \
	   ${RM} -f ex5f90_4.tmp
runex5f90_5:
	-@${MPIEXEC} -n 1 ./ex5f90 -test_appctx > ex5f90_5.tmp 2>&1;\
	   if (${DIFF} output/ex5f90_5.out ex5f90_5.tmp) then true; \
	   else echo "Possible problem with ex5f90_5, diffs above"; fi; \
	   ${RM} -f ex5f90_5.tmp
runex6:
	-@${MPIEXEC} -n 1 ./ex6 -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex6_1.tmp 2>&1;   \
	   if (${DIFF} output/ex6_1.out ex6_1.tmp) then true; \
	   else echo "Possible problem with ex6_1, diffs above"; fi; \
	   ${RM} -f ex6_1.tmp
runex6_2:
	-@${MPIEXEC} -n 1 ./ex6 -snes_mf -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex6_2.tmp 2>&1;   \
	   if (${DIFF} output/ex6_2.out ex6_2.tmp) then true; \
	   else echo "Possible problem with ex6_2, diffs above"; fi; \
	   ${RM} -f ex6_2.tmp
runex8:
	-@${MPIEXEC} -n 2 ./ex8  -mx 16 -my 16 -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always \
                    -lidvelocity 0. -grashof 1000. > ex8_1.tmp 2>&1; \
           if (${DIFF} output/ex8_1.out ex8_1.tmp) then true; \
           else echo "Possible problem with ex8_1, diffs above"; fi; \
           ${RM} -f ex8_1.tmp
runex8_2:
	-@${MPIEXEC} -n 2 ./ex8 -snes_monitor_solution -ksp_gmres_cgs_refinement_type refine_always > /dev/null 2>&1 
runex8_3:
	-@${MPIEXEC} -n 2 ./ex8 -is_coloring_view > ex8_3.tmp 2>&1;\
           if (${DIFF} output/ex8_3.out ex8_3.tmp) then true; \
           else echo "Possible problem with ex8_3, diffs above"; fi; \
           ${RM} -f ex8_3.tmp
runex14:
	-@${MPIEXEC} -n 4 ./ex14 -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always > ex14_1.tmp 2>&1; \
	   if (${DIFF} output/ex14_1.out ex14_1.tmp) then true; \
	   else echo "Possible problem with ex14_1, diffs above"; fi; \
	   ${RM} -f ex14_1.tmp
runex14_2:
	-@${MPIEXEC} -n 4 ./ex14 -snes_monitor_short -ksp_gmres_cgs_refinement_type refine_always  > ex14_2.tmp 2>&1; \
	   if (${DIFF} output/ex14_2.out ex14_2.tmp) then true; \
	   else echo "Possible problem with ex14_2, diffs above"; fi; \
	   ${RM} -f ex14_2.tmp
runex19:
	-@${MPIEXEC} -n 2 ./ex19 -dmmg_nlevels 4 -snes_monitor_short  > ex19_1.tmp 2>&1; \
	   if (${DIFF} output/ex19_1.out ex19_1.tmp) then true; \
	   else echo "Possible problem with ex19, diffs above"; fi; \
	   ${RM} -f ex19_1.tmp
runex19_2:
	-@${MPIEXEC} -n 2 ./ex19 -dmmg_nlevels 4 -snes_monitor_short -dmmg_jacobian_fd -dmmg_iscoloring_type ghosted > ex19_2.tmp 2>&1; \
	   if (${DIFF} output/ex19_1.out ex19_2.tmp) then true; \
	   else echo "Possible problem with ex19_2, diffs above"; fi; \
	   ${RM} -f ex19_2.tmp
runex19_3: #test pc_redundant
	-@${MPIEXEC} -n 4 ./ex19 -dmmg_nlevels 4 -snes_monitor_short -pc_type redundant -mat_type aij -redundant_pc_factor_mat_solver_package mumps -pc_redundant_number 2 > ex19_3.tmp 2>&1; \
	   if (${DIFF} output/ex19_3.out ex19_3.tmp) then true; \
	   else echo "Possible problem with ex19_3, diffs above"; fi; \
	   ${RM} -f ex19_3.tmp
runex19_4: #test pc_redundant
	-@${MPIEXEC} -n 12 ./ex19 -dmmg_nlevels 4 -snes_monitor_short -pc_type redundant -mat_type aij -redundant_pc_factor_mat_solver_package mumps -pc_redundant_number 5 > ex19_4.tmp 2>&1; \
	   if (${DIFF} output/ex19_3.out ex19_4.tmp) then true; \
	   else echo "Possible problem with ex19_4, diffs above"; fi; \
	   ${RM} -f ex19_4.tmp
runex19_5: #test different scatters
	-@for A in " " -vecscatter_rsend -vecscatter_ssend -vecscatter_alltoall "-vecscatter_alltoall -vecscatter_nopack" -vecscatter_window; do \
           for B in " " -vecscatter_merge ; do \
             ${MPIEXEC} -n 4 ./ex19 -dmmg_nlevels 4 -snes_monitor_short $$A $$B > ex19_5.tmp 2>&1; \
	     if (${DIFF} output/ex19_5.out ex19_5.tmp) then true; \
	     else echo "Possible problem with ex19_5 " $$A $$B " diffs above"; fi; \
           done;\
         done;\
	 ${RM} -f ex19_5.tmp
# fieldsplit preconditioner tests
runex19_6: 
	-@${MPIEXEC} -n 1 ./ex19 -snes_monitor_short -ksp_monitor_short -pc_type fieldsplit -snes_view  -preload off > ex19_6.tmp 2>&1; \
	   if (${DIFF} output/ex19_6.out ex19_6.tmp) then true; \
	   else echo "Possible problem with ex19_6, diffs above"; fi; \
	   ${RM} -f ex19_6.tmp
runex19_7: 
	-@${MPIEXEC} -n 3 ./ex19 -snes_monitor_short -ksp_monitor_short -pc_type fieldsplit -snes_view  -preload off > ex19_7.tmp 2>&1; \
	   if (${DIFF} output/ex19_7.out ex19_7.tmp) then true; \
	   else echo "Possible problem with ex19_7, diffs above"; fi; \
	   ${RM} -f ex19_7.tmp
runex19_8: 
	-@${MPIEXEC} -n 1 ./ex19 -snes_monitor_short -ksp_monitor_short -pc_type fieldsplit -pc_fieldsplit_block_size 2 -pc_fieldsplit_0_fields 0,1 -pc_fieldsplit_1_fields 0,1 -pc_fieldsplit_type multiplicative -snes_view  -preload off -fieldsplit_pc_type lu > ex19_8.tmp 2>&1; \
	   if (${DIFF} output/ex19_8.out ex19_8.tmp) then true; \
	   else echo "Possible problem with ex19_8, diffs above"; fi; \
	   ${RM} -f ex19_8.tmp
runex19_9: 
	-@${MPIEXEC} -n 3 ./ex19 -snes_monitor_short -ksp_monitor_short -pc_type fieldsplit -pc_fieldsplit_type multiplicative -snes_view  -preload off > ex19_9.tmp 2>&1; \
	   if (${DIFF} output/ex19_9.out ex19_9.tmp) then true; \
	   else echo "Possible problem with ex19_9, diffs above"; fi; \
	   ${RM} -f ex19_9.tmp
runex19_10: 
	-@${MPIEXEC} -n 3 ./ex19 -snes_monitor_short -ksp_monitor_short -pc_type fieldsplit -pc_fieldsplit_type symmetric_multiplicative -snes_view  -preload off > ex19_10.tmp 2>&1; \
	   if (${DIFF} output/ex19_10.out ex19_10.tmp) then true; \
	   else echo "Possible problem with ex19_10, diffs above"; fi; \
	   ${RM} -f ex19_10.tmp
runex19_11: #test pc_redundant
	-@${MPIEXEC} -n 4 ./ex19 -dmmg_nlevels 4 -snes_monitor_short -pc_type redundant -mat_type aij -redundant_pc_factor_mat_solver_package pastix -pc_redundant_number 2 -preload off > ex19_11.tmp 2>&1; \
	   if (${DIFF} output/ex19_11.out ex19_11.tmp) then true; \
	   else echo "Possible problem with ex19_11, diffs above"; fi; \
	   ${RM} -f ex19_11.tmp
runex19_12: #test pc_redundant
	-@${MPIEXEC} -n 12 ./ex19 -dmmg_nlevels 4 -snes_monitor_short -pc_type redundant -mat_type aij -redundant_pc_factor_mat_solver_package pastix -pc_redundant_number 5 -preload off > ex19_12.tmp 2>&1; \
	   if (${DIFF} output/ex19_12.out ex19_12.tmp) then true; \
	   else echo "Possible problem with ex19_12, diffs above"; fi; \
	   ${RM} -f ex19_12.tmp
runex19_13: #test pc_fieldsplit with -snes_mf_operator
	-@${MPIEXEC} -n 3 ./ex19 -snes_monitor_short -ksp_monitor_short -pc_type fieldsplit -pc_fieldsplit_type multiplicative -snes_view  -preload off -snes_mf_operator > ex19_13.tmp 2>&1; \
	   if (${DIFF} output/ex19_13.out ex19_13.tmp) then true; \
	   else echo "Possible problem with ex19_13, diffs above"; fi; \
	   ${RM} -f ex19_13.tmp


testex19: ex19.PETSc
	-@if [ "${PETSC_WITH_BATCH}" != "" ]; then \
           echo "Running with batch filesystem; to test run src/snes/examples/tutorials/ex19 with" ; \
           echo "your systems batch system"; \
	elif [ -f ex19 ]; then \
           ${MPIEXEC} -n 1 ./ex19 -dmmg_nlevels 4  > ex19_1.tmp 2>&1; \
	   if (${DIFF} output/ex19_1.testout ex19_1.tmp > /dev/null 2>&1) then \
           echo "C/C++ example src/snes/examples/tutorials/ex19 run successfully with 1 MPI process"; \
	   else echo "Possible error running C/C++ src/snes/examples/tutorials/ex19 with 1 MPI process"; \
           echo "See http://www.mcs.anl.gov/petsc/petsc-as/documentation/troubleshooting.html";\
           cat ex19_1.tmp; fi; \
	if [ "${MPIEXEC}" != "${PETSC_DIR}/bin/mpiexec.uni" ]; then \
           ${MPIEXEC} -n 2 ./ex19 -dmmg_nlevels 4   > ex19_1.tmp 2>&1; \
	   if (${DIFF} output/ex19_1.testout ex19_1.tmp > /dev/null 2>&1) then \
           echo "C/C++ example src/snes/examples/tutorials/ex19 run successfully with 2 MPI processes"; \
	   else echo "Possible error running C/C++ src/snes/examples/tutorials/ex19 with 2 MPI processes"; \
           echo "See http://www.mcs.anl.gov/petsc/petsc-as/documentation/troubleshooting.html";\
           cat ex19_1.tmp; fi; fi; \
        ${RM} -f ex19_1.tmp; \
        ${MAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} ex19.rm ; fi

testx11ex19: ex19.PETSc
	-@if [ "${PETSC_WITH_BATCH}" != "" ]; then \
           echo "Running with batch filesystem; to test run src/snes/examples/tutorials/ex19 with" ; \
           echo "your systems batch system"; \
	elif [ -f ex19 ]; then \
          D2=`grep "#define PETSC_HAVE_X11 1" ${PETSC_DIR}/${PETSC_ARCH}/include/petscconf.h`; \
          if [ "$$?" = 0 ]; then \
             ${MPIEXEC} -n 1 ./ex19 -dmmg_nlevels 4 -snes_monitor_draw  > ex19_1.tmp 2>&1; \
	     if (${DIFF} output/ex19_1.testout ex19_1.tmp > /dev/null 2>&1) then \
             echo "Graphics example src/snes/examples/tutorials/ex19 run successfully with 1 MPI process"; \
	     else echo "Possible error running Graphics examples src/snes/examples/tutorials/ex19 1 MPI process"; \
             echo "See http://www.mcs.anl.gov/petsc/petsc-as/documentation/troubleshooting.html";\
             cat ex19_1.tmp; fi; \
	  else echo "PETSc installed without X11. Not running src/snes/examples/tutorials/ex19"; fi; \
        ${RM} -f ex19_1.tmp; \
        ${MAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} ex19.rm ; fi
runex19_14:
	-@${MPIEXEC} -n 2 ./ex19 -dmmg_nlevels 4 -snes_monitor_short -dmmg_mat_type baij -mg_coarse_pc_type bjacobi > ex19_14.tmp 2>&1; \
	   if (${DIFF} output/ex19_14.out ex19_14.tmp) then true; \
	   else echo "Possible problem with ex19_14, diffs above"; fi; \
	   ${RM} -f ex19_14.tmp
runex19_15:
	-@${MPIEXEC} -n 2 ./ex19 -dmmg_nlevels 4 -snes_monitor_short -dmmg_mat_type baij -mg_coarse_pc_type bjacobi -dmmg_iscoloring_type global > ex19_15.tmp 2>&1; \
	   if (${DIFF} output/ex19_15.out ex19_15.tmp) then true; \
	   else echo "Possible problem with ex19_15, diffs above"; fi; \
	   ${RM} -f ex19_15.tmp
runex19_15a:
	-@${MPIEXEC} -n 2 ./ex19 -dmmg_nlevels 4 -snes_monitor_short -dmmg_mat_type baij -mg_coarse_pc_type bjacobi -dmmg_iscoloring_type global -dmmg_coloring_from_mat > ex19_15a.tmp 2>&1; \
	   if (${DIFF} output/ex19_15a.out ex19_15a.tmp) then true; \
	   else echo "Possible problem with ex19_15a, diffs above"; fi; \
	   ${RM} -f ex19_15a.tmp
runex19_16:
	-@${MPIEXEC} -n 2 ./ex19 -dmmg_nlevels 4 -snes_monitor_short -dmmg_mat_type aij -mg_coarse_pc_type bjacobi -dmmg_iscoloring_type global > ex19_16.tmp 2>&1; \
	   if (${DIFF} output/ex19_16.out ex19_16.tmp) then true; \
	   else echo "Possible problem with ex19_16, diffs above"; fi; \
	   ${RM} -f ex19_16.tmp
runex19_16a:
	-@${MPIEXEC} -n 2 ./ex19 -dmmg_nlevels 4 -snes_monitor_short -dmmg_mat_type aij -mg_coarse_pc_type bjacobi -dmmg_iscoloring_type global -dmmg_coloring_from_mat > ex19_16a.tmp 2>&1; \
	   if (${DIFF} output/ex19_16a.out ex19_16a.tmp) then true; \
	   else echo "Possible problem with ex19_16a, diffs above"; fi; \
	   ${RM} -f ex19_16a.tmp
runex21:
	-@${MPIEXEC} -n 4 ./ex21  -snes_mf -snes_monitor_short -nox > ex21_1.tmp 2>&1; \
	   if (${DIFF} output/ex21_1.out ex21_1.tmp) then true; \
	   else echo "Possible problem with ex21_1, diffs above"; fi; \
	   ${RM} -f ex21_1.tmp
runex22:
	-@${MPIEXEC} -n 1 ./ex22 -da_grid_x 10 > ex22_1.tmp 2>&1; \
	   if (${DIFF} output/ex22_1.out ex22_1.tmp) then true; \
	   else echo "Possible problem with ex22_1, diffs above"; fi; \
	   ${RM} -f ex22_1.tmp
runex23:
	-@${MPIEXEC} -n 2 ./ex23 -snes_monitor_short -dmmg_jacobian_mf_fd -mat_mffd_type wp -mat_mffd_compute_normu no > ex23_1.tmp 2>&1; \
	   if (${DIFF} output/ex23_1.out ex23_1.tmp) then true; \
	   else echo "Possible problem with ex23_1, diffs above"; fi; \
	   ${RM} -f ex23_1.tmp

runex25:
	-@${MPIEXEC} -n 1 ./ex25 -preload off > ex25_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex25_1.out ex25_1.tmp) then true; \
	   else echo "Possible problem with ex25_1, diffs above"; fi; \
	   ${RM} -f ex25_1.tmp

runex25_2:
	-@${MPIEXEC} -n 2 ./ex25 -preload off > ex25_2.tmp 2>&1;	  \
	   if (${DIFF} output/ex25_2.out ex25_2.tmp) then true; \
	   else echo "Possible problem with ex25_2, diffs above"; fi; \
	   ${RM} -f ex25_2.tmp
runex27:
	-@${MPIEXEC} -n 1 ./ex27 -lidvelocity 100 -da_grid_x 16 -da_grid_y 16 -print -snes_monitor -grashof 1.e5 -cfl_ini 1.0e-1 -snes_rtol 1.0e-3
runex27_p:
	-@${MPIEXEC} -n 1 ./ex27 -lidvelocity 100 -da_grid_x 16 -da_grid_y 16 -print -snes_monitor -grashof 1.e5 -cfl_ini 1.0e-1 -snes_rtol 1.0e-3 -use_parabolic

runex32:
	-@${MPIEXEC} -n 1 ./ex32 -snes_monitor -dmmg_jacobian_mf_fd_operator -da_grid_x 10 -da_grid_y 10
runex32_2:
	-@${MPIEXEC} -n 1 ./ex32 -snes_monitor -dmmg_jacobian_mf_fd -da_grid_x 10 -da_grid_y 10
runex32_3:
	-@${MPIEXEC} -n 3 ./ex32 -snes_monitor -dmmg_jacobian_mf_fd_operator -da_grid_x 10 -da_grid_y 10
runex32_4:
	-@${MPIEXEC} -n 3 ./ex32 -snes_monitor -dmmg_jacobian_mf_fd -da_grid_x 10 -da_grid_y 10

runex42:
	-@${MPIEXEC} -n 1 ./ex42 -snes_monitor_short -snes_max_it 1000 -snes_rtol 1.e-14  > ex42_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex42_1.out ex42_1.tmp) then true; \
	   else echo "Possible problem with ex42_1, diffs above"; fi; \
	   ${RM} -f ex42_1.tmp

runex44:
	-@${MPIEXEC} -n 1 ./ex44 -n 10 -snes_mf -user_precond > ex44.tmp 2>&1; \
	   if (${DIFF} output/ex44.out ex44.tmp) then true; \
	   else echo "Possible problem with ex44, diffs above"; fi; \
	   ${RM} -f ex44.tmp
runex44_2:
	-@${MPIEXEC} -n 1 ./ex44 -n 10 -snes_mf > ex44.tmp 2>&1; \
	   if (${DIFF} output/ex44.out ex44.tmp) then true; \
	   else echo "Possible problem with ex44_2, diffs above"; fi; \
	   ${RM} -f ex44.tmp

runex45:
	-@${MPIEXEC} -n 1 ./ex45 -n 10 > ex45.tmp 2>&1; \
	   if (${DIFF} output/ex44.out ex45.tmp) then true; \
	   else echo "Possible problem with ex45, diffs above"; fi; \
	   ${RM} -f ex45.tmp
runex45_2:
	-@${MPIEXEC} -n 1 ./ex45 -n 10 -snes_jacobian_default -pc_type jacobi > ex45.tmp 2>&1; \
	   if (${DIFF} output/ex44.out ex45.tmp) then true; \
	   else echo "Possible problem with ex45_2, diffs above"; fi; \
	   ${RM} -f ex45.tmp
runex45_3:
	-@${MPIEXEC} -n 1 ./ex45 -n 10 -snes_jacobian_default -fd_jacobian_coloring -my_jacobian_struct > ex45.tmp 2>&1; \
	   if (${DIFF} output/ex44.out ex45.tmp) then true; \
	   else echo "Possible problem with ex45_3, diffs above"; fi; \
	   ${RM} -f ex45.tmp

runex48:
	-@${MPIEXEC} -n 1 ./ex48 -M 6 -P 4 -thi_nlevels 2 -snes_monitor_short -snes_converged_reason -ksp_monitor_short -ksp_converged_reason -thi_mat_type sbaij > ex48.tmp 2>&1; \
	   ${DIFF} output/ex48_1.out ex48.tmp || echo "Possible problem with ex48_1, diffs above"; \
	   ${RM} -f ex48.tmp
runex48_2:
	-@${MPIEXEC} -n 2 ./ex48 -M 6 -P 4 -thi_hom z -thi_nlevels 2 -snes_monitor_short -snes_converged_reason -ksp_monitor_short -ksp_converged_reason -thi_mat_type sbaij -mg_levels_pc_type asm -mg_levels_pc_asm_blocks 6 -mg_levels_0_pc_type redundant -dmmg_grid_sequence -thi_mat_partitioning_type current > ex48.tmp 2>&1; \
	   ${DIFF} output/ex48_2.out ex48.tmp || echo "Possible problem with ex48_2, diffs above"; \
	   ${RM} -f ex48.tmp
runex48_3:
	-@${MPIEXEC} -n 3 ./ex48 -M 7 -P 4 -thi_hom z -thi_nlevels 2 -snes_monitor_short -snes_converged_reason -ksp_monitor_short -ksp_converged_reason -thi_mat_type baij -mg_levels_pc_type asm -mg_levels_pc_asm_blocks 9 -thi_mat_partitioning_type current > ex48.tmp 2>&1; \
	   ${DIFF} output/ex48_3.out ex48.tmp || echo "Possible problem with ex48_3, diffs above"; \
	   ${RM} -f ex48.tmp


TESTEXAMPLES_C		       = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2  runex2_3 ex2.rm ex3.PETSc runex3 \
                                 runex3_2 runex3_3 runex3_4 ex3.rm ex5.PETSc runex5 runex5_2 runex5_3 runex5_4 \
                                 runex5_5 runex5_6 runex5_7 runex5_9 runex5_13 runex5_14 ex5.rm \
                                 ex6.PETSc runex6 runex6_2 ex6.rm\
                                 ex14.PETSc runex14 runex14_2 ex14.rm ex19.PETSc runex19 runex19_2 runex19_5 runex19_6 runex19_7 runex19_8 runex19_9 \
                                 runex19_10 runex19_14 runex19_15 runex19_15a runex19_16 runex19_16a ex19.rm \
                                 ex22.PETSc runex22 ex22.rm ex18.PETSc ex18.rm ex19.PETSc ex19.rm ex21.PETSc runex21 ex21.rm \
                                 ex23.PETSc runex23 ex23.rm ex24.PETSc ex24.rm   ex25.PETSc runex25 runex25_2 \
                                 ex25.rm ex29.PETSc ex29.rm ex31.PETSc ex31.rm ex42.PETSc runex42 ex42.rm ex43.PETSc ex43.rm \
                                 ex44.PETSc runex44 runex44_2 ex44.rm ex45.PETSc runex45 runex45_2 runex45_3 ex45.rm ex47.PETSc ex47.rm \
                                 ex48.PETSc runex48 runex48_2 runex48_3 ex48.rm
TESTEXAMPLES_C_X11	       = ex5.PETSc runex5_8 runex5_10 ex5.rm 
TESTEXAMPLES_FORTRAN	       = ex1f.PETSc runex1f ex1f.rm
TESTEXAMPLES_C_NOCOMPLEX       = ex26.PETSc ex26.rm ex30.PETSc ex30.rm
TESTEXAMPLES_FORTRAN_NOCOMPLEX = ex5f.PETSc runex5f runex5f_3 ex5f.rm
TESTEXAMPLES_FORTRAN_MPIUNI    = ex1f.PETSc runex1f ex1f.rm
TESTEXAMPLES_C_X11_MPIUNI      = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm ex3.PETSc runex3 ex3.rm ex6.PETSc \
                                 runex6 runex6_2 ex6.rm ex5.PETSc ex5.rm ex14.PETSc ex14.rm 
TESTEXAMPLES_F90	       = ex5f90.PETSc runex5f90 runex5f90_2 runex5f90_3 runex5f90_4 runex5f90_5 ex5f90.rm
TESTEXAMPLES_13		       = 
TESTEXAMPLES_MATLAB_ENGINE     = ex5.PETSc runex5_11 runex5_13 ex5.rm
TESTEXAMPLES_AMS	       = 
TESTEXAMPLES_ADIFOR	       = ex5f.PETSc runex5f_2 ex5f.rm
TESTEXAMPLES_MUMPS             = ex19.PETSc runex19_3 runex19_4 ex19.rm 
TESTEXAMPLES_PASTIX            = ex19.PETSc runex19_11 runex19_12 ex19.rm 

include ${PETSC_DIR}/conf/test
