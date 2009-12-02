
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tutorials/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c util2.c ex7.c ex8.c ex9.c
EXAMPLESF       = ex1f.F ex2f.F
EXAMPLESFH      = ex2f.h
MANSEC          = TS

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_TS_LIB}
	${RM} ex1.o

ex1f: ex1f.o  chkopts
	-${FLINKER} -o ex1f ex1f.o  ${PETSC_TS_LIB}
	${RM} ex1f.o

ex2: ex2.o util2.o chkopts
	-${CLINKER} -o ex2 ex2.o util2.o ${PETSC_TS_LIB}
	${RM} ex2.o util2.o

ex2f: ex2f.o util2.o chkopts
	-${FLINKER} -o ex2f ex2f.o util2.o  ${PETSC_TS_LIB}
	${RM} ex2f.o util2.o

ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_TS_LIB}
	${RM} ex3.o

ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_TS_LIB}
	${RM} ex4.o

ex5: ex5.o  chkopts
	-${CLINKER} -o ex5 ex5.o  ${PETSC_TS_LIB}
	${RM} ex5.o

ex6: ex6.o  chkopts
	-${CLINKER} -o ex6 ex6.o  ${PETSC_TS_LIB}
	${RM} ex6.o

ex7: ex7.o  chkopts
	-${CLINKER} -o ex7 ex7.o  ${PETSC_TS_LIB}
	${RM} ex7.o

ex8: ex8.o  chkopts
	-${CLINKER} -o ex8 ex8.o  ${PETSC_TS_LIB}
	${RM} ex8.o

ex9: ex9.o  chkopts
	-${CLINKER} -o ex9 ex9.o  ${PETSC_TS_LIB}
	${RM} ex9.o

#---------------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1 -ksp_gmres_cgs_refinement_type refine_always  > ex1_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   ${RM} -f ex1_1.tmp

runex1f:
	-@${MPIEXEC} -n 1 ./ex1f -ksp_gmres_cgs_refinement_type refine_always > ex1f_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex1f_1.out ex1f_1.tmp) then true; \
	   else echo "Possible problem with ex1f_1, diffs above"; fi; \
	   ${RM} -f ex1f_1.tmp

runex2f:
	-@${MPIEXEC} -n 1 ./ex2f 

runex4:
	-@${MPIEXEC} -n 1 ./ex4 -nox > ex4_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex4_1.out ex4_1.tmp) then true; \
	   else echo "Possible problem with ex4_1, diffs above"; fi; \
	   ${RM} -f ex4_1.tmp

runex4_2:
	-@${MPIEXEC} -n 3 ./ex4 -nox > ex4_2.tmp 2>&1;	  \
	   if (${DIFF} output/ex4_2.out ex4_2.tmp) then true; \
	   else echo "Possible problem with ex4_2, diffs above"; fi; \
	   ${RM} -f ex4_2.tmp

runex4_3:
	-@${MPIEXEC} -n 1 ./ex4 -nox -nonlinear > ex4_3.tmp 2>&1;	  \
	   if (${DIFF} output/ex4_3.out ex4_3.tmp) then true; \
	   else echo "Possible problem with ex4_3, diffs above"; fi; \
	   ${RM} -f ex4_3.tmp

runex4_4:
	-@${MPIEXEC} -n 3 ./ex4 -nox -nonlinear > ex4_4.tmp 2>&1;	  \
	   if (${DIFF} output/ex4_4.out ex4_4.tmp) then true; \
	   else echo "Possible problem with ex4_4, diffs above"; fi; \
	   ${RM} -f ex4_4.tmp

runex6:
	-@${MPIEXEC} -n 1 ./ex6 -nox -time_steps_max 4   > ex6_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex6_1.out ex6_1.tmp) then true; \
	   else echo "Possible problem with ex6_1, diffs above"; fi; \
	   ${RM} -f ex6_1.tmp

runex7:
	-@${MPIEXEC} -n 1 ./ex7 -max_steps 5   > ex7_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex7_1.out ex7_1.tmp) then true; \
	   else echo "Possible problem with ex7_1, diffs above"; fi; \
	   ${RM} -f ex7_1.tmp

runex7_2:
	-@${MPIEXEC} -n 1 ./ex7 -max_steps 5  -snes_mf_operator > ex7_2.tmp 2>&1;	  \
	   if (${DIFF} output/ex7_2.out ex7_2.tmp) then true; \
	   else echo "Possible problem with ex7_2, diffs above"; fi; \
	   ${RM} -f ex7_2.tmp

runex7_3:
	-@${MPIEXEC} -n 1 ./ex7 -max_steps 5  -snes_mf > ex7_3.tmp 2>&1;	  \
	   if (${DIFF} output/ex7_3.out ex7_3.tmp) then true; \
	   else echo "Possible problem with ex7_3, diffs above"; fi; \
	   ${RM} -f ex7_3.tmp


TESTEXAMPLES_C		  = ex1.PETSc runex1 ex1.rm ex4.PETSc runex4 runex4_2 runex4_3 runex4_4 ex4.rm\
                            ex6.PETSc runex6 ex6.rm ex7.PETSc runex7 runex7_2 runex7_3 ex7.rm
TESTEXAMPLES_C_X11	  = 
TESTEXAMPLES_FORTRAN	  = ex1f.PETSc runex1f ex1f.rm ex2f.PETSc runex2f ex2f.rm
TESTEXAMPLES_C_X11_MPIUNI = 
TESTEXAMPLES_13		  = ex2.PETSc ex2.rm ex3.PETSc ex3.rm ex4.PETSc ex4.rm \
                            ex5.PETSc ex5.rm

include ${PETSC_DIR}/conf/test
