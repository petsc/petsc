
CFLAGS	        =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ksp/pc/examples/tests/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c
EXAMPLESF       = ex8f.F
MANSEC          = PC

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o   chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_KSP_LIB}
	${RM} ex1.o

ex2: ex2.o   chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_KSP_LIB}
	${RM} ex2.o

ex3: ex3.o   chkopts
	-${CLINKER} -o ex3 ex3.o ${PETSC_KSP_LIB}
	${RM} ex3.o

ex4: ex4.o   chkopts
	-${CLINKER} -o ex4 ex4.o ${PETSC_KSP_LIB}
	${RM} ex4.o

ex5: ex5.o   chkopts
	-${CLINKER} -o ex5 ex5.o ${PETSC_KSP_LIB}
	${RM} ex5.o

ex6: ex6.o   chkopts
	-${CLINKER} -o ex6 ex6.o ${PETSC_KSP_LIB}
	${RM} ex6.o

ex7: ex7.o   chkopts
	-${CLINKER} -o ex7 ex7.o ${PETSC_KSP_LIB}
	${RM} ex7.o

ex8f: ex8f.o   chkopts
	-${FLINKER} -o ex8f ex8f.o ${PETSC_KSP_LIB}
	${RM} ex8f.o

#------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1 > ex1_1.tmp 2>&1;   \
	   if (${DIFF} output/ex1_1.out ex1_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex1_1, diffs above \n========================================="; fi; \
	   ${RM} -f ex1_1.tmp 
runex2:
	-@${MPIEXEC} -n 1 ./ex2 -ksp_type cg -ksp_monitor_short > ex2_1.tmp 2>&1;	 \
	   if (${DIFF} output/ex2_1.out ex2_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex2_1, diffs above \n========================================="; fi; \
	   ${RM} -f ex2_1.tmp 
runex3:
	-@${MPIEXEC} -n 1 ./ex3 -ksp_type gmres -ksp_monitor_short -pc_type sor\
			     -pc_sor_symmetric > ex3_1.tmp 2>&1;   \
	   if (${DIFF} output/ex3_1.out ex3_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex3_1, diffs above \n========================================="; fi; \
	   ${RM} -f ex3_1.tmp 
runex4:
	-@${MPIEXEC} -n 1 ./ex4	 > ex4_1.tmp 2>&1;   \
	   if (${DIFF} output/ex4_1.out ex4_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex4_1, diffs above \n========================================="; fi; \
	   ${RM} -f ex4_1.tmp 

runex5:
	-@${MPIEXEC} -n 1 ./ex5 > ex5_1.tmp 2>&1;   \
	   if (${DIFF} output/ex5_1.out ex5_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex5_1, diffs above \n========================================="; fi; \
	   ${RM} -f ex5_1.tmp 

runex6:
	-@${MPIEXEC} -n 1  ./ex6 -m 7 > ex6_1.tmp   2>&1;   \
	   if (${DIFF} output/ex6_1.out ex6_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex6_1, diffs above \n========================================="; fi; \
	   ${RM} -f ex6_1.tmp

runex8f:
	-@${MPIEXEC} -n 1  ./ex8f 

TESTEXAMPLES_C		  = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm ex3.PETSc runex3 ex3.rm \
                            ex4.PETSc runex4 ex4.rm 
TESTEXAMPLES_C_X	  = ex5.PETSc runex5 ex5.rm
TESTEXAMPLES_FORTRAN	  = ex8f.PETSc runex8f ex8f.rm
TESTEXAMPLES_C_X_MPIUNI = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm ex3.PETSc runex3 ex3.rm \
                            ex4.PETSc runex4 ex4.rm ex5.PETSc runex5 ex5.rm

include ${PETSC_DIR}/conf/test


