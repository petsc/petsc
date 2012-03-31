
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        = 
LOCDIR          = src/dm/examples/tutorials/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c ex8.c ex9.c ex10.c ex12.c
EXAMPLESF       = ex6f90.F ex11f90.F
MANSEC          = DM
SUBMANSEC       = DMDA

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o   chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_DM_LIB}
	${RM} -f ex1.o

ex2: ex2.o   chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_DM_LIB}
	${RM} -f ex2.o

ex3: ex3.o   chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_DM_LIB}
	${RM} -f ex3.o

ex4: ex4.o   chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_DM_LIB}
	${RM} -f ex4.o

ex5: ex5.o   chkopts
	-${CLINKER} -o ex5 ex5.o  ${PETSC_DM_LIB}
	${RM} -f ex5.o

ex6: ex6.o   chkopts
	-${CLINKER} -o ex6 ex6.o  ${PETSC_DM_LIB}
	${RM} -f ex6.o

ex6f90: ex6f90.o   chkopts
	-${FLINKER} -o ex6f90 ex6f90.o  ${PETSC_DM_LIB}
	${RM} -f ex6f90.o

ex7: ex7.o   chkopts
	-${CLINKER} -o ex7 ex7.o  ${PETSC_DM_LIB}
	${RM} -f ex7.o

ex8: ex8.o   chkopts
	-${CLINKER} -o ex8 ex8.o  ${PETSC_DM_LIB}
	${RM} -f ex8.o

ex9: ex9.o   chkopts
	-${CLINKER} -o ex9 ex9.o  ${PETSC_DM_LIB}
	${RM} -f ex9.o

ex10: ex10.o   chkopts
	-${CLINKER} -o ex10 ex10.o  ${PETSC_DM_LIB}
	${RM} -f ex10.o

ex11f90: ex11f90.o   chkopts
	-${FLINKER} -o ex11f90 ex11f90.o  ${PETSC_DM_LIB}
	${RM} -f ex11f90.o

ex12: ex12.o   chkopts
	-${CLINKER} -o ex12 ex12.o  ${PETSC_DM_LIB}
	${RM} -f ex12.o
#-----------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 2 ./ex1 -nox
runex2:
	-@${MPIEXEC} -n 3 ./ex2
runex3:
	-@${MPIEXEC} -n 3 ./ex3 -mat_view  > ex3_1.tmp 2>&1;   \
	   if (${DIFF} output/ex3_1.out ex3_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex3_1, diffs above \n========================================="; fi; \
	   ${RM} -f ex3_1.tmp
runex3_2:
	-@${MPIEXEC} -n 3 ./ex3 -mat_view -dim 2 > ex3_2.tmp 2>&1;   \
	   if (${DIFF} output/ex3_2.out ex3_2.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex3_2, diffs above \n========================================="; fi; \
	   ${RM} -f ex3_2.tmp
runex3_3:
	-@${MPIEXEC} -n 3 ./ex3 -mat_view -dim 3 > ex3_3.tmp 2>&1;   \
	   if (${DIFF} output/ex3_3.out ex3_3.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex3_3, diffs above \n========================================="; fi; \
	   ${RM} -f ex3_3.tmp
runex10:
	-@${MPIEXEC} -n 4 ./ex10

runex11f90:
	-@${MPIEXEC} -n 1 ./ex11f90  > ex11f90.tmp 2>&1;   \
	   if (${DIFF} output/ex11f90.out ex11f90.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex11f90, diffs above \n========================================="; fi; \
	   ${RM} -f ex11f90.tmp

TESTEXAMPLES_C		  = ex3.PETSc runex3 runex3_2 runex3_3 ex3.rm ex4.PETSc ex4.rm ex12.PETSc ex12.rm
TESTEXAMPLES_C_X	  = ex1.PETSc runex1 ex1.rm ex5.PETSc ex5.rm  
TESTEXAMPLES_FORTRAN	  = 
TESTEXAMPLES_F90_NOCOMPLEX= ex6f90.PETSc ex6f90.rm ex11f90.PETSc ex11f90.rm
TESTEXAMPLES_C_X_MPIUNI = 
TESTEXAMPLES_C_COMPLEX	  = 
TESTEXAMPLES_C_NOCOMPLEX  = ex2.PETSc runex2 ex2.rm ex6.PETSc ex6.rm
TESTEXAMPLES_13		  = 
TESTEXAMPLES_HDF5         = ex10.PETSc runex10 ex10.rm

include ${PETSC_DIR}/conf/test
