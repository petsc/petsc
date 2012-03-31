
CFLAGS	         =
FFLAGS	         =
CPPFLAGS         =
FPPFLAGS         = 
LOCDIR           = src/vec/is/examples/tutorials/
EXAMPLESC        = ex1.c ex2.c ex3.c ex4.c
EXAMPLESF        = ex1f.F ex1f90.F ex2f.F ex3f90.F

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_VEC_LIB}
	${RM} -f ex1.o

ex1f: ex1f.o chkopts
	-${FLINKER} -o ex1f ex1f.o ${PETSC_VEC_LIB}
	${RM} -f ex1f.o

ex1f90: ex1f90.o chkopts
	-${FLINKER} -o ex1f90 ex1f90.o ${PETSC_VEC_LIB}
	${RM} -f ex1f90.o

ex2: ex2.o chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_VEC_LIB}
	${RM} -f ex2.o

ex2f: ex2f.o chkopts
	-${FLINKER} -o ex2f ex2f.o ${PETSC_VEC_LIB}
	${RM} -f ex2f.o

ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_VEC_LIB}
	${RM} -f ex3.o

ex3f90: ex3f90.o chkopts
	-${FLINKER} -o ex3f90 ex3f90.o ${PETSC_VEC_LIB}
	${RM} -f ex3f90.o

ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_VEC_LIB}
	${RM} -f ex4.o
#------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1  ./ex1  > ex1.tmp 2>&1;\
	   if (${DIFF} output/ex1.out ex1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex1, diffs above \n========================================="; fi ;\
	   ${RM} -f ex1.tmp

runex1f:
	-@${MPIEXEC} -n 1  ./ex1f  | sort > ex1f.tmp 2>&1;\
	   if (${DIFF} output/ex1f.out ex1f.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex1f, diffs above \n========================================="; fi ;\
	   ${RM} -f ex1f.tmp

runex1f90:
	-@${MPIEXEC} -n 1  ./ex1f90  | sort > ex1f90.tmp 2>&1;\
	   if (${DIFF} output/ex1f90.out ex1f90.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex1f90, diffs above \n========================================="; fi ;\
	   ${RM} -f ex1f90.tmp

runex2:
	-@${MPIEXEC} -n 1  ./ex2  > ex2.tmp 2>&1;\
	   if (${DIFF} output/ex2.out ex2.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex2, diffs above \n========================================="; fi ;\
	   ${RM} -f ex2.tmp

runex2f:
	-@${MPIEXEC} -n 1  ./ex2f  > ex2f.tmp 2>&1;\
	   if (${DIFF} output/ex2f.out ex2f.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex2f, diffs above \n========================================="; fi ;\
	   ${RM} -f ex2f.tmp

runex3:
	-@${MPIEXEC} -n 1  ./ex3  > ex3.tmp 2>&1;\
	   if (${DIFF} output/ex3.out ex3.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex3, diffs above \n========================================="; fi ;\
	   ${RM} -f ex3.tmp

runex3f90:
	-@${MPIEXEC} -n 1  ./ex3f90 |sort > ex3f90.tmp 2>&1;\
	   if (${DIFF} output/ex3f90.out ex3f90.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex3f90, diffs above \n========================================="; fi ;\
	   ${RM} -f ex3f90.tmp

TESTEXAMPLES_C		    = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm ex3.PETSc runex3 ex3.rm
TESTEXAMPLES_C_X	    = 
TESTEXAMPLES_FORTRAN	    = ex1f.PETSc runex1f ex1f.rm ex2f.PETSc runex2f ex2f.rm
TESTEXAMPLES_FORTRAN_MPIUNI = 
TESTEXAMPLES_C_X_MPIUNI   = 
TESTEXAMPLES_F90	    = ex1f90.PETSc runex1f90 ex1f90.rm ex3f90.PETSc runex3f90 ex3f90.rm
TESTEXAMPLES_13		    = ex4.PETSc ex4.rm

include ${PETSC_DIR}/conf/test
