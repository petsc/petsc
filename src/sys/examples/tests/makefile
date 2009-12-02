
CFLAGS	        = 
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/sys/examples/tests/
EXAMPLESC       = ex1.c ex2.c ex3.c ex9.c ex10.c ex11.c ex12.c \
		  ex14.c ex15.c ex16.c ex18.c ex19.c
EXAMPLESF       = ex1f.F ex5f.F ex6f.F  ex17f.F
MANSEC          = Sys

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_SYS_LIB}
	${RM} -f ex1.o

ex1f: ex1f.o chkopts
	-${FLINKER} -o ex1f ex1f.o  ${PETSC_SYS_LIB}
	${RM} -f ex1f.o

ex2: ex2.o chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_SYS_LIB}
	${RM} -f ex2.o

ex3: ex3.o chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_SYS_LIB}
	${RM} -f ex3.o

ex4: ex4.o chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_SYS_LIB}
	${RM} -f ex4.o

ex5f: ex5f.o chkopts
	-${FLINKER} -o ex5f ex5f.o  ${PETSC_SYS_LIB}
	${RM} -f ex5f.o

ex6f: ex6f.o chkopts
	-${FLINKER} -o ex6f ex6f.o  ${PETSC_SYS_LIB}
	${RM} -f ex6f.o

ex9: ex9.o chkopts
	-${CLINKER} -o ex9 ex9.o  ${PETSC_SYS_LIB}
	${RM} -f ex9.o

ex10: ex10.o chkopts
	-${CLINKER} -o ex10 ex10.o  ${PETSC_SYS_LIB}
	${RM} -f ex10.o

ex11: ex11.o chkopts
	-${CLINKER} -o ex11 ex11.o  ${PETSC_SYS_LIB}
	${RM} -f ex11.o

ex12: ex12.o chkopts
	-${CLINKER} -o ex12 ex12.o  ${PETSC_SYS_LIB}
	${RM} -f ex12.o

ex14: ex14.o chkopts
	-${CLINKER} -o ex14 ex14.o  ${PETSC_SYS_LIB}
	${RM} -f ex14.o
ex15: ex15.o chkopts
	-${CLINKER} -o ex15 ex15.o  ${PETSC_SYS_LIB}
	${RM} -f ex15.o
ex16: ex16.o chkopts
	-${CLINKER} -o ex16 ex16.o  ${PETSC_SYS_LIB}
	${RM} -f ex16.o
ex17f: ex17f.o chkopts
	-${FLINKER} -o ex17f ex17f.o  ${PETSC_SYS_LIB}
	${RM} -f ex17f.o
ex18: ex18.o chkopts
	-${CLINKER} -o ex18 ex18.o  ${PETSC_SYS_LIB}
	${RM} -f ex18.o
ex19: ex19.o chkopts
	-${CLINKER} -o ex19 ex19.o  ${PETSC_SYS_LIB}
	${RM} -f ex19.o
#----------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1
	-@echo "This is a contrived error to test error handling."

runex1f:
	-@${MPIEXEC} -n 1 ./ex1f
	-@echo "This is a contrived error to test error handling."

runex2:
	-@${MPIEXEC} -n 1 ./ex2
	-@echo "This is a contrived error to test error handling."

runex3:
	-@${MPIEXEC} -n 1 ./ex3 -fp_trap
	-@echo "This is a contrived error to test error handling of floating point exceptions."

runex10:
	-@${MPIEXEC} -n 1 ./ex10 > ex10_1.tmp 2>&1;\
	   if (${DIFF} output/ex10_1.out ex10_1.tmp) then true; \
	   else echo "Error in PetscMemmove."; fi; \
	   ${RM} -f ex10_1.tmp ex10_1.tmp

runex19:
	-@${MPIEXEC} -n 1 ./ex19 > ex19_1.tmp 2>&1;   \
	   if (${DIFF} output/ex19_1.out ex19_1.tmp) then true; \
	   else echo "Possible problem with ex19_1, diffs above"; fi; \
	   ${RM} -f ex19_1.tmp

TESTEXAMPLES_C		       = ex19.PETSc runex19 ex19.rm
TESTEXAMPLES_C_X11	       = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm ex3.PETSc runex3 ex3.rm 
TESTEXAMPLES_FORTRAN	       = ex5f.PETSc ex5f.rm ex6f.PETSc ex6f.rm ex17f.PETSc ex17f.rm
TESTEXAMPLES_FORTRAN_NOCOMPLEX = ex1f.PETSc runex1f ex1f.rm
TESTEXAMPLES_FORTRAN_COMPLEX   = ex1f.PETSc runex1f ex1f.rm
TESTEXAMPLES_FORTRAN_MPIUNI    = ex5f.PETSc ex5f.rm ex6f.PETSc ex6f.rm
TESTEXAMPLES_C_X11_MPIUNI      = ex1.PETSc ex1.rm ex2.PETSc ex2.rm ex3.PETSc ex3.rm
TESTEXAMPLES_13		       = ex9.PETSc ex9.rm ex10.PETSc ex10.rm ex11.PETSc ex11.rm ex12.PETSc ex12.rm \
                                 ex14.PETSc ex14.rm

include ${PETSC_DIR}/conf/test
