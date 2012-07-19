
CFLAGS	        = 
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/sys/examples/tests/
EXAMPLESC       = ex1.c ex2.c ex3.c ex7.c ex9.c ex10.c ex11.c ex12.c \
                ex14.c ex15.c ex16.c ex18.c ex19.c ex20.c ex21.c \
                ex22.c ex23.c ex24.c
EXAMPLESF       = ex1f.F ex5f.F ex6f.F ex17f.F
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

ex7: ex7.o chkopts
	-${CLINKER} -o ex7 ex7.o  ${PETSC_SYS_LIB}
	${RM} -f ex7.o

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
ex20: ex20.o chkopts
	-${CLINKER} -o ex20 ex20.o  ${PETSC_SYS_LIB}
	${RM} -f ex20.o
ex21: ex21.o chkopts
	-${CLINKER} -o ex21 ex21.o  ${PETSC_SYS_LIB}
	${RM} -f ex21.o
ex22: ex22.o chkopts
	-${CLINKER} -o ex22 ex22.o  ${PETSC_SYS_LIB}
	${RM} -f ex22.o
ex23: ex23.o chkopts
	-${CLINKER} -o ex23 ex23.o  ${PETSC_SYS_LIB}
	${RM} -f ex23.o
ex24: ex24.o chkopts
	-${CLINKER} -o ex24 ex24.o  ${PETSC_SYS_LIB}
	${RM} -f ex24.o
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

runex4:
	-@${MPIEXEC} -n 1 ./ex4 > ex4_1.tmp 2>&1;   \
	   if (${DIFF} output/ex4_1.out ex4_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with ex4_1, diffs above \n========================================="; fi; \
	   ${RM} -f ex4_1.tmp

runex7:
	-@${MPIEXEC} -n 1 ./ex7 > ex7.tmp 2>&1;\
	   if (${DIFF} output/ex7.out ex7.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex7_1, diffs above \n========================================="; fi; \
	   ${RM} -f ex7.tmp ex7.tmp

runex10:
	-@${MPIEXEC} -n 1 ./ex10 > ex10_1.tmp 2>&1;\
	   if (${DIFF} output/ex10_1.out ex10_1.tmp) then true; \
	   else echo "Error in PetscMemmove."; fi; \
	   ${RM} -f ex10_1.tmp ex10_1.tmp

runex19:
	-@${MPIEXEC} -n 1 ./ex19 > ex19_1.tmp 2>&1;   \
	   if (${DIFF} output/ex19_1.out ex19_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex19_1, diffs above \n========================================="; fi; \
	   ${RM} -f ex19_1.tmp
runex20:
	-@${MPIEXEC} -n 1 ./ex20 > ex20_1.tmp 2>&1;   \
	   ${DIFF} output/ex20_1.out ex20_1.tmp || echo ${PWD} "\nPossible problem with ex20_1, diffs above \n========================================="; \
	   ${RM} -f ex20_1.tmp
runex20_2:
	-@${MPIEXEC} -n 1 ./ex20 -cl -prefix_push cl_ -zero 10 -prefix_push a_ -one 11 -prefix_push bb_ -two 12 -prefix_pop -three 13 -prefix_push cc_ -prefix_push ddd_ -four 14 -prefix_pop -prefix_pop -prefix_pop -five 15 -prefix_pop > ex20_2.tmp 2>&1;   \
	   ${DIFF} output/ex20_2.out ex20_2.tmp || echo  ${PWD} "\nPossible problem with ex20_2, diffs above \n========================================="; \
	   ${RM} -f ex20_2.tmp
runex20_3:
	-@${MPIEXEC} -n 1 ./ex20 -str > ex20_3.tmp 2>&1;   \
	   ${DIFF} output/ex20_3.out ex20_3.tmp || echo  ${PWD} "\nPossible problem with ex20_3, diffs above \n========================================="; \
	   ${RM} -f ex20_3.tmp
runex22:
	-@${MPIEXEC} -n 1 ./ex22 > ex22.tmp 2>&1;   \
	   ${DIFF} output/ex22.out ex22.tmp || echo  ${PWD} "\nPossible problem with ex22, diffs above \n========================================="; \
	   ${RM} -f ex22.tmp
runex23:
	-@${MPIEXEC} -n 1 ./ex23 -options_file_yaml ex23options > ex23.tmp 2>&1;   \
	   ${DIFF} output/ex23.out ex23.tmp || echo  ${PWD} "\nPossible problem with ex23, diffs above \n========================================="; \
	   ${RM} -f ex23.tmp


TESTEXAMPLES_C		       = ex4.PETSc runex4 ex4.rm ex19.PETSc runex19 ex19.rm \
                                 ex20.PETSc runex20 runex20_2 runex20_3 ex20.rm  ex21.PETSc ex21.rm \
                                 ex22.PETSc runex22 ex22.rm ex24.PETSc ex24.rm
TESTEXAMPLES_C_X	       = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm ex3.PETSc runex3 ex3.rm 
TESTEXAMPLES_FORTRAN	       = ex5f.PETSc ex5f.rm ex6f.PETSc ex6f.rm ex17f.PETSc ex17f.rm
TESTEXAMPLES_FORTRAN_NOCOMPLEX = ex1f.PETSc runex1f ex1f.rm
TESTEXAMPLES_FORTRAN_COMPLEX   = ex1f.PETSc runex1f ex1f.rm
TESTEXAMPLES_FORTRAN_MPIUNI    = ex5f.PETSc ex5f.rm ex6f.PETSc ex6f.rm
TESTEXAMPLES_C_X_MPIUNI      = ex1.PETSc ex1.rm ex2.PETSc ex2.rm ex3.PETSc ex3.rm
TESTEXAMPLES_13		       = ex9.PETSc ex9.rm ex10.PETSc ex10.rm ex11.PETSc ex11.rm ex12.PETSc ex12.rm \
                                 ex14.PETSc ex14.rm

include ${PETSC_DIR}/conf/test
