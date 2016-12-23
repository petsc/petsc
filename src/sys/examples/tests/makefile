
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/sys/examples/tests/
EXAMPLESC       = ex1.c ex2.c ex3.c ex7.c ex8.c ex9.c ex10.c ex11.c ex12.c \
                ex14.c ex15.c ex16.c ex18.c ex19.c ex20.c ex21.c \
                ex22.c ex23.c ex24.c ex27.c ex28.c ex29.c ex30.c ex31.c ex32.c
EXAMPLESF       = ex1f.F ex5f.F ex6f.F ex17f.F
MANSEC          = Sys

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

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

ex8: ex8.o chkopts
	-${CLINKER} -o ex8 ex8.o  ${PETSC_SYS_LIB}
	${RM} -f ex8.o

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

ex13: ex13.o chkopts
	-${CLINKER} -o ex13 ex13.o  ${PETSC_SYS_LIB}
	${RM} -f ex13.o

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
ex25: ex25.o chkopts
	-${CLINKER} -o ex25 ex25.o  ${PETSC_SYS_LIB}
	${RM} -f ex25.o
ex27: ex27.o chkopts
	-${CLINKER} -o ex27 ex27.o  ${PETSC_SYS_LIB}
	${RM} -f ex27.o
ex28: ex28.o chkopts
	-${CLINKER} -o ex28 ex28.o  ${PETSC_SYS_LIB}
	${RM} -f ex28.o
ex29: ex29.o chkopts
	-${CLINKER} -o ex29 ex29.o  ${PETSC_SYS_LIB}
	${RM} -f ex29.o
ex30: ex30.o chkopts
	-${CLINKER} -o ex30 ex30.o  ${PETSC_SYS_LIB}
	${RM} -f ex30.o
ex31: ex31.o chkopts
	-${CLINKER} -o ex31 ex31.o  ${PETSC_SYS_LIB}
	${RM} -f ex31.o
ex32: ex32.o chkopts
	-${CLINKER} -o ex32 ex32.o  ${PETSC_SYS_LIB}
	${RM} -f ex32.o
#----------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1 > ex1.tmp1 2>&1; egrep "(PETSC ERROR)" ex1.tmp1 | egrep "(main|CreateError|Error Created)" | cut -f1,2,3,4,5 -d" " > ex1.tmp;\
	if (${DIFF} output/ex1_1.out ex1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex1_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex1.tmp1 ex1.tmp

# cannot have this test in nightly builds because some computer systems can detect the error appropriate but not others
runex1f:
	-@${MPIEXEC} -n 1 ./ex1f  > ex1f.tmp1 2>&1; egrep "(PETSC ERROR)" ex1f.tmp1  > ex1f.tmp;\
	if (${DIFF} output/ex1f_1.out ex1f.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex1f_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex1f.tmp1 ex1f.tmp

runex2:
	-@x="bad"; ${MPIEXEC} -n 1 ./ex2 > ex2.tmp1 2>&1; egrep "(Caught signal number 11 SEGV|Caught signal number 4 Illegal)" ex2.tmp1  > ex2.tmp;\
           if (${DIFF} output/ex2_1.out ex2.tmp > /dev/null 2>&1) then x='good'; fi ;\
           if (${DIFF} output/ex2_1_alt.out ex2.tmp > /dev/null 2>&1) then x='good'; fi; \
           if [ "$$x" = "bad" ]; then ${DIFF} output/ex2_1.out ex2.tmp ; ${DIFF} output/ex2_1_alt.out ex2.tmp ; printf "${PWD}\nPossible problem with ex2_1, diffs above\n=============\
============================\n"; fi; \
           ${RM} -f ex2.tmp1 ex2.tmp

#  Since this test produces very different information on different machines there is no way to automate
#  the checking of the output to see if it is correct; hence this is not in the nightly tests, though it would be nice if we could add it
runex3:
	-@${MPIEXEC} -n 1 ./ex3 -fp_trap 

runex6f:
	-@${MPIEXEC} -n 1 ./ex6f > ex6f.tmp 2>&1;\
	   if (${DIFF} output/ex6f_1.out ex6f.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex6f_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex6f.tmp

runex14:
	-@${MPIEXEC} -n 1 ./ex14 -array 1.0,-2-3i,4.5+6.2i,4.5,6.8+4i,i,-i,-1.2i -rarray 1,2,3 -a 1.5+2.1i > ex14.tmp 2>&1;\
	   ${DIFF} output/ex14.out ex14.tmp || printf "${PWD}\nPossible problem with ex14, diffs above\n=========================================\n"; \
	   ${RM} ex14.tmp

runex23:
	-@${MPIEXEC} -n 1 ./ex23 -options_file_yaml ex23options > ex23.tmp 2>&1;   \
	   ${DIFF} output/ex23.out ex23.tmp || printf "${PWD}\nPossible problem with ex23, diffs above\n=========================================\n"; \
	   ${RM} -f ex23.tmp

runex31:
	-@${MPIEXEC} -n 1 ./ex31 > ex31.tmp 2>&1;   \
           echo ~/somefile > ex31-sh.tmp  2>&1;   \
           echo ${PWD}/someotherfile >> ex31-sh.tmp  2>&1;   \
	   ${DIFF} ex31-sh.tmp ex31.tmp || echo  ${PWD} "\nPossible problem with ex31, diffs above \n========================================="; \
	   ${RM} -f ex31.tmp ex31-sh.tmp

TESTEXAMPLES_C		       = ex4.PETSc ex4.rm \
                                 ex21.PETSc ex21.rm \
                                 ex24.PETSc ex24.rm 

TESTEXAMPLES_C_COMPLEX         = ex14.PETSc runex14 ex14.rm

TESTEXAMPLES_C_X	       = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm ex3.PETSc ex3.rm
TESTEXAMPLES_FORTRAN	       = ex5f.PETSc ex5f.rm ex6f.PETSc ex6f.rm
TESTEXAMPLES_FORTRAN_NOCOMPLEX = ex1f.PETSc ex1f.rm
TESTEXAMPLES_FORTRAN_COMPLEX   = ex1f.PETSc ex1f.rm
TESTEXAMPLES_FORTRAN_MPIUNI    = ex5f.PETSc ex5f.rm ex6f.PETSc ex6f.rm
TESTEXAMPLES_C_X_MPIUNI      = ex1.PETSc ex1.rm ex2.PETSc ex2.rm ex3.PETSc ex3.rm ex4.PETSc ex4.rm
TESTEXAMPLES_13		       = ex11.PETSc ex11.rm ex12.PETSc ex12.rm \
                                 ex14.PETSc ex14.rm

include ${PETSC_DIR}/lib/petsc/conf/test
