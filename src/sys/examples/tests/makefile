
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/sys/examples/tests/
EXAMPLESC       = ex1.c ex2.c ex3.c ex7.c ex8.c ex9.c ex10.c ex11.c ex12.c \
                ex14.c ex15.c ex16.c ex18.c ex19.c ex20.c ex21.c \
                ex22.c ex23.c ex24.c ex27.c ex28.c ex29.c
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

runex7:
	-@${MPIEXEC} -n 1 ./ex7 > ex7.tmp 2>&1;\
	   if (${DIFF} output/ex7.out ex7.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex7_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex7.tmp ex7.tmp

runex8:
	-@${MPIEXEC} -n 4 ./ex8 -verbose -build_twosided allreduce > ex8.tmp 2>&1;\
	   ${DIFF} output/ex8_1.out ex8.tmp || printf "${PWD}\nPossible problem with with ex8, diffs above\n=========================================\n"; \
	   ${RM} -f ex8.tmp

runex8_redscatter:
	-@${MPIEXEC} -n 4 ./ex8 -verbose -build_twosided redscatter > ex8.tmp 2>&1;\
	   ${DIFF} output/ex8_1.out ex8.tmp || printf "${PWD}\nPossible problem with with ex8_redscatter, diffs above\n=========================================\n"; \
	   ${RM} -f ex8.tmp

runex8_ibarrier:
	-@${MPIEXEC} -n 4 ./ex8 -verbose -build_twosided ibarrier > ex8.tmp 2>&1;\
	   ${DIFF} output/ex8_1.out ex8.tmp || printf "${PWD}\nPossible problem with with ex8_redscatter, diffs above\n=========================================\n"; \
	   ${RM} -f ex8.tmp

runex8_f:
	-@${MPIEXEC} -n 4 ./ex8 -verbose -build_twosided_f -build_twosided allreduce > ex8.tmp 2>&1;\
	   ${DIFF} output/ex8_1.out ex8.tmp || printf "${PWD}\nPossible problem with with ex8_f, diffs above\n=========================================\n"; \
	   ${RM} -f ex8.tmp

runex8_f_ibarrier:
	-@${MPIEXEC} -n 4 ./ex8 -verbose -build_twosided_f -build_twosided ibarrier > ex8.tmp 2>&1;\
	   ${DIFF} output/ex8_1.out ex8.tmp || printf "${PWD}\nPossible problem with with ex8_f, diffs above\n=========================================\n"; \
	   ${RM} -f ex8.tmp

runex10:
	-@${MPIEXEC} -n 1 ./ex10 > ex10_1.tmp 2>&1;\
	   if (${DIFF} output/ex10_1.out ex10_1.tmp) then true; \
	   else  printf "${PWD}\nPossible problem with ex10\n=========================================\n"; fi; \
	   ${RM} -f ex10_1.tmp ex10_1.tmp

runex13:
	-@${MPIEXEC} -n 1 ./ex13 > ex13_1.tmp 2>&1;\
	   ${DIFF} output/ex13_1.out ex13_1.tmp || printf "${PWD}\nPossible problem with ex13_1, diffs above\n=========================================\n"; \
	   ${RM} ex13_1.tmp

runex14:
	-@${MPIEXEC} -n 1 ./ex14 -array 1.0,-2-3i,4.5+6.2i,4.5,6.8+4i,i,-i,-1.2i -rarray 1,2,3 -a 1.5+2.1i > ex14.tmp 2>&1;\
	   ${DIFF} output/ex14.out ex14.tmp || printf "${PWD}\nPossible problem with ex14, diffs above\n=========================================\n"; \
	   ${RM} ex14.tmp

runex19:
	-@${MPIEXEC} -n 1 ./ex19 > ex19_1.tmp 2>&1;   \
	   if (${DIFF} output/ex19_1.out ex19_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex19_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex19_1.tmp
runex20:
	-@${MPIEXEC} -n 1 ./ex20 > ex20_1.tmp 2>&1;   \
	   ${DIFF} output/ex20_1.out ex20_1.tmp || printf "${PWD}\nPossible problem with ex20_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex20_1.tmp
runex20_2:
	-@${MPIEXEC} -n 1 ./ex20 -cl -prefix_push cl_ -zero 10 -prefix_push a_ -one 11 -prefix_push bb_ -two 12 -prefix_pop -three 13 -prefix_push cc_ -prefix_push ddd_ -four 14 -prefix_pop -prefix_pop -prefix_pop -five 15 -prefix_pop > ex20_2.tmp 2>&1;   \
	   ${DIFF} output/ex20_2.out ex20_2.tmp || printf "${PWD}\nPossible problem with ex20_2, diffs above\n=========================================\n"; \
	   ${RM} -f ex20_2.tmp
runex20_3:
	-@${MPIEXEC} -n 1 ./ex20 -str > ex20_3.tmp 2>&1;   \
	   ${DIFF} output/ex20_3.out ex20_3.tmp || printf "${PWD}\nPossible problem with ex20_3, diffs above\n=========================================\n"; \
	   ${RM} -f ex20_3.tmp
runex22:
	-@${MPIEXEC} -n 1 ./ex22 > ex22.tmp 2>&1;   \
	   ${DIFF} output/ex22.out ex22.tmp || printf "${PWD}\nPossible problem with ex22, diffs above\n=========================================\n"; \
	   ${RM} -f ex22.tmp
runex23:
	-@${MPIEXEC} -n 1 ./ex23 -options_file_yaml ex23options > ex23.tmp 2>&1;   \
	   ${DIFF} output/ex23.out ex23.tmp || printf "${PWD}\nPossible problem with ex23, diffs above\n=========================================\n"; \
	   ${RM} -f ex23.tmp
runex25:
	-@${MPIEXEC} -n 1 ./ex25 > ex25.tmp 2>&1;   \
	   ${DIFF} output/ex25.out ex25.tmp || echo  ${PWD} "\nPossible problem with ex25, diffs above \n========================================="; \
	   ${RM} -f ex25.tmp
runex28:
	-@${MPIEXEC} -n 1 ./ex28 > ex28.tmp 2>&1;   \
	   ${DIFF} output/ex28.out ex28.tmp || echo  ${PWD} "\nPossible problem with ex28, diffs above \n========================================="; \
	   ${RM} -f ex28.tmp


TESTEXAMPLES_C		       = ex8.PETSc runex8 runex8_f ex8.rm ex19.PETSc runex19 ex19.rm \
                                 ex20.PETSc runex20 runex20_2 runex20_3 ex20.rm  ex21.PETSc ex21.rm \
                                 ex22.PETSc runex22 ex22.rm ex24.PETSc ex24.rm \
                                 ex25.PETSc runex25 ex25.rm ex28.PETSc ex28.rm

TESTEXAMPLES_C_COMPLEX         = ex14.PETSc runex14 ex14.rm

TESTEXAMPLES_C_X	       = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm ex3.PETSc ex3.rm
TESTEXAMPLES_FORTRAN	       = ex5f.PETSc ex5f.rm ex6f.PETSc ex6f.rm ex17f.PETSc ex17f.rm
TESTEXAMPLES_FORTRAN_NOCOMPLEX = ex1f.PETSc ex1f.rm
TESTEXAMPLES_FORTRAN_COMPLEX   = ex1f.PETSc ex1f.rm
TESTEXAMPLES_FORTRAN_MPIUNI    = ex5f.PETSc ex5f.rm ex6f.PETSc ex6f.rm
TESTEXAMPLES_C_X_MPIUNI      = ex1.PETSc ex1.rm ex2.PETSc ex2.rm ex3.PETSc ex3.rm
TESTEXAMPLES_13		       = ex9.PETSc ex9.rm ex10.PETSc ex10.rm ex11.PETSc ex11.rm ex12.PETSc ex12.rm \
                                 ex14.PETSc ex14.rm

include ${PETSC_DIR}/lib/petsc/conf/test
