
ALL:

CFLAGS	        =
FFLAGS	        = -I${PETSC_DIR}/include/petsc/finclude
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/sys/examples/tutorials/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex9.c ex11.c ex12.cxx ex13.cxx ex14.cxx ex15.c ex16.c
EXAMPLESF       = ex3f.F ex4f.F ex4f90.F90 ex5f90.F90 ex8f90.F90 ex10f90.F90
MANSEC          = Sys
CLEANFILES      = binaryoutput binaryoutput.info

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o   chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_SYS_LIB}
	${RM} -f ex1.o

ex2: ex2.o   chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_SYS_LIB}
	${RM} -f ex2.o

ex3: ex3.o   chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_SYS_LIB}
	${RM} -f ex3.o

ex3f: ex3f.o   chkopts
	-${FLINKER} -o ex3f ex3f.o  ${PETSC_SYS_LIB}
	${RM} -f ex3f.o

ex4: ex4.o   chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_SYS_LIB}
	${RM} -f ex4.o

ex4f: ex4f.o   chkopts
	-${FLINKER} -o ex4f ex4f.o  ${PETSC_SYS_LIB}
	${RM} -f ex4f.o

ex4f90: ex4f90.o   chkopts
	-${FLINKER} -o ex4f90 ex4f90.o  ${PETSC_SYS_LIB}
	${RM} -f ex4f90.o

ex5: ex5.o   chkopts
	-${CLINKER} -o ex5 ex5.o  ${PETSC_SYS_LIB}
	${RM} -f ex5.o

ex5f90: ex5f90.o  chkopts
	-${FLINKER} -o ex5f90 ex5f90.o ${PETSC_SYS_LIB}
	${RM} ex5f90.o

ex6: ex6.o   chkopts
	-${CLINKER} -o ex6 ex6.o  ${PETSC_SYS_LIB}
	${RM} -f ex6.o

ex7: ex7.o   chkopts
	-${CLINKER} -o ex7 ex7.o  ${PETSC_SYS_LIB}
	${RM} -f ex7.o

ex8f90: ex8f90.o  chkopts
	-${FLINKER} -o ex8f90 ex8f90.o ${PETSC_SYS_LIB}
	${RM} ex8f90.o

ex9: ex9.o   chkopts
	-${CLINKER} -o ex9 ex9.o  ${PETSC_SYS_LIB}
	${RM} -f ex9.o

ex10f90: ex10f90.o   chkopts
	-${FLINKER} -o ex10f90 ex10f90.o  ${PETSC_SYS_LIB}
	${RM} -f ex10f90.o

ex11: ex11.o   chkopts
	-${CLINKER} -o ex11 ex11.o  ${PETSC_SYS_LIB}
	${RM} -f ex11.o

ex12: ex12.o   chkopts
	-${CLINKER} -o ex12 ex12.o  ${PETSC_SYS_LIB}
	${RM} -f ex12.o

ex13: ex13.o   chkopts
	-${CLINKER} -o ex13 ex13.o  ${PETSC_SYS_LIB}
	${RM} -f ex13.o

ex14: ex14.o   chkopts
	-${CLINKER} -o ex14 ex14.o  ${PETSC_SYS_LIB}
	${RM} -f ex14.o

ex15: ex15.o   chkopts
	-${CLINKER} -o ex15 ex15.o  ${PETSC_SYS_LIB}
	${RM} -f ex15.o

ex16: ex16.o   chkopts
	-${CLINKER} -o ex16 ex16.o  ${PETSC_SYS_LIB}
	${RM} -f ex16.o
#
#
# ------------------------------------------------------------------------
#
runex1:
	-@${MPIEXEC} -n 1 ./ex1 > ex1_1.tmp 2>&1; \
	   if (${DIFF} output/ex1_1.out ex1_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex1_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex1_1.tmp

runex2:
	-@${MPIEXEC} -n 1 ./ex2 > ex2_1.tmp 2>&1; \
	   if (${DIFF} output/ex2_1.out ex2_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex2_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex2_1.tmp

runex3:
	-@${MPIEXEC} -n 1 ./ex3

runex3f:
	-@${MPIEXEC} -n 1 ./ex3f

runex4:
	-@${MPIEXEC} -n 5 ./ex4  | sort -b  > ex4_1.tmp 2>&1; \
           if (sort -b output/ex4_1.out | ${DIFF} - ex4_1.tmp) then true; \
           else printf "${PWD}\nPossible problem with ex4_1, diffs above\n=========================================\n"; fi; \
           ${RM} -f ex4_1.tmp

runex4f:
	-@${MPIEXEC} -n 5 ./ex4f  | sort -b > ex4f_1.tmp 2>&1; \
           if (sort -b output/ex4f_1.out | ${DIFF} - ex4f_1.tmp) then true; \
           else printf "${PWD}\nPossible problem with ex4f_1, diffs above\n=========================================\n"; fi; \
           ${RM} -f ex4f_1.tmp

runex4f90:
	-@${MPIEXEC} -n 1 ./ex4f90 > ex4f90_1.tmp 5>&1; \
	   if (${DIFF} output/ex4f90_1.out ex4f90_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex4f90_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex4f90_1.tmp

runex5:
	-@${MPIEXEC} -n 1 ./ex5 -pbag_rho 44 -pbag_do_output true > ex5_1.tmp 5>&1; \
	   if (${DIFF} output/ex5_1.out ex5_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex5_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex5_1.tmp

runex5f90:
	-@${MPIEXEC} -n 1 ./ex5f90 -pbag_rarray 4,5,88 > ex5f90_1.tmp 5>&1; \
	   if (${DIFF} output/ex5f90_1.out ex5f90_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex5f90_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex5f90_1.tmp

runex6:
	-@${MPIEXEC} -n 1 ./ex6

runex8f90:
	-@${MPIEXEC} -n 1 ./ex8f90 -joe_jeff b456 > ex8f90_1.tmp 8>&1; \
	   if (${DIFF} output/ex8f90_1.out ex8f90_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex8f90_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex8f90_1.tmp

runex9_1:
	-@PETSC_OPTIONS= ${MPIEXEC} -n 1 ./ex9 -f petsc.yml > ex9_1.tmp 5>&1; \
	   if (${DIFF} output/ex9_1.out ex9_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex9_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex9_1.tmp

runex9_2:
	-@PETSC_OPTIONS= ${MPIEXEC} -n 1 ./ex9 -options_file_yaml petsc.yml > ex9_2.tmp 5>&1; \
	   if (${DIFF} output/ex9_2.out ex9_2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex9_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex9_2.tmp

runex10f90_1:
	-@PETSC_OPTIONS= ${MPIEXEC} -n 1 ./ex10f90 -f petsc.yml > ex10f90_1.tmp 8>&1; \
	   if (${DIFF} output/ex10f90_1.out ex10f90_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex10f90_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex10f90_1.tmp

runex10f90_2:
	-@PETSC_OPTIONS= ${MPIEXEC} -n 1 ./ex10f90 -options_file_yaml petsc.yml > ex10f90_2.tmp 8>&1; \
	   if (${DIFF} output/ex10f90_2.out ex10f90_2.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex10f90_2, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex10f90_2.tmp

runex11:
	-@${MPIEXEC} -n 1 ./ex11

runex12:
	-@${MPIEXEC} -n 1 ./ex12

runex13:
	-@${MPIEXEC} -n 3 ./ex13

runex14:
	-@${MPIEXEC} -n 3 ./ex14

runex15:
	-@PETSC_OPTIONS= ${MPIEXEC} -n 4 ./ex15 -viewer > ex15.tmp 5>&1; \
	   if (${DIFF} output/ex15_1.out ex15.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex15, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex15.tmp

runex16:
	-@PETSC_OPTIONS= ${MPIEXEC} -n 2  ./ex16 -options_view -get_total_flops |grep -v malloc | grep -v display | grep -v nox | grep -v "Total flops" > ex16.tmp 5>&1; \
	   if (${DIFF} output/ex16_1.out ex16.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex16, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex16.tmp

TESTEXAMPLES_C	          = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm \
                            ex3.PETSc runex3 ex3.rm ex4.PETSc runex4 ex4.rm ex6.PETSc runex6 ex6.rm ex11.PETSc runex11 ex11.rm ex15.PETSc runex15 ex15.rm \
                            ex16.PETSc runex16 ex16.rm
TESTEXAMPLES_C_NOCOMPLEX  = ex5.PETSc runex5 ex5.rm
TESTEXAMPLES_C_X_MPIUNI   = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm \
                            ex3.PETSc runex3 ex3.rm
TESTEXAMPLES_FORTRAN	  = ex3f.PETSc runex3f ex3f.rm ex4f.PETSc runex4f ex4f.rm
TESTEXAMPLES_F90	      = ex4f90.PETSc runex4f90 ex4f90.rm
TESTEXAMPLES_F2003        = ex5f90.PETSc runex5f90 ex5f90.rm  ex8f90.PETSc runex8f90 ex8f90.rm
TESTEXAMPLES_YAML         = ex9.PETSc runex9_1 runex9_2 ex9.rm ex10f90.PETSc runex10f90_1 runex10f90_2 ex10f90.rm
TESTEXAMPLES_CHOMBO       = ex12.PETSc runex12 ex12.rm
TESTEXAMPLES_TRILINOS     = ex13.PETSc runex13 ex13.rm ex14.PETSc runex14 ex14.rm
include ${PETSC_DIR}/lib/petsc/conf/test

