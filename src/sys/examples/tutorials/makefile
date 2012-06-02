
ALL:

CFLAGS	        = 
FFLAGS	        = -I${PETSC_DIR}/include/finclude
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/sys/examples/tutorials/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c
EXAMPLESF       = ex3f.F ex4f.F ex4f90.F90 ex5f90.F90 ex8f90.F90
MANSEC          = Sys
CLEANFILES      = binaryoutput binaryoutput.info

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

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
#
# ------------------------------------------------------------------------
#
runex1:
	-@${MPIEXEC} -n 1 ./ex1 > ex1_1.tmp 2>&1; \
	   if (${DIFF} output/ex1_1.out ex1_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex1_1, diffs above \n========================================="; fi; \
	   ${RM} -f ex1_1.tmp 

runex2:
	-@${MPIEXEC} -n 1 ./ex2 > ex2_1.tmp 2>&1; \
	   if (${DIFF} output/ex2_1.out ex2_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex2_1, diffs above \n========================================="; fi; \
	   ${RM} -f ex2_1.tmp 

runex3:
	-@${MPIEXEC} -n 1 ./ex3 

runex3f:
	-@${MPIEXEC} -n 1 ./ex3f

runex4: 
	-@${MPIEXEC} -n 5 ./ex4  | sort  > ex4_1.tmp 2>&1; \
           if (${DIFF} output/ex4_1.out ex4_1.tmp) then true; \
           else echo ${PWD} ; echo "Possible problem with with ex4_1, diffs above \n========================================="; fi; \
           ${RM} -f ex4_1.tmp

runex4f: 
	-@${MPIEXEC} -n 5 ./ex4f  | sort > ex4f_1.tmp 2>&1; \
           if (${DIFF} output/ex4f_1.out ex4f_1.tmp) then true; \
           else echo ${PWD} ; echo "Possible problem with with ex4f_1, diffs above \n========================================="; fi; \
           ${RM} -f ex4f_1.tmp

runex4f90:
	-@${MPIEXEC} -n 1 ./ex4f90 > ex4f90_1.tmp 5>&1; \
	   if (${DIFF} output/ex4f90_1.out ex4f90_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex4f90_1, diffs above \n========================================="; fi; \
	   ${RM} -f ex4f90_1.tmp 

runex5:
	-@${MPIEXEC} -n 1 ./ex5 -pbag_rho 44 -pbag_do_output true > ex5_1.tmp 5>&1; \
	   if (${DIFF} output/ex5_1.out ex5_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex5_1, diffs above \n========================================="; fi; \
	   ${RM} -f ex5_1.tmp 

runex5f90:
	-@${MPIEXEC} -n 1 ./ex5f90 -pbag_rarray 4,5,88 > ex5f90_1.tmp 5>&1; \
	   if (${DIFF} output/ex5f90_1.out ex5f90_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex5f90_1, diffs above \n========================================="; fi; \
	   ${RM} -f ex5f90_1.tmp 

runex6:
	-@${MPIEXEC} -n 1 ./ex6 

runex8f90:
	-@${MPIEXEC} -n 1 ./ex8f90 -joe_jeff b456 > ex8f90_1.tmp 8>&1; \
	   if (${DIFF} output/ex8f90_1.out ex8f90_1.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex8f90_1, diffs above \n========================================="; fi; \
	   ${RM} -f ex8f90_1.tmp 

TESTEXAMPLES_C		  = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm \
                            ex3.PETSc runex3 ex3.rm ex4.PETSc runex4 ex4.rm ex6.PETSc runex6 ex6.rm 
TESTEXAMPLES_C_NOCOMPLEX  = ex5.PETSc runex5 ex5.rm
TESTEXAMPLES_C_X_MPIUNI = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm \
                            ex3.PETSc runex3 ex3.rm
TESTEXAMPLES_FORTRAN	  = ex3f.PETSc runex3f ex3f.rm ex4f.PETSc runex4f ex4f.rm
TESTEXAMPLES_F90	  = ex4f90.PETSc runex4f90 ex4f90.rm
TESTEXAMPLES_F2003        = ex5f90.PETSc runex5f90 ex5f90.rm  ex8f90.PETSc runex8f90 ex8f90.rm
TESTEXAMPLES_PTHREAD      = ex7.PETSc ex7.rm 

include ${PETSC_DIR}/conf/test

