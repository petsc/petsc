
ALL:

CFLAGS	        = 
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/sys/examples/tutorials/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c
EXAMPLESF       = ex3f.F
MANSEC          = Sys

include ${PETSC_DIR}/bmake/common/base

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
	-${FLINKER} -o ex3f ex3f.o  ${PETSC_FORTRAN_LIB} ${PETSC_SYS_LIB}
	${RM} -f ex3f.o

ex4: ex4.o   chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_SYS_LIB}
	${RM} -f ex4.o

ex5: ex5.o   chkopts
	-${CLINKER} -o ex5 ex5.o  ${PETSC_SYS_LIB}
	${RM} -f ex5.o
#
# ------------------------------------------------------------------------
#
runex1:
	-@${MPIRUN} -np 1 ./ex1 > ex1_1.tmp 2>&1; \
	   if (${DIFF} output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   ${RM} -f ex1_1.tmp 

runex2:
	-@${MPIRUN} -np 1 ./ex2 > ex2_1.tmp 2>&1; \
	   if (${DIFF} output/ex2_1.out ex2_1.tmp) then true; \
	   else echo "Possible problem with ex2_1, diffs above"; fi; \
	   ${RM} -f ex2_1.tmp 

runex3:
	-@${MPIRUN} -np 1 ./ex3 

runex3f:
	-@${MPIRUN} -np 1 ./ex3f

runex5:
	-@${MPIRUN} -np 1 ./ex5 -rho 44 -do_output true > ex5_1.tmp 5>&1; \
	   if (${DIFF} output/ex5_1.out ex5_1.tmp) then true; \
	   else echo "Possible problem with ex5_1, diffs above"; fi; \
	   ${RM} -f ex5_1.tmp 

TESTEXAMPLES_C		  = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm \
                            ex3.PETSc runex3 ex3.rm ex5.PETSc runex5 ex5.rm
TESTEXAMPLES_C_X11_MPIUNI = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm \
                            ex3.PETSc runex3 ex3.rm
TESTEXAMPLES_FORTRAN	  = ex3f.PETSc runex3f ex3f.rm
TESTEXAMPLES_13		  = ex4.PETSc ex4.rm

include ${PETSC_DIR}/bmake/common/test

