
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR  = src/sys/classes/viewer/examples/tutorials/
EXAMPLESC       = ex1.c ex2.c
EXAMPLESF       = ex1f90.F90
MANSEC          = Sys
SUBMANSEC       = Viewer

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_SYS_LIB}
	${RM} ex1.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_SYS_LIB}
	${RM} ex2.o

ex1f90: ex1f90.o  chkopts
	-${FLINKER} -o ex1f90 ex1f90.o ${PETSC_SYS_LIB}
	${RM} ex1f90.o

#----------------------------------------------------------------------------

runex1f90:
	-@${MPIEXEC} -n 1 ./ex1f90 > ex1f90_1.tmp 2>&1; \
	   if (${DIFF} output/ex1f90_1.out ex1f90_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with ex1f90_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f ex1f90_1.tmp



TESTEXAMPLES_C          = ex2.PETSc ex2.rm 
TESTEXAMPLES_F90        =
TESTEXAMPLES_C_X_MPIUNI =

include ${PETSC_DIR}/lib/petsc/conf/test

