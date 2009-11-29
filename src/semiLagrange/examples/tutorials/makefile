
# This directory contains SemiLagrange example programs for
# advecting fields using the method of characteristics.

CFLAGS		   =
FFLAGS		   =
CPPFLAGS           =
FPPFLAGS           =
LOCDIR		   = src/semiLagrange/examples/tutorials/
MANSEC             = SemiLagrange
EXAMPLESC	   = ex1.c ex2.c ex4.c
EXAMPLESF	   =
EXAMPLESCH	   =
EXAMPLESFH         =
EXAMPLESMATLAB     =
DIRS		   =

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_SNES_LIB} ${PETSC_CONTRIB}
	${RM} ex1.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_SNES_LIB} ${PETSC_CONTRIB}
	${RM} ex2.o

ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o ${PETSC_SNES_LIB} ${PETSC_CONTRIB}
	${RM} ex4.o
#--------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1 > ex1.tmp 2>&1;   \
	   if (${DIFF} output/ex1.out ex1.tmp) then true; \
	   else echo "Possible problem with ex1, diffs above"; fi; \
	   ${RM} -f ex1.tmp

runex2:
	-@${MPIEXEC} -n 1 ./ex2 > ex2.tmp 2>&1;   \
	   if (${DIFF} output/ex2.out ex2.tmp) then true; \
	   else echo "Possible problem with ex2, diffs above"; fi; \
	   ${RM} -f ex2.tmp

runex4:
	-@${MPIEXEC} -n 1 ./ex4 > ex4.tmp 2>&1;   \
	   if (${DIFF} output/ex4.out ex4.tmp) then true; \
	   else echo "Possible problem with ex4, diffs above"; fi; \
	   ${RM} -f ex4.tmp

TESTEXAMPLES_C_NOCOMPLEX = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm ex4.PETSc runex4 ex4.rm

include ${PETSC_DIR}/conf/test
