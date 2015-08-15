
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
EXAMPLESC       = minsurf1.c
LOCDIR          = src/tao/unconstrained/examples/tests/



include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

minsurf1: minsurf1.o  chkopts
	-${CLINKER} -o minsurf1 minsurf1.o ${PETSC_TAO_LIB}
	${RM} minsurf1.o


runminsurf1:
	-@${MPIEXEC} -n 1 ./minsurf1 -tao_smonitor -tao_type cg -tao_view -mx 10 -my 8 -tao_catol 1.0e-5 > minsurf1_1.tmp 2>&1;\
	${DIFF} output/minsurf1_1.out minsurf1_1.tmp || printf '${PWD}\nPossible problem with minsurf1 stdout, diffs above \n=========================================\n';\
	${RM} -f minsurf1_1.tmp

TESTEXAMPLES_C   = minsurf1.PETSc runminsurf1 minsurf1.rm

include ${PETSC_DIR}/lib/petsc/conf/test


