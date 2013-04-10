
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/dm/impls/moab/examples/tests
EXAMPLESC       = ex1.cxx
EXAMPLESF       =
MANSEC          = DM

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_DM_LIB}
	${RM} -f ex1.o

#--------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1 > ex1_1.tmp 2>&1;\
	   if (${DIFF} output/ex1_1.out ex1_1.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with with runex1, diffs above \n========================================="; fi ;\
	   ${RM} -f ex1_1.tmp

TESTEXAMPLES_MOAB = ex1.PETSc runex1 ex1.rm

include ${PETSC_DIR}/conf/test
