
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/dm/impls/moab/examples
EXAMPLESC       = ex1.c
EXAMPLESF       =
MANSEC          = DM

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_DM_LIB}
	${RM} -f ex1.o

#--------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1 -dim 3 -filename 8hex.g > ex1_0.tmp 2>&1;\
	   if (${DIFF} output/ex1_0.out ex1_0.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with with runex1, diffs above \n========================================="; fi ;\
	   ${RM} -f ex1_0.tmp
