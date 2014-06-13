#requirespackage  'PETSC_HAVE_MOAB'

CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/dm/impls/moab/examples/tutorials/
EXAMPLESC       = ex1.cxx ex2.cxx
EXAMPLESF       =
MANSEC          = DM

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_DM_LIB}
	${RM} -f ex1.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_DM_LIB}
	${RM} -f ex2.o

#--------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1 -ts_monitor > ex1_1.tmp 2>&1;\
	   if (${DIFF} output/ex1_1.out ex1_1.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with with runex1, diffs above \n========================================="; fi ;\
	   ${RM} -f ex1_1.tmp

runex1_2:
	-@${MPIEXEC} -n 2 ./ex1 -ts_monitor -n 100 > ex1_2.tmp 2>&1;\
	   if (${DIFF} output/ex1_2.out ex1_2.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with with runex1_2, diffs above \n========================================="; fi ;\
	   ${RM} -f ex1_2.tmp

runex2:
	-@${MPIEXEC} -n 1 ./ex2 -bc_type dirichlet -nu .01 -n 10 -ksp_type cg -pc_type sor -ksp_converged_reason > ex2_1.tmp 2>&1;\
	   if (${DIFF} output/ex2_1.out ex2_1.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with with runex2, diffs above \n========================================="; fi ;\
	   ${RM} -f ex2_1.tmp

runex2_2:
	-@${MPIEXEC} -n 2 ./ex2 -bc_type dirichlet -nu .01 -n 20 -ksp_type cg -pc_type sor -ksp_converged_reason > ex2_2.tmp 2>&1;\
	   if (${DIFF} output/ex2_2.out ex2_2.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with with runex2_2, diffs above \n========================================="; fi ;\
	   ${RM} -f ex2_2.tmp


TESTEXAMPLES_MOAB = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm

include ${PETSC_DIR}/conf/test
