#requirespackage  'PETSC_HAVE_MOAB'
#requireslanguage 'CXXONLY'
#requiresprecision double
#requiresscalar    real

CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/dm/impls/moab/examples/tutorials/
EXAMPLESC       = ex1.cxx ex2.cxx ex3.cxx
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

ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o ${PETSC_DM_LIB}
	${RM} -f ex3.o


#--------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1 -ts_monitor > ex1_1.tmp 2>&1;\
	   if (${DIFF} output/ex1_1.out ex1_1.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with with runex1, diffs above \n========================================="; fi ;\
	   ${RM} -f ex1_1.tmp

runex2:
	-@${MPIEXEC} -n 1 ./ex2 -n 10 -snes_mf -ts_monitor -ts_max_snes_failures 100 > ex2_1.tmp 2>&1;\
	   if (${DIFF} output/ex2_1.out ex2_1.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with with runex2, diffs above \n========================================="; fi ;\
	   ${RM} -f ex2_1.tmp

runex3:
	-@${MPIEXEC} -n 1 ./ex3 -bc_type dirichlet -nu .01 -n 10 -ksp_converged_reason > ex3_1.tmp 2>&1;\
	   if (${DIFF} output/ex3_1.out ex3_1.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with with runex3, diffs above \n========================================="; fi ;\
	   ${RM} -f ex3_1.tmp


TESTEXAMPLES_MOAB = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm ex3.PETSc runex3 ex3.rm 

include ${PETSC_DIR}/conf/test
