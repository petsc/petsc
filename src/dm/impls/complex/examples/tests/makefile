
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/dm/imples/complex/examples/tests/
EXAMPLESC       = ex1.c
EXAMPLESF       = ex1f90.F ex2f90.F
MANSEC          = DM

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_DM_LIB}
	${RM} -f ex1.o

ex1f90: ex1f90.o  chkopts
	-${FLINKER} -o ex1f90 ex1f90.o  ${PETSC_DM_LIB}
	${RM} -f ex1f90.o

ex2f90: ex2f90.o  chkopts
	-${FLINKER} -o ex2f90 ex2f90.o  ${PETSC_DM_LIB}
	${RM} -f ex2f90.o

#--------------------------------------------------------------------------
runex1_0:
	-@${MPIEXEC} -n 1 ./ex1 -dim 3 -ctetgen_verbose 4 -dm_view_detail -info -info_exclude null > ex1_0.tmp 2>&1;\
	   if (${DIFF} output/ex1_0.out ex1_0.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with with ex1_0, diffs above \n========================================="; fi ;\
	   ${RM} -f ex1_0.tmp
runex1_1:
	-@${MPIEXEC} -n 1 ./ex1 -dim 3 -ctetgen_verbose 4 -refinement_limit 0.0625 -dm_view_detail -info -info_exclude null > ex1_1.tmp 2>&1;\
	   if (${DIFF} output/ex1_1.out ex1_1.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with with ex1_1, diffs above \n========================================="; fi ;\
	   ${RM} -f ex1_1.tmp

runex1f90_0:
	-@${MPIEXEC} -n 1 ./ex1f90 > ex1f90_0.tmp 2>&1;\
	   if (${DIFF} output/ex1f90_0.out ex1f90_0.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex1f90_0, diffs above \n========================================="; fi;\
	   ${RM} -f ex1f90_0.tmp

runex2f90_0:
	-@${MPIEXEC} -n 1 ./ex2f90 > ex2f90_0.tmp 2>&1;\
	   if (${DIFF} output/ex2f90_0.out ex2f90_0.tmp) then true; \
	   else echo ${PWD} ; echo "Possible problem with with ex2f90_0, diffs above \n========================================="; fi;\
	   ${RM} -f ex2f90_0.tmp

TESTEXAMPLES_C       = ex1.PETSc runex1_0 runex1_1 ex1.rm
TESTEXAMPLES_FORTRAN = ex1f90.PETSc runex1f90_0 ex1f90.rm ex2f90.PETSc runex2f90_0 ex2f90.rm

include ${PETSC_DIR}/conf/test
