ALL: 

CFLAGS		 = 
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = src/tao/leastsquares/examples/tutorials/
MANSEC		 =
EXAMPLESC        = chwirut1.c chwirut2.c
EXAMPLESCH       =
EXAMPLESF        = chwirut1f.F90 chwirut2f.F90
EXAMPLESFH       = chwirut2f.h
TUTORIALS        = chwirut1.c chwirut2.c chwirut1f.F90 chwirut2f.F90
DOCS		 = 
DIRS		 = 
CLEANFILES       = chwirut1 chwirut2 chwirut1f chwirut2f

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

chwirut1: chwirut1.o chkopts
	-${CLINKER} -o chwirut1 chwirut1.o ${PETSC_TAO_LIB}
	${RM} chwirut1.o

chwirut2: chwirut2.o chkopts
	-${CLINKER} -o chwirut2 chwirut2.o ${PETSC_TAO_LIB}
	${RM} chwirut2.o

chwirut1f: chwirut1f.o chkopts
	-${FLINKER} -o chwirut1f chwirut1f.o ${PETSC_TAO_LIB}
	${RM} chwirut1f.o

chwirut2f: chwirut2f.o chkopts
	-${FLINKER} -o chwirut2f chwirut2f.o ${PETSC_TAO_LIB}
	${RM} chwirut2f.o

runchwirut1:
	-@${MPIEXEC} -n 1 ./chwirut1 -tao_smonitor -tao_max_it 100 -tao_type pounders > chwirut1_1.tmp 2>&1;\
	${DIFF} output/chwirut1_1.out chwirut1_1.tmp || printf '${PWD}\nPossible problem with chwirut1_1 stdout, diffs above \n=====================================\n';\
	${RM} -f chwirut1_1.tmp

runchwirut2:
	-@${MPIEXEC} -n 3 ./chwirut2 -tao_smonitor -tao_max_it 100 -tao_type pounders > chwirut2_1.tmp 2>&1;\
	${DIFF} output/chwirut2_1.out chwirut2_1.tmp || printf '${PWD}\nPossible problem with chwirut2_1 stdout, diffs above \n=====================================\n';\
	${RM} -f chwirut2_1.tmp

runchwirut1f:
	-@${MPIEXEC} -n 1 ./chwirut1f -tao_smonitor -tao_max_it 100 -tao_type pounders > chwirut1f_1.tmp 2>&1;\
	${DIFF} output/chwirut1f_1.out chwirut1f_1.tmp || printf '${PWD}\nPossible problem with chwirut1f_1 stdout, diffs above \n=====================================\n';\
	${RM} -f chwirut1f_1.tmp

runchwirut2f:
	-@${MPIEXEC} -n 3 ./chwirut2f -tao_smonitor -tao_max_it 100 -tao_type pounders > chwirut2f_1.tmp 2>&1;\
	${DIFF} output/chwirut2f_1.out chwirut2f_1.tmp || printf '${PWD}\nPossible problem with chwirut2f_1 stdout, diffs above \n=====================================\n';\
	${RM} -f chwirut2f_1.tmp

TESTEXAMPLES_C_NOTSINGLE   = chwirut1.PETSc runchwirut1 chwirut1.rm chwirut2.PETSc runchwirut2 chwirut2.rm
TESTEXAMPLES_C_X_MPIUNI    = chwirut1.PETSc runchwirut1 chwirut1.rm 
TESTEXAMPLES_FORTRAN_NOTSINGLE = chwirut1f.PETSc runchwirut1f chwirut1f.rm printdot chwirut2f.PETSc runchwirut2f chwirut2f.rm 
TESTEXAMPLES_FORTRAN_MPIUNI= chwirut1f.PETSc runchwirut1f chwirut1f.rm 


include ${PETSC_DIR}/lib/petsc/conf/test


