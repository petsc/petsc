ALL: 

CFLAGS		 = 
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
EXAMPLESC        = maros.c toy.c
EXAMPLESF        = toyf.F
EXAMPLESCH       =
EXAMPLESFH       = 
TUTORIALS        = 
LOCDIR		 = src/tao/constrained/examples/tutorials/
MANSEC		 =
DOCS		 = 
DIRS		 = 

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

maros: maros.o chkopts
	-${CLINKER} -o maros maros.o ${PETSC_TAO_LIB}
	${RM} maros.o

toy: toy.o chkopts
	-${CLINKER} -o toy toy.o ${PETSC_TAO_LIB}
	${RM} toy.o

toyf: toyf.o chkopts
	-${CLINKER} -o toyf toyf.o ${PETSC_TAO_LIB}
	${RM} toyf.o

runmaros:
	-@${MPIEXEC} -n 1 ./maros   > maros_1.tmp 2>&1; \
           ${DIFF} output/maros_1.out maros_1.tmp || printf  "${PWD}\nPossible problem with maros_1, diffs above\n=========================================\n"; \
           ${RM} -f maros_1.tmp

runtoy:
	-@${MPIEXEC} -n 1 ./toy -tao_converged_reason  > toy_1.tmp 2>&1; \
           ${DIFF} output/toy_1.out toy_1.tmp || printf  "${PWD}\nPossible problem with toy_1, diffs above\n=========================================\n"; \
           ${RM} -f toy_1.tmp

TESTEXAMPLES_C             =  maros.PETSc maros.rm toy.PETSc toy.rm
TESTEXAMPLES_C_SUPERLU     =  maros.PETSc runmaros maros.rm toy.PETSc runtoy toy.rm
TEXTEXAMPLES_FORTRAN       = toyf.PETSc toyf.rm 

include ${PETSC_DIR}/lib/petsc/conf/test

