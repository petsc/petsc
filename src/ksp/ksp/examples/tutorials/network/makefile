#requiresdefine   'PETSC_HAVE_ATTRIBUTEALIGNED'

CFLAGS	         =
FFLAGS	         =
CPPFLAGS         =
FPPFLAGS         =
EXAMPLESC        = ex1.c ex2.c ex1_nest.c
LOCDIR		 = src/ksp/ksp/examples/tutorials/network/
MANSEC           = KSP

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


ex1: ex1.o chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_KSP_LIB}
	${RM} ex1.o

ex2: ex2.o chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_KSP_LIB}
	${RM} ex2.o

ex1_nest: ex1_nest.o chkopts
	-${CLINKER} -o ex1_nest ex1_nest.o ${PETSC_KSP_LIB}
	${RM} ex1_nest.o

include ${PETSC_DIR}/lib/petsc/conf/test
