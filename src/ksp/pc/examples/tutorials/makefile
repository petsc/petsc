
CFLAGS	        = 
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ksp/pc/examples/tutorials/
EXAMPLESC       = ex1.c ex2.c
EXAMPLESF       =
MANSEC          = PC

include ${PETSC_DIR}/bmake/common/base

ex1: ex1.o  chkopts
	-${CLINKER}  -o ex1 ex1.o ${PETSC_KSP_LIB} 
	${RM} ex1.o

ex2: ex2.o  chkopts
	-${CLINKER}  -o ex2 ex2.o ${PETSC_KSP_LIB} 
	${RM} ex2.o

TESTEXAMPLES_C	   = ex1.PETSc ex1.rm
TESTEXAMPLES_C_X11 = ex2.PETSc ex2.rm

include ${PETSC_DIR}/bmake/common/test

