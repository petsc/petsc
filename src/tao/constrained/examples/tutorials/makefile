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


TESTEXAMPLES_C             =  maros.PETSc maros.rm toy.PETSc toy.rm
TEXTEXAMPLES_FORTRAN       = toyf.PETSc toyf.rm 

include ${PETSC_DIR}/lib/petsc/conf/test

