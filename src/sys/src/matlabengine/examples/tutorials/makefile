# $Id: makefile,v 1.2 2000/05/04 16:24:56 bsmith Exp bsmith $ 



CFLAGS	        = 
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/sys/src/matlabengine/examples/tutorials
EXAMPLESC       = ex1.c
EXAMPLESF       = 

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_SYS_LIB}
	${RM} -f ex1.o

include ${PETSC_DIR}/bmake/common_test



