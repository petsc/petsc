# $Id: makefile,v 1.4 2001/03/22 20:29:12 bsmith Exp balay $ 



CFLAGS	        = 
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/sys/src/matlabengine/examples/tutorials/
EXAMPLESC       = ex1.c
EXAMPLESF       = 

include ${PETSC_DIR}/bmake/common/base

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_SYS_LIB}
	${RM} -f ex1.o

include ${PETSC_DIR}/bmake/common/test



