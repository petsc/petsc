# $Id: makefile,v 1.1 1998/04/22 18:02:53 curfman Exp balay $ 

ALL: ex1

CFLAGS	         = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
FFLAGS	         = 
LOCDIR           = src/contrib/elliott/

EXAMPLESC       = ex1.c
EXAMPLESF       = 

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex1: ex1.o rbtc.o chkopts
	-${CLINKER} -o ex1 ex1.o rbtc.o
	${RM} ex1.o rbtc.o


