# $Id: makefile,v 1.46 1998/04/17 20:10:46 balay Exp $ 

ALL: ex1

CFLAGS	         = ${CPPFLAGS} ${FCONF} -D__SDIR__='"${LOCDIR}"'
FFLAGS	         = 
LOCDIR           = src/contrib/elliott/

EXAMPLESC       = ex1.c
EXAMPLESF       = 

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex1: ex1.o rbtc.o chkopts
	-${CLINKER} -o ex1 ex1.o rbtc.o
	${RM} ex1.o rbtc.o


