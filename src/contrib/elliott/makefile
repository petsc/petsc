# $Id: makefile,v 1.2 1998/04/23 22:50:58 balay Exp balay $ 

ALL: ex1

CFLAGS   =
FFLAGS   =
CPPFLAGS =
FPPFLAGS =
LOCDIR   = src/contrib/elliott/

EXAMPLESC       = ex1.c
EXAMPLESF       = 

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex1: ex1.o rbtc.o chkopts
	-${CLINKER} -o ex1 ex1.o rbtc.o
	${RM} ex1.o rbtc.o


