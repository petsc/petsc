# $Id: makefile,v 1.4 1998/06/15 16:22:59 balay Exp balay $ 

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


