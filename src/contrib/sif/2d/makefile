# $Id: makefile,v 1.5 1999/07/17 17:02:11 balay Exp $ 

ALL: sif

CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/contrib/sif/2d/

EXAMPLESC       = 
EXAMPLESF       = sif.F

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

sif: sif.o chkopts
	-${FLINKER} -o sif sif.o  ${PETSC_LIB}
	${RM} sif.o 

