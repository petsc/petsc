# $Id: makefile,v 1.5 1998/06/13 05:26:09 balay Exp bsmith $ 

ALL: sif

CFLAGS	        =
FFLAGS	        = -O2
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/contrib/sif/1d/

EXAMPLESC       = 
EXAMPLESF       = sif.F

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

sif: sif.o chkopts
	-${FLINKER} -o sif sif.o  ${PETSC_FORTRAN_LIB} ${PETSC_LIB}
	${RM} sif.o 

