# $Id: makefile,v 1.4 1998/06/13 05:26:46 balay Exp balay $ 

ALL: sif

CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/contrib/sif/2d

EXAMPLESC       = 
EXAMPLESF       = sif.F

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

sif: sif.o chkopts
	-${FLINKER} -o sif sif.o  ${PETSC_FORTRAN_LIB} ${PETSC_LIB}
	${RM} sif.o 

