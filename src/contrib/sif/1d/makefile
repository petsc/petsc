# $Id: makefile,v 1.3 1998/03/24 17:27:26 balay Exp balay $ 

ALL: sif

CFLAGS	        = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' 
FFLAGS	        =
LOCDIR          = src/contrib/sif
EXAMPLESC       = 
EXAMPLESF       = sif.F

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

sif: sif.o chkopts
	-${FLINKER} -o sif sif.o  ${PETSC_FORTRAN_LIB} ${PETSC_LIB}
	${RM} sif.o 

