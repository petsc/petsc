# $Id: makefile,v 1.1 1997/12/31 21:15:30 curfman Exp balay $ 

ALL: sif

CFLAGS	        = ${CPPFLAGS} ${FCONF} -D__SDIR__='"${LOCDIR}"' 
FFLAGS	        =
LOCDIR          = src/contrib/sif
EXAMPLESC       = 
EXAMPLESF       = sif.F

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

sif: sif.o chkopts
	-${FLINKER} -o sif sif.o  ${PETSC_FORTRAN_LIB} ${PETSC_LIB}
	${RM} sif.o 

