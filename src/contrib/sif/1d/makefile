# $Id: makefile,v 1.2 1997/10/14 21:27:38 bsmith Exp balay $ 

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

