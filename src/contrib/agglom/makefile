# $Id: makefile,v 1.4 1998/06/15 16:30:20 balay Exp $ 

All: triconvert

CFLAGS   = 
FFLAGS   = 

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

triconvert: chkopts triconvert.o
	-${CLINKER} -o triconvert triconvert.o ${PETSC_LIB}
	-${RM} triconvert.o