# $Id: makefile,v 1.3 1998/06/15 16:35:08 balay Exp $ 

CFLAGS   =
FFLAGS   =
CPPFLAGS = ${BS_INCLUDE}
LOCDIR   = src/contrib/reusememory/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex1: chkopts ex1.o reuse.o
	-${CLINKER}  -o ex1 ex1.o reuse.o ${PETSC_LIB} 
	${RM} ex1.o reuse.o