# $Id: makefile,v 1.27 1997/08/22 15:13:06 bsmith Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = matrix.c mhas.c fdmatrix.c matcnv.c mgtype.c
SOURCEF  =
SOURCEH  = 
OBJSC    = matrix.o mhas.o fdmatrix.o matcnv.o mgtype.o
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2
LOCDIR   = src/mat/interface/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
