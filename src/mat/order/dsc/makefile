# $Id: makefile,v 1.31 2000/11/28 17:28:44 bsmith Exp $ 

#requirespackage 'PETSC_HAVE_DSCPACK'

ALL: lib

CFLAGS   = ${DSCPACK_INCLUDE}
FFLAGS   =
SOURCEC  = dscorder.c
SOURCEH  = 
OBJSC    = dscorder.o
LIBBASE  = libpetscmat
DOCS     = 
LOCDIR   = src/mat/order/dsc
MANSEC   = MatOrderings



include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test
