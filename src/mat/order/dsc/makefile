# $Id: makefile,v 1.2 2001/01/16 18:18:44 balay Exp bsmith $ 

#requirespackage 'PETSC_HAVE_DSCPACK'

ALL: lib

CFLAGS   = ${DSCPACK_INCLUDE}
FFLAGS   =
SOURCEC  = dscorder.c
SOURCEH  = 
OBJSC    = dscorder.o
LIBBASE  = libpetscmat
LOCDIR   = src/mat/order/dsc
MANSEC   = MatOrderings



include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test
