# $Id: makefile,v 1.3 2001/03/22 20:30:50 bsmith Exp balay $ 

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



include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
