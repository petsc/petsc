# $Id: makefile,v 1.4 2000/10/24 20:26:18 bsmith Exp bsmith $ 
#
ALL: lib

#requirespackage 'PETSC_HAVE_PARMETIS'

CFLAGS   =
FFLAGS   =
CPPFLAGS = ${PARMETIS_INCLUDE}
SOURCEC  = pmetis.c
SOURCEH  = 
OBJSC    = pmetis.o
LIBBASE  = libpetscmat
LOCDIR   = src/mat/partition/impls/pmetis/partition/
MANSEC   = MatOrderings



include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test
