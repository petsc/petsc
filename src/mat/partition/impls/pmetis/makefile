# $Id: makefile,v 1.3 2000/08/17 04:51:58 bsmith Exp bsmith $ 
#
ALL: lib

#requirespackage 'PETSC_HAVE_PARMETIS'

CFLAGS   =
FFLAGS   =
CPPFLAGS = ${PARMETIS_INCLUDE}
SOURCEC  = pmetis.c
SOURCEH  = 
OBJSC    = pmetis.o
DOCS     = 
LIBBASE  = libpetscmat
LOCDIR   = src/mat/partition/impls/pmetis/partition/
MANSEC   = MatOrderings



include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test
