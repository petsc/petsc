# $Id: makefile,v 1.2 1999/11/05 14:46:06 bsmith Exp bsmith $ 
#
ALL: lib

#requirespackage '-DPETSC_HAVE_PARMETIS'

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
