# $Id: makefile,v 1.6 2001/08/22 18:01:12 balay Exp $ 
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
LOCDIR   = src/mat/partition/impls/pmetis/
MANSEC   = MatOrderings



include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
