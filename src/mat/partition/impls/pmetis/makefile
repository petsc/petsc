# $Id: makefile,v 1.5 2001/03/22 20:30:45 bsmith Exp balay $ 
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



include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
