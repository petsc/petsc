# $Id: makefile,v 1.1 1999/11/02 21:01:49 bsmith Exp bsmith $ 
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

runexamples:

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
