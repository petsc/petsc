# $Id: makefile,v 1.9 1999/03/17 23:23:28 bsmith Exp bsmith $ 
#
ALL: lib

CFLAGS   =
FFLAGS   =
CPPFLAGS = ${PARMETIS_INCLUDE}
SOURCEC  = partition.c spartition.c pmetis.c
SOURCEH  = 
OBJSC    = partition.o spartition.o pmetis.o
DOCS     = 
LIBBASE  = libpetscmat
LOCDIR   = src/mat/impls/partition/
MANSEC   = MatOrderings

runexamples:

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
