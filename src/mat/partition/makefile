# $Id: makefile,v 1.8 1999/02/02 21:27:16 balay Exp bsmith $ 
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
