# $Id: makefile,v 1.6 1998/06/12 21:41:28 balay Exp balay $ 
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
MANSEC   = 2

runexamples:

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
