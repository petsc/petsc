# $Id: makefile,v 1.7 1998/06/15 15:58:42 balay Exp balay $ 
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
MANSEC   = Mat

runexamples:

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
