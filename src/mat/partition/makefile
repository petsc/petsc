# $Id: makefile,v 1.4 1998/03/19 02:08:40 balay Exp balay $ 
#
ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' ${PARMETIS_INCLUDE}
SOURCEC  = partition.c spartition.c pmetis.c
SOURCEH  = 
OBJSC    = partition.o spartition.o pmetis.o
DOCS     = 
LIBBASE  = libpetscmat
LOCDIR   = src/mat/impls/partition/
MANSEC   = 2

runexamples:

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
