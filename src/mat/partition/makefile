# $Id: makefile,v 1.3 1997/11/03 04:46:32 bsmith Exp balay $ 
#
ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"' ${PARMETIS_INCLUDE}
SOURCEC  = partition.c spartition.c pmetis.c
SOURCEH  = 
OBJSC    = partition.o spartition.o pmetis.o
DOCS     = 
LIBBASE  = libpetscmat
LOCDIR   = src/mat/impls/partition/
MANSEC   = 2

runexamples:

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
