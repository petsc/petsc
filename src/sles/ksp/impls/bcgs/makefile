# $Id: makefile,v 1.25 1997/08/22 15:11:14 bsmith Exp balay $

ALL: lib 

CFLAGS   =  ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = bcgs.c
SOURCEF  =
OBJSC    = bcgs.o
OBJSF    =
LIBBASE  = libpetscsles
LOCDIR   = src/ksp/impls/bcgs/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
