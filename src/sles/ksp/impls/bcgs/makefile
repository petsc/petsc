# $Id: makefile,v 1.26 1998/03/19 02:05:25 balay Exp balay $

ALL: lib 

CFLAGS   =  ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = bcgs.c
SOURCEF  =
OBJSC    = bcgs.o
OBJSF    =
LIBBASE  = libpetscsles
LOCDIR   = src/ksp/impls/bcgs/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
