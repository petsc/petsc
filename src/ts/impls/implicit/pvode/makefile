# $Id: makefile,v 1.3 1997/09/10 19:05:30 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"' ${PVODE_INCLUDE}
SOURCEC  = petscpvode.c
SOURCEF  =
SOURCEH  = petscpvode.h
OBJSC    = petscpvode.o
OBJSF    =
LIBBASE  = libpetscts
MANSEC   = t
LOCDIR   = src/ts/impls/implicit/pvode/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

