# $Id: makefile,v 1.4 1998/03/19 02:11:22 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' ${PVODE_INCLUDE}
SOURCEC  = petscpvode.c
SOURCEF  =
SOURCEH  = petscpvode.h
OBJSC    = petscpvode.o
OBJSF    =
LIBBASE  = libpetscts
MANSEC   = t
LOCDIR   = src/ts/impls/implicit/pvode/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

