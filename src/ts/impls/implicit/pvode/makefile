# $Id: makefile,v 1.9 1999/11/05 14:47:34 bsmith Exp bsmith $ 

ALL: lib

#requirespackage  '-DPETSC_HAVE_PVODE'
#requireslanguage CONLY

CFLAGS   =
FFLAGS   =
CPPFLAGS = ${PVODE_INCLUDE}
SOURCEC  = petscpvode.c
SOURCEF  =
SOURCEH  = petscpvode.h
OBJSC    = petscpvode.o
OBJSF    =
LIBBASE  = libpetscts
MANSEC   = TS
LOCDIR   = src/ts/impls/implicit/pvode/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test

