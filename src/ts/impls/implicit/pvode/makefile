# $Id: makefile,v 1.11 2000/10/24 20:27:26 bsmith Exp bsmith $ 

ALL: lib

#requirespackage   'PETSC_HAVE_PVODE'
#requireslanguage  CONLY
#requiresprecision double

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

