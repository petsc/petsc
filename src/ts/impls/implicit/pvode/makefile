# $Id: makefile,v 1.13 2001/08/22 18:02:23 balay Exp $ 

ALL: lib

#requirespackage   'PETSC_HAVE_PVODE'
#requiresprecision double
#requiresscalar    real

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

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

