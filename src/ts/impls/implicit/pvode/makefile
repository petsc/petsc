# $Id: makefile,v 1.12 2001/08/06 21:18:13 bsmith Exp balay $ 

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

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

