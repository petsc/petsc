# $Id: makefile,v 1.8 1999/02/02 21:30:33 balay Exp bsmith $ 

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

