# $Id: makefile,v 1.1 2001/08/29 20:54:49 bsmith Exp bsmith $ 

#requirespackage  'PETSC_HAVE_ESI'
#requireslanguage  CPP
#requiresscalar    real 
#requiresprecision double

ALL: lib

CFLAGS   = ${ESI_INCLUDE}
FFLAGS   =
SOURCEC  = ebvec1.c
SOURCEF  =
SOURCEH  = 
OBJSC    = ebvec1.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/impls/esi/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

