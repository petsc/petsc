# $Id: makefile,v 1.2 2001/09/07 20:09:03 bsmith Exp $ 

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

