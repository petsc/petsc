# $Id: makefile,v 1.1 2001/09/12 03:29:26 bsmith Exp bsmith $

#requirespackage  'PETSC_HAVE_ESI'
#requireslanguage  CPP
#requiresscalar    real 
#requiresprecision double

ALL: lib

CFLAGS   = ${ESI_INCLUDE}  ${CCA_INCLUDE}
FFLAGS   =
SOURCEC  = mesi.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = mesi.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = 
LOCDIR   = src/mat/impls/esi/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

