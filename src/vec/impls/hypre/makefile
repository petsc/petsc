# $Id: makefile,v 1.39 2001/08/22 17:59:56 balay Exp $ 

#requirespackage  'PETSC_HAVE_HYPRE'
#requiresprecision double

ALL: lib

CFLAGS   = ${HYPRE_INCLUDE}
FFLAGS   =
SOURCEC  = vhyp.c
SOURCEF  =
SOURCEH  = 
OBJSC    = vhyp.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/impls/hypre/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

