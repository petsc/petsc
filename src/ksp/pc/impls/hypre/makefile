# $Id: makefile,v 1.39 2001/08/22 17:59:56 balay Exp $ 

#requirespackage  'PETSC_HAVE_HYPRE'
#requiresprecision double
#requiresscalar    real


ALL: lib

CFLAGS   = ${HYPRE_INCLUDE}
FFLAGS   =
SOURCEC  = hyppilut.c
SOURCEF  =
SOURCEH  = 
OBJSC    = hyppilut.o
OBJSF    =
LIBBASE  = libpetscksp
MANSEC   = PC
LOCDIR   = src/ksp/pc/impls/hypre/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

