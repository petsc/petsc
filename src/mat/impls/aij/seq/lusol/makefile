# $Id: makefile,v 1.50 2001/08/22 18:00:26 balay Exp $ 
#requirespackage   'PETSC_HAVE_LUSOL'
#requiresscalar    real
#requiresprecision double

ALL: lib

CFLAGS   = ${MATLAB_ENGINE_INCLUDE} ${DSCPACK_INCLUDE}
FFLAGS   =
SOURCEC  = lusol.c
SOURCEF  = 
SOURCEH  =
OBJSC    = lusol.o
OBJSF    = 
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = Mat
LOCDIR   = src/mat/impls/aij/seq/lusol

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
