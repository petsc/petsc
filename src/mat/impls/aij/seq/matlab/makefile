# $Id: makefile,v 1.50 2001/08/22 18:00:26 balay Exp $ 
#requirespackage   'PETSC_HAVE_MATLAB'
#requiresscalar    real
#requiresprecision double

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = aijmatlab.c
SOURCEF  = 
SOURCEH  =
OBJSC    = aijmatlab.o
OBJSF    = 
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = Mat
LOCDIR   = src/mat/impls/aij/seq/matlab/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
