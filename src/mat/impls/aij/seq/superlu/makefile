# $Id: makefile,v 1.50 2001/08/22 18:00:26 balay Exp $ 
#requirespackage   'PETSC_HAVE_SUPERLU'
#requiresprecision double

ALL: lib

CFLAGS   = ${SUPERLU_INCLUDE}
FFLAGS   =
SOURCEC  = superlu.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = superlu.o
OBJSF    = 
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = Mat
LOCDIR   = src/mat/impls/aij/seq/superlu/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
