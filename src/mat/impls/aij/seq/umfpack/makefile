# $Id: makefile,v 1.50 2001/08/22 18:00:26 balay Exp $ 
#requirespackage   'PETSC_HAVE_UMFPACK'
#requiresprecision double
#requiresscalar    real

ALL: lib

CFLAGS   = ${UMFPACK_INCLUDE}
FFLAGS   =
SOURCEC  = mfpack.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = umfpack.o
OBJSF    = 
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = Mat
LOCDIR   = src/mat/impls/aij/seq/umfpack

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
