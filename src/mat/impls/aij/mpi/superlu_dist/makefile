# $Id: makefile,v 1.44 2001/08/22 18:00:29 balay Exp $ 
#requirespackage   'PETSC_HAVE_SUPERLU_DIST'
#requiresprecision double

ALL: lib

CFLAGS   = ${SUPERLU_DIST_INCLUDE}
FFLAGS   =
SOURCEC	 = superlu_dist.c
SOURCEF	 =
SOURCEH	 = 
OBJSC	 = superlu_dist.o 
OBJSF	 =
LIBBASE	 = libpetscmat
DIRS	 = 
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/aij/mpi/superlu_dist/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
