
#requirespackage  'PETSC_HAVE_BLOCKSOLVE95'
#requiresscalar real 

ALL: lib

CFLAGS   =
FFLAGS   =
CPPFLAGS = ${BLOCKSOLVE95_INCLUDE}
SOURCEC  = mpirowbs.c cholbs.c iccbs.c
SOURCEF  =
SOURCEH  = mpirowbs.h
OBJSC    = mpirowbs.o cholbs.o iccbs.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     =
MANSEC   = Mat
LOCDIR   = src/mat/impls/rowbs/mpi/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
