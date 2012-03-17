#requirespackage 'PETSC_HAVE_PTHREADCLASSES'
ALL: lib

CFLAGS   = ${PNETCDF_INCLUDE}
FFLAGS   =
SOURCEC  = mpivecpthread.c
SOURCEF  =
SOURCEH  =
OBJSC    = mpivecpthread.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/vec/impls/mpi/mpipthread

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
