
ALL: lib

CFLAGS   = ${PNETCDF_INCLUDE}
FFLAGS   =
SOURCEC  = pbvec.c pdvec.c pvec2.c pmap.c vmpicr.c
SOURCEF  =
SOURCEH  = pvecimpl.h 
OBJSC    = pbvec.o pdvec.o pvec2.o pmap.o vmpicr.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/impls/mpi/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
