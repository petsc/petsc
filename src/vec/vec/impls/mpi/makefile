
ALL: lib

CFLAGS   = ${PNETCDF_INCLUDE}
FFLAGS   =
SOURCEC  = pbvec.c pdvec.c pvec2.c pmap.c vmpicr.c commonmpvec.c
SOURCEF  =
SOURCEH  = pvecimpl.h 
LIBBASE  = libpetscvec
MANSEC   = Vec
DIRS     = mpicusp
LOCDIR   = src/vec/vec/impls/mpi/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
