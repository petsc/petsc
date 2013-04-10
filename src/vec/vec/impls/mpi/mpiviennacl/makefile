#requirespackage 'PETSC_HAVE_VIENNACL'

ALL: lib

CFLAGS   = ${PNETCDF_INCLUDE}
FFLAGS   =
SOURCEC  = mpiviennacl.cxx
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/vec/impls/mpi/mpiviennacl/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
