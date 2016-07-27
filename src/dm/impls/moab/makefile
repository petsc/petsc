#requirespackage  'PETSC_HAVE_MOAB'

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCECXX= dmmoab.cxx dmmbvec.cxx dmmbmat.cxx dmmbfield.cxx dmmbmg.cxx dmmbfem.cxx dmmbio.cxx dmmbutil.cxx
SOURCEF  =
SOURCEH  =
DIRS     = examples
LIBBASE  = libpetscdm
MANSEC   = DM
LOCDIR   = src/dm/impls/moab/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

