#requirespackage 'PETSC_HAVE_VIENNACL'
#requiresdefine 'PETSC_HAVE_VIENNACL_NO_CUDA'

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCECXX= vecviennacl.cxx
SOURCEF  =
SOURCEH  = viennaclvecimpl.h
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/vec/impls/seq/seqviennacl/
DIRS     =

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

