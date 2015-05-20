#requirespackage 'PETSC_HAVE_VIENNACL'

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

