#requirespackage 'PETSC_HAVE_VIENNACL'
#requireslanguage CXXONLY
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = vecviennacl.cpp
SOURCEF  =
SOURCEH  = viennaclvecimpl.h
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/vec/impls/seq/seqviennacl/
DIRS     =

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

