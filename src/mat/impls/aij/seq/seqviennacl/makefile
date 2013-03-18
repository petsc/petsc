#requirespackage 'PETSC_HAVE_VIENNACL'

ALL: lib

CFLAGS    =
FFLAGS    =
SOURCEC   = aijviennacl.cxx
SOURCEF   =
SOURCEH   = viennaclmatimpl.h
LIBBASE   = libpetscmat
DIRS      =
MANSEC    = Mat
LOCDIR    = src/mat/impls/aij/seq/seqviennacl/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
