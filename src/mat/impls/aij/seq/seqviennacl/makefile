-include ../petscdir.mk
#requirespackage 'PETSC_HAVE_VIENNACL'
#requiresdefine 'PETSC_HAVE_VIENNACL_NO_CUDA'

ALL: lib

CFLAGS    =
FFLAGS    =
SOURCECXX = aijviennacl.cxx
SOURCEF   =
SOURCEH   = viennaclmatimpl.h
LIBBASE   = libpetscmat
DIRS      =
MANSEC    = Mat
LOCDIR    = src/mat/impls/aij/seq/seqviennacl/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
