-include ../../../../../../petscdir.mk
#requirespackage 'PETSC_HAVE_VIENNACL'
#requiresdefine 'PETSC_HAVE_VIENNACL_NO_CUDA'

SOURCECXX = aijviennacl.cxx
SOURCEH   = viennaclmatimpl.h
LIBBASE   = libpetscmat
MANSEC    = Mat

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
