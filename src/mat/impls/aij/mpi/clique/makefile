-include ../../../../../../petscdir.mk
#requirespackage   'PETSC_HAVE_ELEMENTAL'

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCECXX= clique.cxx
SOURCEF	 =
SOURCEH	 = matcliqueimpl.h
LIBBASE	 = libpetscmat
DIRS	 =
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/aij/mpi/clique/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
