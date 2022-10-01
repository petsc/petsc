-include ../../../../../../petscdir.mk
#requirespackage 'PETSC_HAVE_HIP'
ALL: lib

SOURCEHIP = mpiaijhipsparse.hip.cpp
SOURCEH	 	= mpihipsparsematimpl.h
LIBBASE	 	= libpetscmat
DIRS	 		=
MANSEC	 	= Mat
LOCDIR	 	= src/mat/impls/aij/mpi/mpihipsparse/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
