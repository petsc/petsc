#requiresdefine 'PETSC_HAVE_CUPM'

SOURCEC	  =
SOURCECXX = cupminterface.cxx cupmblasinterface.cxx cupmdevice.cxx
SOURCEF	  =
SOURCEH   = cupmcontext.hpp
MANSEC	  = Sys
LIBBASE	  = libpetscsys
DIRS	  = cuda hip
LOCDIR	  = src/sys/objects/device/impls/cupm/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

