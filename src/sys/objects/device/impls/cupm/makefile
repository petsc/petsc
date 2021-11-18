#requiresdefine 'PETSC_HAVE_CXX_DIALECT_CXX11'
ALL: lib

CFLAGS    =
FFLAGS    =
CPPFLAGS  =
SOURCEC	  =
SOURCECXX = cupminterface.cxx cupmdevice.cxx
SOURCEF	  =
SOURCEH   = cupmcontext.hpp
MANSEC	  = Sys
LIBBASE	  = libpetscsys
DIRS	  = cuda hip
LOCDIR	  = src/sys/objects/device/impls/cupm/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

