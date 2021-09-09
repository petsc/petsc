#requiresdefine 'PETSC_HAVE_CXX_DIALECT_CXX11'
ALL: lib

CFLAGS    =
FFLAGS    =
CPPFLAGS  =
SOURCEC	  =
SOURCECXX = dcontext.cxx device.cxx devicereg.cxx cupminterface.cxx
SOURCEF	  =
SOURCEH   = objpool.hpp cupmdevice.hpp
MANSEC	  = Sys
LIBBASE	  = libpetscsys
DIRS	  =
LOCDIR	  = src/sys/objects/device/interface/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

