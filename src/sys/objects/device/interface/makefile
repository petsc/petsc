-include ../../../../../petscdir.mk
#requiresdefine 'PETSC_HAVE_CXX'

SOURCEH   = petscdevice_interface_internal.hpp
SOURCECXX = dcontext.cxx device.cxx memory.cxx global_dcontext.cxx mark_dcontext.cxx
MANSEC	  = Sys
LIBBASE	  = libpetscsys
DIRS	  =
LOCDIR	  = src/sys/objects/device/interface/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

