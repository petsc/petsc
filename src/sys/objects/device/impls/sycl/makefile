-include ../../../../../../petscdir.mk
#requirespackage 'PETSC_HAVE_SYCL'

SOURCEC	  =
SOURCESYCL = sycldevice.sycl.cxx syclcontext.sycl.cxx
SOURCEF	  =
SOURCEH   = sycldevice.hpp
MANSEC	  = Sys
LIBBASE	  = libpetscsys
DIRS	    =
LOCDIR	  = src/sys/objects/device/impls/sycl/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

