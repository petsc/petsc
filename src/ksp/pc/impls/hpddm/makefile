-include ../../../../../petscdir.mk
#requirespackage 'PETSC_HAVE_HPDDM'
#requiresdefine  'PETSC_HAVE_DYNAMIC_LIBRARIES'
#requiresdefine  'PETSC_USE_SHARED_LIBRARIES'

CXXFLAGS  = ${HPDDM_INCLUDE}
SOURCECXX = hpddm.cxx
SOURCEF   =
SOURCEH   =
LIBBASE   = libpetscksp
DIRS      =
MANSEC    = KSP
SUBMANSEC = PC
LOCDIR    = src/ksp/pc/impls/hpddm/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
