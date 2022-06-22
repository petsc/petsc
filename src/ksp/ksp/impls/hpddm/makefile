-include ../../../../../petscdir.mk
#requirespackage 'PETSC_HAVE_HPDDM'

CXXFLAGS  = ${HPDDM_INCLUDE}
SOURCECXX = hpddm.cxx
SOURCEF   =
SOURCEH   =
LIBBASE   = libpetscksp
DIRS      =
MANSEC    = KSP
LOCDIR    = src/ksp/ksp/impls/hpddm/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
