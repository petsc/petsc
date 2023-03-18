-include ../../../../../petscdir.mk
#requirespackage 'PETSC_HAVE_HPDDM'
#requiresdefine  'PETSC_HAVE_DYNAMIC_LIBRARIES'
#requiresdefine  'PETSC_USE_SHARED_LIBRARIES'

CXXFLAGS  = ${HPDDM_INCLUDE}
LIBBASE   = libpetscksp
MANSEC    = KSP
SUBMANSEC = PC

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules.doc
