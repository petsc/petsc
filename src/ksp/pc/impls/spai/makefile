
#requirespackage   'PETSC_HAVE_SPAI'
#requiresprecision double
#requiresscalar    real

ALL: lib
#
#
CFLAGS   = 
FFLAGS   = 
CPPFLAGS = ${SPAI_INCLUDE}
SOURCEC  = ispai.c dspai.c
SOURCEF  =
SOURCEH  = 
LIBBASE  = libpetscksp
DIRS     = 
MANSEC   = PC
LOCDIR   = src/ksp/pc/impls/spai/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test


