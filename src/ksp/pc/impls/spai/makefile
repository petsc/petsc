
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
OBJSC    = ispai.o dspai.o
OBJSF    =
LIBBASE  = libpetscksp
DIRS     = 
MANSEC   = PC
LOCDIR   = src/ksp/pc/impls/spai/src/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test


