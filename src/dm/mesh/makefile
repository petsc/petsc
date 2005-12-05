#requirespackage   'PETSC_USE_SIEVE'
#requireslanguage CXXONLY

ALL: lib

CFLAGS   = -Isieve ${TRIANGLE_INCLUDE}
FFLAGS   =
SOURCEC  = mesh.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = mesh.o
OBJSF    =
DIRS     = sieve
LIBBASE  = libpetscdm
MANSEC   = DA
LOCDIR   = src/dm/mesh/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
