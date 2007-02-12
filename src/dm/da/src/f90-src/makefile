ALL: lib
SPECIALFASTLIB = yes
specialfastlib: lib
#requirespackage 'PETSC_USING_F90'

CFLAGS   =
FFLAGS   =
SOURCEC  = 
SOURCEF  = daindexf90.F
SOURCEH  = 
OBJSC    =
OBJSF    = ${SOURCEF:.F=.o}
DIRS     = f90-src
LIBBASE  = libpetscdm
MANSEC   = Vec
LOCDIR   = src/dm/da/src/f90-src/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
