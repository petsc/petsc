ALL: lib
SPECIALFASTLIB = yes
specialfastlib: lib
#requirespackage 'PETSC_USE_F90_SRC_IMPL'

CFLAGS   =
FFLAGS   =
SOURCEC  = f90_cwrap.c
SOURCEF  = f90_fwrap.F
SOURCEH  = 
OBJSC    =
OBJSC	 = ${SOURCEC:.c=.o}
OBJSF	 = ${SOURCEF:.F=.o}
DIRS     = 
LIBBASE  = libpetsc
MANSEC   = Sys
LOCDIR   = src/sys/f90-src/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
