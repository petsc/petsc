
#requirespackage   'PETSC_HAVE_SILO'

ALL: lib 

CFLAGS    =
FFLAGS    =
SOURCEC   = vsilo.c
SOURCEH   = vsilo.h
OBJSC     = ${SOURCEC:.c=.o}
MANSEC    = Viewer
LIBBASE   = libpetsc
DIRS      = 
LOCDIR    = src/viewer/impls/silo/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
