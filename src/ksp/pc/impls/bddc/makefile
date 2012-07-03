#requirespackage 'PETSC_HAVE_PCBDDC'

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = bddc.c
SOURCEF  =
SOURCEH  = bddc.h
OBJSC    = $(SOURCEC:.c=.o)
OBJSF    =
LIBBASE  = libpetscksp
DIRS     =
MANSEC   = PC
LOCDIR   = src/ksp/pc/impls/bddc/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
