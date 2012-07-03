#requirespackage 'PETSC_HAVE_HYPRE'
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = mhyp.c
SOURCEF  =
SOURCEH  = mhyp.h
OBJSC    = $(SOURCEC:.c=.o)
OBJSF    =
LIBBASE  = libpetscdm
MANSEC   = DM
SUBMANSEC= DMDA
DIRS     =
LOCDIR   = src/dm/impls/da/hypre/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
