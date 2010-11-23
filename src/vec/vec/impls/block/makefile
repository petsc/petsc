
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = vecblockops.c vecblockinterface.c vecblockcreate.c
SOURCEF  =
SOURCEH  = vecblockprivate.h
OBJSC    = $(SOURCEC:.c=.o)
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/vec/impls/block/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

