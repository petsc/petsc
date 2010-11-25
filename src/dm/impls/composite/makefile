
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = pack.c packm.c
SOURCEF  =
SOURCEH  = packimpl.h
OBJSC    = $(SOURCEC:.c=.o)
OBJSF    =
LIBBASE  = libpetscdm
MANSEC   = DM
DIRS     = ftn-custom f90-custom
LOCDIR   = src/dm/impls/composite/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
