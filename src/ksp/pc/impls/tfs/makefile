
#requirespackage  'PETSC_USE_32BIT_INT'
#requiresscalar real 
#requiresprecision  double                                                                                                                                                             

ALL: lib

CPPFLAGS = -DSAFE
CFLAGS   =
FFLAGS   =
SOURCEC  = bit_mask.c comm.c error.c gs.c ivec.c queue.c stack.c xxt.c xyt.c tfs.c
SOURCEF  =
SOURCEH  = xyt.h xxt.h queue.h stack.h const.h error.h types.h bit_mask.h comm.h
OBJSC    = bit_mask.o comm.o error.o gs.o ivec.o queue.o stack.o xxt.o xyt.o tfs.o
OBJSF    =
LIBBASE  = libpetscksp
MANSEC   = PC
LOCDIR   = src/ksp/pc/impls/tfs/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
