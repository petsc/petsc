
#requirespackage  'PETSC_USE_32BIT_INT'
#requiresscalar real 
#requiresprecision  double                                                                                                                                                             

ALL: lib

CPPFLAGS = 
CFLAGS   =
FFLAGS   =
SOURCEC  = bit_mask.c comm.c error.c gs.c ivec.c  xxt.c xyt.c tfs.c
SOURCEF  =
SOURCEH  = tfs.h 
OBJSC    = bit_mask.o comm.o error.o gs.o ivec.o xxt.o xyt.o tfs.o
OBJSF    =
LIBBASE  = libpetscksp
MANSEC   = PC
LOCDIR   = src/ksp/pc/impls/tfs/

include ${PETSC_DIR}/conf/base
include ${PETSC_DIR}/conf/test
