
#requirespackage 'PETSC_HAVE_RAMG'
#requiresprecision double
#requiresscalar real 

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = petscfunc.c  ramgpetsc.c
SOURCEF  = 
SOURCEH  = petscfunc.h  ramgfunc.h
OBJSC    = petscfunc.o  ramgpetsc.o
OBJSF    =
LIBBASE  = libpetscksp
MANSEC   = PC
LOCDIR   = src/ksp/pc/impls/ramg/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
