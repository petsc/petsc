# $Id: makefile,v 1.40 1999/02/02 21:20:42 balay Exp bsmith $ 

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = pbvec.c pdvec.c pvec2.c pmap.c vmpicr.c
SOURCEF  =
SOURCEH  = pvecimpl.h 
OBJSC    = pbvec.o pdvec.o pvec2.o pmap.o vmpicr.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/impls/mpi/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test
