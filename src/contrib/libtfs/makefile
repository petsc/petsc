# $Id: makefile,v 1.4 2001/08/22 18:02:53 balay Exp $ 

#requiresscalar real 

ALL: lib

#CFLAGS   = -Dr8 -DMPISRC -DSAFE   -DINFO -DDEBUG -DV -O
CFLAGS   = -Dr8 -DMPISRC -DSAFE
FFLAGS   =
SOURCEC  = bit_mask.c bss_malloc.c comm.c error.c gs.c ivec.c queue.c stack.c xxt.c xyt.c 
SOURCEF  =
SOURCEH  = xyt.h xxt.h blas.h queue.h stack.h bss_malloc.h const.h error.h types.h bit_mask.h comm.h
OBJSC    = bit_mask.o bss_malloc.o comm.o error.o gs.o ivec.o queue.o stack.o xxt.o xyt.o 
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = Mat
LOCDIR   = src/contrib/libtfs/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
