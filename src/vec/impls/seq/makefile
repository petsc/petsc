# $Id: makefile,v 1.37 1999/02/02 21:20:36 balay Exp bsmith $ 

ALL: lib

CFLAGS   = ${AMS_INCLUDE}
FFLAGS   =
SOURCEC  = bvec2.c bvec1.c dvec2.c vseqcr.c
SOURCEF  =
SOURCEH  = 
OBJSC    = bvec2.o  bvec1.o dvec2.o vseqcr.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/impls/seq/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test

