# $Id: makefile,v 1.38 2000/08/17 04:51:11 bsmith Exp balay $ 

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

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

