# $Id: makefile,v 1.36 1999/02/02 21:29:16 balay Exp bsmith $

ALL: lib 

CFLAGS   =
FFLAGS   =
SOURCEC  = cg.c cgeig.c cgtype.c
SOURCEF  =
SOURCEH  = cgctx.h 
OBJSC    = cg.o cgeig.o cgtype.o
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = KSP
LOCDIR   = src/sles/ksp/impls/cg/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test
