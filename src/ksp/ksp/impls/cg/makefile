# $Id: makefile,v 1.38 2001/08/22 18:01:39 balay Exp $

ALL: lib 

CFLAGS   =
FFLAGS   =
SOURCEC  = cg.c cgeig.c cgtype.c
SOURCEF  =
SOURCEH  = cgctx.h 
OBJSC    = cg.o cgeig.o cgtype.o
OBJSF    =
LIBBASE  = libpetscksp
MANSEC   = KSP
LOCDIR   = src/ksp/ksp/impls/cg/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
