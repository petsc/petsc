# $Id: makefile,v 1.37 2000/08/17 04:52:25 bsmith Exp balay $

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

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
