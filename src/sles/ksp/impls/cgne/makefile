# $Id: makefile,v 1.38 2001/08/22 18:01:39 balay Exp $

ALL: lib 

CFLAGS   =
FFLAGS   =
SOURCEC  = cgne.c
# cgeig.c cgtype.c
SOURCEF  =
SOURCEH  =
# cgctx.h 
OBJSC    = cgne.o
# cgeig.o cgtype.o
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = KSP
LOCDIR   = src/sles/ksp/impls/cgne/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
