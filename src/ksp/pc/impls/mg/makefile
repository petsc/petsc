# $Id: makefile,v 1.32 2001/08/22 18:01:28 balay Exp $ 

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = mg.c fmg.c smg.c mgfunc.c 
SOURCEF  =
SOURCEH  = mgimpl.h ../../../../../include/petscmg.h
OBJSC    = mg.o fmg.o smg.o mgfunc.o
OBJSF    =
LIBBASE  = libpetscksp
MANSEC   = PC
LOCDIR   = src/ksp/pc/impls/mg/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
