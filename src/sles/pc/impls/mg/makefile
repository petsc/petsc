# $Id: makefile,v 1.29 1999/02/02 21:28:03 balay Exp bsmith $ 

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = mg.c fmg.c smg.c mgfunc.c 
SOURCEF  =
SOURCEH  = mgimpl.h
OBJSC    = mg.o fmg.o smg.o mgfunc.o
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = PC
LOCDIR   = src/sles/pc/impls/mg/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test
