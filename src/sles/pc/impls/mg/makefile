# $Id: makefile,v 1.30 2000/08/17 04:52:10 bsmith Exp bsmith $ 

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = mg.c fmg.c smg.c mgfunc.c 
SOURCEF  =
SOURCEH  = mgimpl.h ../../../../../include/petscmg.h
OBJSC    = mg.o fmg.o smg.o mgfunc.o
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = PC
LOCDIR   = src/sles/pc/impls/mg/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test
