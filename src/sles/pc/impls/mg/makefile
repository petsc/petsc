# $Id: makefile,v 1.31 2000/11/28 17:30:19 bsmith Exp balay $ 

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

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
