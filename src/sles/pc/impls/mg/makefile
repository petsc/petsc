# $Id: makefile,v 1.23 1997/09/10 18:52:02 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = mg.c fmg.c smg.c mgfunc.c 
SOURCEF  =
SOURCEH  = mgimpl.h
OBJSC    = mg.o fmg.o smg.o mgfunc.o
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = 4
LOCDIR   = src/pc/impls/mg/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
