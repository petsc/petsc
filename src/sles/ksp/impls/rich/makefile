# $Id: makefile,v 1.31 1998/03/19 02:05:34 balay Exp balay $ 

ALL: lib 

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = rich.c richscale.c
SOURCEF  =
SOURCEH  = richctx.h
OBJSC    = rich.o richscale.o
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = 5
LOCDIR   = src/ksp/impls/rich/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
