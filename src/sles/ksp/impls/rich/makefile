# $Id: makefile,v 1.30 1997/09/10 18:45:56 balay Exp balay $ 

ALL: lib 

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = rich.c richscale.c
SOURCEF  =
SOURCEH  = richctx.h
OBJSC    = rich.o richscale.o
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = 5
LOCDIR   = src/ksp/impls/rich/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
