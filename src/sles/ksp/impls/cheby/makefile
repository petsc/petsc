# $Id: makefile,v 1.29 1998/03/19 02:05:33 balay Exp balay $ 

ALL: lib 

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = cheby.c 
SOURCEF  =
SOURCEH  = chebctx.h
OBJSC    = cheby.o 
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = 5
LOCDIR   = src/ksp/impls/cheby/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
