# $Id: makefile,v 1.28 1997/09/10 18:46:12 balay Exp balay $ 

ALL: lib 

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = cheby.c 
SOURCEF  =
SOURCEH  = chebctx.h
OBJSC    = cheby.o 
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = 5
LOCDIR   = src/ksp/impls/cheby/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
