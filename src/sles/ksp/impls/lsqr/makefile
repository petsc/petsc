# $Id: makefile,v 1.25 1997/09/10 18:45:47 balay Exp balay $ 

ALL: lib 

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = lsqr.c
SOURCEF  =
OBJSC    = lsqr.o
OBJSF    =
LIBBASE  = libpetscsles
LOCDIR   = src/ksp/impls/lsqr/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
