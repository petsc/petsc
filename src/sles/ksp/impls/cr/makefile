# $Id: makefile,v 1.25 1997/09/10 18:45:34 balay Exp balay $ 

ALL: lib 

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
FFLAGS   =
SOURCEC  = cr.c 
SOURCEF  = 
OBJSC    = cr.o 
OBJSF    =
LIBBASE  = libpetscsles
LOCDIR   = src/ksp/impls/cr/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
