# $Id: makefile,v 1.25 1997/09/10 18:45:51 balay Exp balay $ 

ALL: lib 


CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = preonly.c
SOURCEF  =
OBJSC    = preonly.o
OBJSF    =
LIBBASE  = libpetscsles
LOCDIR   = src/ksp/impls/preonly/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
