# $Id: makefile,v 1.26 1997/09/10 18:46:07 balay Exp balay $ 

ALL: lib 

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = tfqmr.c
SOURCEF  =
OBJSC    = tfqmr.o
OBJSF    =
LIBBASE  = libpetscsles
LOCDIR   = src/ksp/impls/tfqmr/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
