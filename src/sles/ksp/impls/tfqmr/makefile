# $Id: makefile,v 1.27 1998/03/19 02:05:39 balay Exp balay $ 

ALL: lib 

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = tfqmr.c
SOURCEF  =
OBJSC    = tfqmr.o
OBJSF    =
LIBBASE  = libpetscsles
LOCDIR   = src/ksp/impls/tfqmr/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
