# $Id: makefile,v 1.26 1998/03/19 02:05:36 balay Exp balay $ 

ALL: lib 

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = preonly.c
SOURCEF  =
OBJSC    = preonly.o
OBJSF    =
LIBBASE  = libpetscsles
LOCDIR   = src/ksp/impls/preonly/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
