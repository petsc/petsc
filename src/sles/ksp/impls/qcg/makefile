# $Id: makefile,v 1.17 1998/03/19 02:05:40 balay Exp balay $ 

ALL: lib 

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
FFLAGS   =
SOURCEC  = qcg.c
SOURCEF  =
SOURCEH  = qcg.h 
OBJSC    = qcg.o
OBJSF    =
LIBBASE  = libpetscsles
LINCLUDE = 
LOCDIR   = src/ksp/impls/qcg/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
