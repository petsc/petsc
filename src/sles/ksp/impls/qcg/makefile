# $Id: makefile,v 1.16 1997/09/10 18:46:20 balay Exp balay $ 

ALL: lib 

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
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
