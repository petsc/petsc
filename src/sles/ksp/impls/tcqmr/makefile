# $Id: makefile,v 1.27 1997/09/10 18:46:02 balay Exp balay $ 

ALL: lib 

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = tcqmr.c
SOURCEF  =
SOURCEH  = tcqmrp.h
OBJSC    = tcqmr.o
OBJSF    = 
LIBBASE  = libpetscsles
LOCDIR   = src/ksp/impls/tcqmr/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
