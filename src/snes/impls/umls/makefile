# $Id: makefile,v 1.18 1997/09/10 19:02:44 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = umls.c search.c
SOURCEF  = 
SOURCEH  = umls.h
OBJSC    = umls.o search.o
OBJSF    = 
LIBBASE  = libpetscsnes
MANSEC   = n
LOCDIR   = src/snes/impls/umls/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

