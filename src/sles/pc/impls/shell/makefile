# $Id: makefile,v 1.24 1997/09/10 18:51:53 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = shellpc.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = shellpc.o 
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = 4
LOCDIR   = src/pc/impls/shell/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
