# $Id: makefile,v 1.26 1997/08/22 15:09:54 bsmith Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = index.c 
SOURCEF  = 
SOURCEH  = 
OBJSC    = index.o 
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 6
LOCDIR   = src/is/interface/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

