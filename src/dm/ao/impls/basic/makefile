# $Id: makefile,v 1.11 1997/12/06 15:16:09 bsmith Exp balay $

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = aobasic.c aodatabasic.c aoreduced.c
SOURCEF  =
SOURCEH  = 
OBJSC    = aobasic.o aodatabasic.o aoreduced.o
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = 
MANSEC   = 6
LOCDIR   = src/ao/impls/basic/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

