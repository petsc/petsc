# $Id: makefile,v 1.12 1998/03/19 02:15:29 balay Exp balay $

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' 
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

