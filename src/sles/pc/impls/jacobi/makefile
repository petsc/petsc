# $Id: makefile,v 1.26 1998/03/19 02:06:14 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = jacobi.c
SOURCEF  =
SOURCEH  = 
OBJSC    = jacobi.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4
LOCDIR   = src/pc/impls/jacobi/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
