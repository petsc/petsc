# $Id: makefile,v 1.24 1998/03/19 02:06:35 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' ${BS_INCLUDE}
SOURCEC  = ilu.c
SOURCEF  =
SOURCEH  = ilu.h
OBJSC    = ilu.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4
LOCDIR   = src/pc/impls/ilu/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
