# $Id: makefile,v 1.23 1998/03/19 02:06:38 balay Exp balay $ 

ALL: lib

# This matrix dependency should be fixed!
CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' ${BS_INCLUDE} 
SOURCEC  = icc.c
SOURCEF  =
SOURCEH  = icc.h
OBJSC    = icc.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4
LOCDIR   = src/pc/impls/icc/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
