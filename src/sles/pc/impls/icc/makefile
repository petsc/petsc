# $Id: makefile,v 1.22 1997/09/10 18:52:14 balay Exp balay $ 

ALL: lib

# This matrix dependency should be fixed!
CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'   \
           ${BS_INCLUDE} 
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
