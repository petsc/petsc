# $Id: makefile,v 1.10 1997/09/10 18:52:22 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = asm.c
SOURCEF  =
SOURCEH  = 
OBJSC    = asm.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4
LOCDIR   = src/pc/impls/asm/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
