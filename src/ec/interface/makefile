# $Id: makefile,v 1.179 1997/08/15 16:24:04 bsmith Exp bsmith $ 

ALL: lib

CFLAGS   = ${CPPFLAGS}  -D__SDIR__='"${LOCDIR}"'
SOURCEC  = ec.c ecall.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = ec.o ecall.o
LIBBASE  = libpetscec
DIRS     =
MANSEC   = e
LOCDIR   = src/ec/interface/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
