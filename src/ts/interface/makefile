# $Id: makefile,v 1.11 1998/03/19 02:11:06 balay Exp balay $

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = ts.c tsreg.c tsregall.c tsfd.c dlregis.c
SOURCEF  =
SOURCEH  = 
OBJSC    = ts.o tsreg.o tsregall.o tsfd.o dlregis.o
OBJSF    =
LIBBASE  = libpetscts
MANSEC   = t
LOCDIR   = src/ts/interface/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

