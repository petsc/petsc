# $Id: makefile,v 1.10 1998/01/17 17:38:11 bsmith Exp balay $

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = ts.c tsreg.c tsregall.c tsfd.c dlregis.c
SOURCEF  =
SOURCEH  = 
OBJSC    = ts.o tsreg.o tsregall.o tsfd.o dlregis.o
OBJSF    =
LIBBASE  = libpetscts
MANSEC   = t
LOCDIR   = src/ts/interface/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

