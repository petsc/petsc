# $Id: makefile,v 1.19 2001/08/22 18:02:18 balay Exp $

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = ts.c tscreate.c tsreg.c tsregall.c tsfd.c dlregis.c
SOURCEF  =
SOURCEH  = 
OBJSC    = ${SOURCEC:.c=.o}
OBJSF    =
LIBBASE  = libpetscts
MANSEC   = TS
LOCDIR   = src/ts/interface/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

