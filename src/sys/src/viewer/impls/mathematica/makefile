# $Id: makefile,v 1.2 1999/06/07 20:24:38 knepley Exp $ 

ALL: lib 

CFLAGS    = ${MATHEMATICA_INCLUDE} ${PLAPACK_INCLUDE}
SOURCEC   = mathematica.c
SOURCEH   = mathematica.h
OBJSC     = ${SOURCEC:.c=.o}
MANSEC    = Viewer
LIBBASE   = libpetsc
DIRS      = 
LOCDIR    = src/sys/src/viewer/impls/mathematica/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
