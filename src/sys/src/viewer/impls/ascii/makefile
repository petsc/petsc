# $Id: makefile,v 1.24 1997/09/10 19:01:06 balay Exp balay $ 

ALL: lib 
CFLAGS    = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"' 
SOURCEC   = filev.c
SOURCEF   =
SOURCEH   =
OBJSC     = filev.o
OBJSF     =
MANSEC    = v
LIBBASE   = libpetscsys
DIRS      = 
LOCDIR    = src/viewer/impls/file/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
