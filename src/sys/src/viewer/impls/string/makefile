# $Id: makefile,v 1.10 1997/09/10 19:01:23 balay Exp balay $ 

ALL: lib 
CFLAGS    = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"' 
SOURCEC   = stringv.c
SOURCEF   =
SOURCEH   =
OBJSC     = stringv.o
OBJSF     =
MANSEC    = v
LIBBASE   = libpetscsys
DIRS      = 
LOCDIR    = src/viewer/impls/string/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
