# $Id: makefile,v 1.23 1997/08/22 15:15:29 bsmith Exp balay $ 

ALL: lib

CFLAGS   =  ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = sles.c 
SOURCEF  = 
SOURCEH  = 
OBJSC    = sles.o 
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = 3
LOCDIR   = src/sles/interface/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
