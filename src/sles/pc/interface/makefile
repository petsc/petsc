# $Id: makefile,v 1.26 1997/08/22 15:12:25 bsmith Exp balay $ 

ALL: lib

CFLAGS   =  ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = precon.c pcset.c pcregis.c pcnull.c
SOURCEF  =
SOURCEH  = 
OBJSC    = precon.o pcset.o pcregis.o pcnull.o
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = 4
LOCDIR   = src/pc/interface/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
