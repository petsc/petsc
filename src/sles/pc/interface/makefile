# $Id: makefile,v 1.27 1998/03/19 02:06:07 balay Exp balay $ 

ALL: lib

CFLAGS   =  ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = precon.c pcset.c pcregis.c pcnull.c
SOURCEF  =
SOURCEH  = 
OBJSC    = precon.o pcset.o pcregis.o pcnull.o
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = 4
LOCDIR   = src/pc/interface/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
