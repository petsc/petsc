# $Id: makefile,v 1.2 1997/09/10 19:03:03 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = snesmfj2.c snesnoise.c
SOURCEF  = snesdnest.F
SOURCEH  = 
OBJSC    = snesmfj2.o snesnoise.o
OBJSF    = snesdnest.o
LIBBASE  = libpetscsnes
MANSEC   = n
LOCDIR   = src/snes/interface/noise

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

