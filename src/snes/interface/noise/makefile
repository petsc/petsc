# $Id: makefile,v 1.3 1998/03/24 17:36:27 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = snesmfj2.c snesnoise.c
SOURCEF  = snesdnest.F
SOURCEH  = 
OBJSC    = snesmfj2.o snesnoise.o
OBJSF    = snesdnest.o
LIBBASE  = libpetscsnes
MANSEC   = n
LOCDIR   = src/snes/interface/noise

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

