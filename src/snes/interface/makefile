# $Id: makefile,v 1.33 1998/03/06 00:18:38 bsmith Exp balay $

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = snes.c snesj.c snesmfj.c snesregi.c snesut.c snesj2.c dlregis.c
SOURCEF  =
SOURCEH  = 
OBJSC    = snesregi.o snes.o snesj.o snesmfj.o snesut.o snesj2.o dlregis.o
OBJSF    =
LIBBASE  = libpetscsnes
MANSEC   = n
LOCDIR   = src/snes/interface/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

