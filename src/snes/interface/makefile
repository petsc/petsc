# $Id: makefile,v 1.36 1998/03/24 17:37:22 balay Exp balay $

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = snes.c snesj.c snesmfj.c snesregi.c snesut.c snesj2.c dlregis.c
SOURCEF  =
SOURCEH  = 
OBJSC    = snesregi.o snes.o snesj.o snesmfj.o snesut.o snesj2.o dlregis.o
OBJSF    =
LIBBASE  = libpetscsnes
MANSEC   = n
DIRS     = 
LOCDIR   = src/snes/interface/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

