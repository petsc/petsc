# $Id: makefile,v 1.32 1997/08/22 15:17:50 bsmith Exp bsmith $

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = snes.c snesj.c snesmfj.c snesregi.c snesut.c snesj2.c dlregis.c
SOURCEF  =
SOURCEH  = 
OBJSC    = snesregi.o snes.o snesj.o snesmfj.o snesut.o snesj2.o dlregis.o
OBJSF    =
LIBBASE  = libpetscsnes
MANSEC   = n
LOCDIR   = src/snes/interface/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

