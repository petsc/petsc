# $Id: makefile,v 1.23 1997/08/22 15:15:29 bsmith Exp $ 

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = snesmfj2.c snesnoise.c
SOURCEF  = snesdnest.F
SOURCEH  = 
OBJSC    = snesmfj2.o snesnoise.o
OBJSF    = snesdnest.o
LIBBASE  = libpetscsnes
MANSEC   = n
LOCDIR   = src/snes/interface/noise

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

