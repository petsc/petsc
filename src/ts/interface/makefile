# $Id: makefile,v 1.8 1997/08/22 15:16:35 bsmith Exp bsmith $

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = ts.c tsreg.c tsregall.c tsfd.c
SOURCEF  =
SOURCEH  = 
OBJSC    = ts.o tsreg.o tsregall.o tsfd.o
OBJSF    =
LIBBASE  = libpetscts
MANSEC   = t
LOCDIR   = src/ts/interface/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

