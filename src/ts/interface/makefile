# $Id: makefile,v 1.9 1997/10/12 22:17:09 bsmith Exp bsmith $

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = ts.c tsreg.c tsregall.c tsfd.c dlregis.c
SOURCEF  =
SOURCEH  = 
OBJSC    = ts.o tsreg.o tsregall.o tsfd.o dlregis.o
OBJSF    =
LIBBASE  = libpetscts
MANSEC   = t
LOCDIR   = src/ts/interface/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

