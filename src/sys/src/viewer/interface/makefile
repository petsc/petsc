# $Id: makefile,v 1.179 1997/08/15 16:24:04 bsmith Exp bsmith $

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF)  -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = view.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = view.o 
OBJSF    =
LIBBASE  = libpetscsys
MANSEC   = v
LOCDIR   = src/viewer/interface/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

