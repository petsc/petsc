# $Id: makefile,v 1.179 1997/08/15 16:24:04 bsmith Exp bsmith $ 

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  =
SOURCEF  =
SOURCEH  = drawimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscsys
DIRS     = interface impls examples utils
LOCDIR   = src/draw/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
