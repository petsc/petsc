# $Id: makefile,v 1.179 1997/08/15 16:24:04 bsmith Exp bsmith $ 

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  =
SOURCEF  =
SOURCEH  = tsimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscts
DIRS     = interface impls  examples
LOCDIR   = src/ts/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

