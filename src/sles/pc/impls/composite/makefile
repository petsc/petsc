# $Id: makefile,v 1.1 1998/01/05 18:40:33 bsmith Exp bsmith $ 

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = composite.c
SOURCEF  =
SOURCEH  = 
OBJSC    = composite.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4
LOCDIR   = src/pc/impls/composite

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
