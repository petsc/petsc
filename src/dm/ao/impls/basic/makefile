# $Id: makefile,v 1.8 1997/08/22 15:19:36 bsmith Exp bsmith $

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"' 
SOURCEC  = aobasic.c aodatabasic.c
SOURCEF  =
SOURCEH  = 
OBJSC    = aobasic.o aodatabasic.o
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = 
MANSEC   = 6
OCDIR    = src/ao/impls/basic/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

