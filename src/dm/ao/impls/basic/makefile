# $Id: makefile,v 1.9 1997/09/26 02:21:56 bsmith Exp bsmith $

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"' 
SOURCEC  = aobasic.c aodatabasic.c aoreduced.c
SOURCEF  =
SOURCEH  = 
OBJSC    = aobasic.o aodatabasic.o aoreduced.o
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = 
MANSEC   = 6
OCDIR    = src/ao/impls/basic/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

