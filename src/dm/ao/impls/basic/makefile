# $Id: makefile,v 1.10 1997/10/10 04:07:52 bsmith Exp bsmith $

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
LOCDIR   = src/ao/impls/basic/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

