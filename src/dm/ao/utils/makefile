# $Id: makefile,v 1.1 1997/11/05 23:24:15 bsmith Exp bsmith $

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"' 
SOURCEC  = aopart.c
SOURCEF  =
SOURCEH  = 
OBJSC    = aopart.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 1
LOCDIR   = src/ao/utils/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

