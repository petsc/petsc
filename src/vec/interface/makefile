# $Id: makefile,v 1.179 1997/08/15 16:24:04 bsmith Exp bsmith $

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = vector.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = vector.o 
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 1
LOCDIR   = src/vec/interface/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

