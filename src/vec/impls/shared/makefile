# $Id: makefile,v 1.1 1997/11/17 21:44:29 bsmith Exp bsmith $ 

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = shvec.c
SOURCEF  =
SOURCEH  = 
OBJSC    = shvec.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 1
LOCDIR   = src/vec/impls/shared/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

