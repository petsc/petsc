# $Id: makefile,v 1.179 1997/08/15 16:24:04 bsmith Exp bsmith $

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  =  
SOURCEF  =
SOURCEH  = dvecimpl.h
OBJSC    =  
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = seq mpi
LOCDIR   = src/vec/impls/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

