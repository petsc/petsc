# $Id: makefile,v 1.24 1997/08/22 15:10:34 bsmith Exp bsmith $

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  =  
SOURCEF  =
SOURCEH  = dvecimpl.h
OBJSC    =  
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = seq mpi shared
LOCDIR   = src/vec/impls/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

