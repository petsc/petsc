#PETSC_DIR = ../../..

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

