# $Id: makefile,v 1.1 1997/09/23 20:11:18 bsmith Exp bsmith $ 

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = mpiadj.c
SOURCEF  =
SOURCEH  = mpiadj.h
OBJSC    = mpiadj.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = 2
LOCDIR   = src/mat/impls/adj/mpi/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
