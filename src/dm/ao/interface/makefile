# $Id: makefile,v 1.179 1997/08/15 16:24:04 bsmith Exp bsmith $ 

ALL: lib

CFLAGS   = $(CPPFLAGS)  $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = ao.c 
SOURCEF  = 
SOURCEH  = 
OBJSC    = ao.o 
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 6
LOCDIR   = src/ao/interface/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
