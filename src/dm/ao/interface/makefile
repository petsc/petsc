# $Id: makefile,v 1.7 1997/09/26 02:21:53 bsmith Exp bsmith $ 

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
