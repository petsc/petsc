# $Id: makefile,v 1.6 1997/08/22 15:19:32 bsmith Exp bsmith $ 

ALL: lib

CFLAGS   = $(CPPFLAGS)  $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = ao.c  aodata.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = ao.o aodata.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 6
LOCDIR   = src/ao/interface/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
