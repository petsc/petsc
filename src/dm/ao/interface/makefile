# $Id: makefile,v 1.8 1997/10/03 20:51:07 bsmith Exp bsmith $ 

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
