# $Id: makefile,v 1.179 1997/08/15 16:24:04 bsmith Exp bsmith $

ALL: lib

CFLAGS   =  $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"' 
SOURCEC  = stride.c
SOURCEF  =
SOURCEH  = 
OBJSC    = stride.o 
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 6
LOCDIR   = src/is/impls/seq/stride/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

