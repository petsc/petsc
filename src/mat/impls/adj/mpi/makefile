# $Id: makefile,v 1.4 1997/09/10 18:48:33 balay Exp $ 

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = adj.c
SOURCEF  =
SOURCEH  = adj.h
OBJSC    = adj.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = 2
LOCDIR   = src/mat/impls/adj/seq/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
