# $Id: makefile,v 1.25 1997/09/10 18:51:36 balay Exp $ 

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = jacobi.c
SOURCEF  =
SOURCEH  = 
OBJSC    = jacobi.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4
LOCDIR   = src/pc/impls/jacobi/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
