# $Id: makefile,v 1.23 1997/08/22 15:15:29 bsmith Exp $ 

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = eisen.c
SOURCEF  =
SOURCEH  = 
OBJSC    = eisen.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4
LOCDIR   = src/pc/impls/eisen/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
