# $Id: makefile,v 1.23 1997/08/22 15:15:29 bsmith Exp $ 

ALL: lib

# This matrix dependency should be fixed!
CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'   \
           $(BS_INCLUDE) 
SOURCEC  = icc.c
SOURCEF  =
SOURCEH  = icc.h
OBJSC    = icc.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4
LOCDIR   = src/pc/impls/icc/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
