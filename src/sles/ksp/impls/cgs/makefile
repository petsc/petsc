# $Id: makefile,v 1.179 1997/08/15 16:24:04 bsmith Exp bsmith $

ALL: lib 

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = cgs.c
SOURCEF  =
OBJSC    = cgs.o
OBJSF    = 
LIBBASE  = libpetscsles
LOCDIR   = src/ksp/impls/cgs/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
