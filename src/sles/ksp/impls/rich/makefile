# $Id: makefile,v 1.23 1997/08/22 15:15:29 bsmith Exp $ 

ALL: lib 

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = rich.c richscale.c
SOURCEF  =
SOURCEH  = richctx.h
OBJSC    = rich.o richscale.o
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = 5
LOCDIR   = src/ksp/impls/rich/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
