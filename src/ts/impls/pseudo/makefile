# $Id: makefile,v 1.23 1997/08/22 15:15:29 bsmith Exp $ 

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = posindep.c
SOURCEF  =
SOURCEH  = 
OBJSC    = posindep.o
OBJSF    =
LIBBASE  = libpetscts
MANSEC   = t
LOCDIR   = src/ts/impls/pseudo/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

