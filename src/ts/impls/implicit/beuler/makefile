# $Id: makefile,v 1.23 1997/08/22 15:15:29 bsmith Exp $ 

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"' 
SOURCEC  = beuler.c
SOURCEF  =
SOURCEH  = 
OBJSC    = beuler.o
OBJSF    =
LIBBASE  = libpetscts
MANSEC   = t
LOCDIR   = src/ts/impls/implicit/beuler/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

