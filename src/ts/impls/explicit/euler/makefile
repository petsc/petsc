# $Id: makefile,v 1.23 1997/08/22 15:15:29 bsmith Exp $ 

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' 
SOURCEC  = euler.c
SOURCEF  =
SOURCEH  = 
OBJSC    = euler.o
OBJSF    =
LIBBASE  = libpetscts
MANSEC   = t
LOCDIR   = src/ts/impls/explicit/euler/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

