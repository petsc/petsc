# $Id: makefile,v 1.23 1997/08/22 15:15:29 bsmith Exp $ 

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = sda2.c sda2f.c
SOURCEF  =
SOURCEH  = sda.h
OBJSC    = sda2.o sda2f.o
OBJSF    =
LIBBASE  = libpetsccontrib
DIRS     = 
LOCDIR   = src/contrib/sda/src

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

