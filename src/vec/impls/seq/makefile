# $Id: makefile,v 1.23 1997/08/22 15:15:29 bsmith Exp $ 

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"' 
SOURCEC  = bvec2.c bvec1.c dvec2.c
SOURCEF  =
SOURCEH  = 
OBJSC    = bvec2.o  bvec1.o dvec2.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 1
LOCDIR   = src/vec/impls/seq/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

