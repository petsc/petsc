# $Id: makefile,v 1.33 1997/09/10 18:53:34 balay Exp $ 

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = pbvec.c pdvec.c pvec2.c
SOURCEF  =
SOURCEH  = pvecimpl.h 
OBJSC    = pbvec.o pdvec.o pvec2.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 1
LOCDIR   = src/vec/impls/mpi/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

