# $Id: makefile,v 1.23 1997/08/22 15:15:29 bsmith Exp $ 

ALL: lib 


CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = preonly.c
SOURCEF  =
OBJSC    = preonly.o
OBJSF    =
LIBBASE  = libpetscsles
LOCDIR   = src/ksp/impls/preonly/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
