# $Id: makefile,v 1.23 1997/08/22 15:15:29 bsmith Exp $ 

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"' 
SOURCEC  = shell.c shellcnv.c
SOURCEF  =
SOURCEH  = 
OBJSC    = shell.o shellcnv.o
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2
LOCDIR   = src/mat/impls/shell/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
