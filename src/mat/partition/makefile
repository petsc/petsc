# $Id: makefile,v 1.1 1997/09/24 20:32:41 bsmith Exp bsmith $ 
#
ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"' 
SOURCEC  = partition.c spartition.c
SOURCEH  = 
OBJSC    = partition.o spartition.o
DOCS     = 
LIBBASE  = libpetscmat
LOCDIR   = src/mat/impls/partition/
MANSEC   = 2

runexamples:

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
