# $Id: makefile,v 1.2 1997/10/01 22:45:42 bsmith Exp bsmith $ 
#
ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"' $(PARMETIS_INCLUDE)
SOURCEC  = partition.c spartition.c pmetis.c
SOURCEH  = 
OBJSC    = partition.o spartition.o pmetis.o
DOCS     = 
LIBBASE  = libpetscmat
LOCDIR   = src/mat/impls/partition/
MANSEC   = 2

runexamples:

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
