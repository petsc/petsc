# $Id: makefile,v 1.23 1997/08/22 15:15:29 bsmith Exp $ 

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = aij.c aijfact.c ij.c essl.c dxml.c aijnode.c fdaij.c
SOURCEF  =
SOURCEH  = aij.h
OBJSC    = aij.o aijfact.o ij.o essl.o dxml.o aijnode.o fdaij.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = 2
LOCDIR   = src/mat/impls/aij/seq/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
