# $Id: makefile,v 1.179 1997/08/15 16:24:04 bsmith Exp bsmith $ 

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  = matrix.c mhas.c fdmatrix.c matcnv.c mgtype.c
SOURCEF  =
SOURCEH  = 
OBJSC    = matrix.o mhas.o fdmatrix.o matcnv.o mgtype.o
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2
LOCDIR   = src/mat/interface/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
