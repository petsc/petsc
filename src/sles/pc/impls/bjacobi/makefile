PETSC_DIR = ../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../..\
           $(CONF) -I../../../mat/impls/aij/seq \
           -I../../../mat/impls/row/seq \
           -I../../../mat/impls/bdiag/seq -I../../../mat\
           -D__DIR__='"pc/impls/bjacobi/"'
SOURCEC  = bjacobi.c
SOURCEF  =
SOURCEH  = bjacobi.h
OBJSC    = bjacobi.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
