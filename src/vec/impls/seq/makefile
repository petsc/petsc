PETSC_DIR = ../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I.. -I../.. \
           -I$(PETSC_DIR)/src/is $(CONF) $(PCONF)\
           -D__DIR__='"vec/impls/seq/"'
SOURCEC  = bvec2.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = bvec2.o 
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 1
LOCDIR   = src/vec/impls/seq/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

