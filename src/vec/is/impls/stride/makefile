PETSC_DIR = ../../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../../.. -I$(PETSC_DIR) $(PCONF)\
           $(CONF) -D__DIR__='"is/impls/seq/stride/"'
SOURCEC  = stride.c
SOURCEF  =
SOURCEH  = 
OBJSC    = stride.o 
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 6
LOCDIR   = src/is/impls/seq/stride/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

