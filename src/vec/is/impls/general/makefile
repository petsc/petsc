PETSC_DIR = ../../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../../.. -I$(PETSC_DIR) $(PCONF) \
           $(CONF) -D__DIR__='"is/impls/seq/general/"'
SOURCEC  = general.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = general.o 
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 6
LOCDIR   = src/is/impls/seq/general/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

