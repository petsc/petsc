PETSC_DIR = ../../..

ALL: lib

CFLAGS   = $(PETSC_INCLUDE) -I.. -I$(PETSC_DIR) $(PCONF) \
           $(CONF) -D__DIR__='"is/interface/"'
SOURCEC  = index.c 
SOURCEF  = 
SOURCEH  = 
OBJSC    = index.o 
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 6

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
