PETSC_DIR = ../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../.. $(PCONF) $(CONF) -D__DIR__='"pc/impls/asm/"'
SOURCEC  = asm.c
SOURCEF  =
SOURCEH  = 
OBJSC    = asm.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
