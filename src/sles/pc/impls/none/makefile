PETSC_DIR = ../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../.. $(PCONF)\
           $(CONF) -D__DIR__='"pc/impls/none/"'
SOURCEC  = none.c
SOURCEF  =
SOURCEH  = 
OBJSC    = none.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
