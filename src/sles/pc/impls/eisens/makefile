PETSC_DIR = ../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../.. \
           $(CONF) -D__DIR__='"pc/impls/eisen/"'
SOURCEC  = eisen.c
SOURCEF  =
SOURCEH  = 
OBJSC    = eisen.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
