PETSC_DIR = ../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../.. $(BS_INCLUDE) -I../.. \
           $(CONF) -I../../../../src/mat -D__DIR__='"pc/impls/ilu/"' 
SOURCEC  = ilu.c
SOURCEF  =
SOURCEH  = 
OBJSC    = ilu.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
