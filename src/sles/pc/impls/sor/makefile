PETSC_DIR = ../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../.. $(PCONF) \
           $(CONF) -D__DIR__='"pc/impls/sor/"'
SOURCEC  = sor.c
SOURCEF  =
SOURCEH  = 
OBJSC    = sor.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4
LOCDIR   = src/pc/impls/sor/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
