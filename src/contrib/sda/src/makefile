PETSC_DIR = ../../../..

ALL: lib

CFLAGS   = $(PCONF) $(PETSC_INCLUDE) -D__DIR__='"contrib/sda/src"' $(CONF) 
SOURCEC  = sda2.c sda2f.c
SOURCEF  =
SOURCEH  = sda.h
OBJSC    = sda2.o sda2f.o
OBJSF    =
LIBBASE  = libpetsccontrib
DIRS     = 
LOCDIR   = src/contrib/sda/src

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

