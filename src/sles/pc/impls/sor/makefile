PETSC_DIR = ../../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' -I../.. 
SOURCEC  = sor.c
SOURCEF  =
SOURCEH  = 
OBJSC    = sor.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4
LOCDIR   = src/pc/impls/sor/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
