#PETSC_DIR = ../../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"' 
SOURCEC  = lu.c
SOURCEF  =
SOURCEH  = 
OBJSC    = lu.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4
LOCDIR   = src/pc/impls/lu/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
