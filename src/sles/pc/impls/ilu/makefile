PETSC_DIR = ../../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' \
           $(BS_INCLUDE)
SOURCEC  = ilu.c
SOURCEF  =
SOURCEH  = 
OBJSC    = ilu.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4
LOCDIR   = src/pc/impls/ilu/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
