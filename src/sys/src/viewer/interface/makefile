PETSC_DIR = ../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF)  -D__DIR__='"$(LOCDIR)"'
SOURCEC  = view.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = view.o 
OBJSF    =
LIBBASE  = libpetscsys
MANSEC   = v
LOCDIR   = src/viewer/interface/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

