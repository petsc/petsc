PETSC_DIR = ../../..

ALL: lib

CFLAGS   = $(OPT) $(PETSC_INCLUDE) -I..  \
           $(CONF) $(PCONF) -D__DIR__='"viewer/interface/"'
SOURCEC  = view.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = view.o 
OBJSF    =
LIBBASE  = libpetscsys
MANSEC   = v
LOCDIR   = src/viewer/interface/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

