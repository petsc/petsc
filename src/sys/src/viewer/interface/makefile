PETSC_DIR = ../../..

ALL: lib

CFLAGS   = $(OPT) $(PETSC_INCLUDE) -I..  \
           $(CONF) -D__DIR__='"viewer/src/"'
SOURCEC  = view.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = view.o 
OBJSF    =
LIBBASE  = libpetscsys
MANSEC   = 7

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

