IPETSCDIR = ../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../..\
           $(CONF) -D__DIR__='"pc/impls/none/"'
SOURCEC  = none.c
SOURCEF  =
SOURCEH  = 
OBJSC    = none.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
