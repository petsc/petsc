IPETSCDIR = ../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../..\
           $(CONF) -D__DIR__='"pc/impls/jacobi/"'
SOURCEC  = jacobi.c
SOURCEF  =
SOURCEH  = 
OBJSC    = jacobi.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
