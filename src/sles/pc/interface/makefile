IPETSCDIR = ../../..

ALL: lib

CFLAGS   = $(PETSC_INCLUDE) -I..  \
           $(CONF) -D__DIR__='"pc/interface/"'
SOURCEC  = precon.c pcset.c pcregis.c
SOURCEF  =
SOURCEH  = 
OBJSC    = precon.o pcset.o pcregis.o
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = 4

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
