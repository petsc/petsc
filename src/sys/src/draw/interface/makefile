IPETSCDIR = ../../..

ALL: lib

CFLAGS   = $(PETSC_INCLUDE) -I..  \
           $(CONF) -D__DIR__='"draw/src/"'
SOURCEC  = draw.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = draw.o 
OBJSF    =
LIBBASE  = libpetscdraw
MANSEC   = x

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
