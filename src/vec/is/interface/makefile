IPETSCDIR = ../../..

ALL: lib

CFLAGS   = $(PETSC_INCLUDE) -I.. -I$(IPETSCDIR) \
           $(CONF) -D__DIR__='"is/src/"'
SOURCEC  = index.c 
SOURCEF  = 
SOURCEH  = 
OBJSC    = index.o 
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 6

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
