IPETSCDIR = ../../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../../.. -I$(IPETSCDIR) \
           $(CONF) -D__DIR__='"is/impls/seq/stride/"'
SOURCEC  = stride.c
SOURCEF  =
SOURCEH  = 
OBJSC    = stride.o 
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 6

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

