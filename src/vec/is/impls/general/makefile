IPETSCDIR = ../../../../..

ALL: lib

CFLAGS   = $(OPT) $(PETSC_INCLUDE) -I../../.. -I$(IPETSCDIR) \
           -I$(IPETSCDIR)/pinclude $(CONF) -D__DIR__='"is/impls/seq/general/"'
SOURCEC  = general.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = general.o 
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 6

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

