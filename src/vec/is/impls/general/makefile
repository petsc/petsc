IPETSCDIR = ../../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../../.. -I$(IPETSCDIR) \
           $(CONF) -D__DIR__='"is/impls/seq/general/"'
SOURCEC  = general.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = general.o 
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 6

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

