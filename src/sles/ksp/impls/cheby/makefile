ALL: lib 
IPETSCDIR = ../../../..

CFLAGS   = $(OPT) $(CONF) $(INCLUDEDIRS)  \
           -I$(IPETSCDIR)/pinclude -I../../ -D__DIR__='"ksp/impls/cheby/"'
SOURCEC  = cheby.c 
SOURCEF  =
SOURCEH  = chebctx.h
WSOURCE = chebyf.c
OBJSC    = cheby.o 
OBJSF    =
WOBJS    = chebyf.o
LIBBASE  = libpetscksp
MANSEC   = 5

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
