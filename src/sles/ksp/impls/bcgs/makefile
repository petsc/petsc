ALL: lib 
IPETSCDIR = ../../../../

CFLAGS   = $(OPT) $(INCLUDEDIRS) $(CONF)  \
           -I$(IPETSCDIR)/pinclude -I../../ -D__DIR__='"ksp/impls/bcgs/"'
SOURCEC  = bcgs.c
WSOURCEC = 
SOURCEF  =
OBJSC    = bcgs.o
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscksp
LINCLUDE = 

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
