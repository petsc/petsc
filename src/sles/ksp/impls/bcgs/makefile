IPETSCDIR = ../../../..
ALL: lib 

CFLAGS   = $(OPT) $(INCLUDEDIRS) $(CONF)  \
           -I$(IPETSCDIR)/pinclude -I../../ -D__DIR__='"ksp/impls/bcgs/"'
SOURCEC  = bcgs.c
WSOURCE = 
SOURCEF  =
OBJSC    = bcgs.o
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscksp

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
