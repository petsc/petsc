IPETSCDIR = ../../../..
ALL: lib 

CFLAGS   = $(OPT) $(CONF) $(INCLUDEDIRS) \
           -I$(IPETSCDIR)/pinclude -I../../ -D__DIR__='"ksp/impls/cgs/"'
SOURCEC  = cgs.c
WSOURCE = 
SOURCEF  =
OBJSC    = cgs.o
WOBJS    = 
OBJSF    = 
LIBBASE  = libpetscksp

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
