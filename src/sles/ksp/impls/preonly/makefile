ALL: lib 
IPETSCDIR = ../../../..

CFLAGS   = $(OPT) $(CONF) $(INCLUDEDIRS) \
           -I$(IPETSCDIR)/pinclude -I../.. -D__DIR__='"ksp/impls/preonly/"'
SOURCEC  = preonly.c
WSOURCE = 
SOURCEF  =
OBJSC    = preonly.o
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscksp

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
