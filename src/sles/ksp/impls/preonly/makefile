ALL: lib 
IPETSCDIR = ../../../..

CFLAGS   =  $(CONF) $(PETSC_INCLUDE) \
           -I../.. -D__DIR__='"ksp/impls/preonly/"'
SOURCEC  = preonly.c
SOURCEF  =
OBJSC    = preonly.o
OBJSF    =
LIBBASE  = libpetscksp

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
