ALL: lib 
IPETSCDIR = ../../../../

CFLAGS   = $(OPT) $(CONF) $(INCLUDEDIRS) -I../..  \
           -I$(IPETSCDIR)/pinclude -D__DIR__='"ksp/impls/rich/"'
SOURCEC  = rich.c
WSOURCEC = richf.c
SOURCEF  =
SOURCEH  = richctx.h
OBJSC    = rich.o 
WOBJS    = richf.o 
OBJSF    =
LIBBASE  = libpetscksp
LINCLUDE = 
MANSEC   = 5

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
