ALL: lib 
IPETSCDIR = ../../../../

CFLAGS   = $(OPT) $(INCLUDEDIRS) $(CONF)  \
           -I$(IPETSCDIR)/pinclude -I../.. -D__DIR__='"ksp/impls/qcg/"'
FFLAGS   =
SOURCEC  = qcg.c
SOURCEF  =
SOURCEH  = qcg.h 
WSOURCEC = 
OBJSC    = qcg.o
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscksp
LINCLUDE = 

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
