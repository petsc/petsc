ALL: lib 
IPETSCDIR = ../../../../

CFLAGS   =  $(PETSC_INCLUDE) $(CONF)  \
           -I../.. -D__DIR__='"ksp/impls/qcg/"'
FFLAGS   =
SOURCEC  = qcg.c
SOURCEF  =
SOURCEH  = qcg.h 
OBJSC    = qcg.o
OBJSF    =
LIBBASE  = libpetscksp
LINCLUDE = 

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
