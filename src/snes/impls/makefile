IPETSCDIR = ../../..

ALL: lib

CFLAGS   = $(OPT) $(PETSC_INCLUDE) -I$(IPETSCDIR)/pinclude \
           -I../.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =  
SOURCEF  =
SOURCEH  = 
OBJSC    =  
OBJSF    =
LIBBASE  = libpetscsnes
DIRS     = ls tr test umtr umls

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)


