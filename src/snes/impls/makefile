IPETSCDIR = ../../..

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I$(IPETSCDIR)/pinclude \
           -I../.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =  
SOURCEF  =
SOURCEH  = 
OBJSC    =  
OBJSF    =
LIBBASE  = libpetscsnes
DIRS     = ls tr test umtr

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)


