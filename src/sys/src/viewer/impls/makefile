IPETSCDIR = ../../..

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I$(IPETSCDIR)/pinclude \
           -I../.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =  
SOURCEF  =
SOURCEH  = 
OBJSC    =  
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = matlab file

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

