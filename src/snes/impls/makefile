IPETSCDIR = ../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I$(IPETSCDIR)/pinclude \
           -I../.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =  
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    =  
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscsnes
LINCLUDE = $(SOURCEH)
DIRS     = ls tr test umtr

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)


