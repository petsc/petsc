IPETSCDIR = ../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../.. -I$(IPETSCDIR)/pinclude\
           $(CONF) -D__DIR__='"pc/impls/ilu/"'
SOURCEC  = ilu.c
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = ilu.o
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscsles
LINCLUDE = $(SOURCEH)
DIRS     = 
MANSEC   = 4

runexamples:

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
