IPETSCDIR = ../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I$(IPETSCDIR)/pinclude \
           -I../..   $(CONF) -D__DIR__='"pc/impls/shell"'
SOURCEC  = shell.c 
SOURCEF  =
WSOURCEC = shellf.c
SOURCEH  = 
OBJSC    = shell.o 
WOBJS    = shellf.o
OBJSF    =
LIBBASE  = libpetscsles
LINCLUDE = $(SOURCEH)
MANSEC   = 4

runexamples:

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
