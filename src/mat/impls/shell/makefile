IPETSCDIR = ../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I$(IPETSCDIR)/pinclude \
           -I../..   $(CONF) -D__DIR__='"mat/impls/shell/"'
SOURCEC  = shell.c 
SOURCEF  =
WSOURCEC = shellf.c
SOURCEH  = 
OBJSC    = shell.o 
WOBJS    = shellf.o
OBJSF    =
LIBBASE  = libpetscmat
LINCLUDE = $(SOURCEH)
MANSEC   = 2

runexamples:

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
