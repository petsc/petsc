IPETSCDIR = ../../../../

ALL: lib

# This matrix dependency should be fixed!
CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../.. -I$(IPETSCDIR)/pinclude \
           $(BS_INCLUDE) -I../../../../src/mat \
           $(CONF) -D__DIR__='"pc/impls/icc/"'
SOURCEC  = icc.c iccbs.c
SOURCEF  =
WSOURCEC = 
SOURCEH  = icc.h
OBJSC    = icc.o iccbs.o
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscsles
LINCLUDE = $(SOURCEH)
DIRS     = 
MANSEC   = 4

runexamples:

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
