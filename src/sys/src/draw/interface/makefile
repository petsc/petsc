IPETSCDIR = ../../..

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I..  \
          -I$(IPETSCDIR)/pinclude $(CONF) -D__DIR__='"draw/src/"'
SOURCEC  = draw.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = draw.o 
OBJSF    =
WSOURCE = drawf.c
WOBJS    = drawf.o
LIBBASE  = libpetscdraw
MANSEC   = x

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
