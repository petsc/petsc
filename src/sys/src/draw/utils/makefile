IPETSCDIR = ../../..

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../impls -I.. -I$(IPETSCDIR)/pinclude\
           $(COPT) $(CONF) -D__DIR__='"draw/utils/"'
SOURCEC  = axis.c lg.c
SOURCEF  =
WSOURCE = axisf.c lgf.c
SOURCEH  = 
OBJSC    = axis.o lg.o
WOBJS    = axisf.o lgf.o
OBJSF    =
LIBBASE  = libpetscdraw
MANSEC   = x

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
