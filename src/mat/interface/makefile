ITOOLSDIR = ../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I$(ITOOLSDIR)/pinclude \
           -I..   $(CONF)
SOURCEC  = matrix.c 
SOURCEF  =
WSOURCEC = matrixf.c
SOURCEH  = 
OBJSC    = matrix.o 
WOBJS    = matrixf.o
OBJSF    =
LIBBASE  = libpetscmat
LINCLUDE = $(SOURCEH)
MANSEC   = 2

include $(ITOOLSDIR)/bmake/$(PARCH)
