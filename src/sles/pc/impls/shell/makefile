ITOOLSDIR = ../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I$(ITOOLSDIR)/pinclude \
           -I../..   $(CONF)
SOURCEC  = shell.c 
SOURCEF  =
WSOURCEC = shellf.c
SOURCEH  = 
OBJSC    = shell.o 
WOBJS    = shellf.o
OBJSF    =
LIBBASE  = libpetscpc
LINCLUDE = $(SOURCEH)
MANSEC   = 4

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
