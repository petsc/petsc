ITOOLSDIR = ../../../../

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
LIBBASE  = libpetscmat
LINCLUDE = $(SOURCEH)
MANSEC   = 2

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
