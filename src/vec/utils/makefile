ITOOLSDIR = ../../

ALL: lib

CFLAGS   = $(OPT) -I$(ITOOLSDIR)/include -I.. -I$(ITOOLSDIR)/pinclude\
           $(COPT) $(CONF)
SOURCEC  = vinv.c gcreate.c vscat.c
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = vinv.o gcreate.o vscat.o
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)
DIRS     = complex
MANSEC   = 1

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
