#
#   I'd like to add the dependancies pvec.c depends on pdvec.c ...
#  but can't figure out how.
#
ITOOLSDIR = ../../..

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I.. -I../.. \
           -I$(ITOOLSDIR)/pinclude $(CONF) $(COPT)
SOURCEC  = pvec.c pbvec.c
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = pvec.o pbvec.o 
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)
MANSEC   = 1

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
