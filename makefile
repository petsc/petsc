ITOOLSDIR = ./

CFLAGS   = $(OPT) -I$(ITOOLSDIR)/include -I.. -I$(ITOOLSDIR) $(CONF)
SOURCEC  =
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)
DIRS     = is vec ksp sys

include $(ITOOLSDIR)/bmake/$(PARCH)

all:
	-cd sys; make BOPT=$(BOPT) PARCH=$(PARCH) workers
	-@$(MAKE) BOPT=$(BOPT) PARCH=$(PARCH) libfasttree 


deletelibs:
	-$(RM) -f $(LDIR)/*.o $(LDIR)/*.a $(LDIR)/complex/*
