ITOOLSDIR = .

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
DIRS     = is vec ksp sys pc mat sles options draw include/pinclude

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)

all:
	-@$(OMAKE) BOPT=$(BOPT) PARCH=$(PARCH) libfasttree 

ranlib:
	$(RANLIB) $(LDIR)/$(COMPLEX)/*.a

deletelibs:
	-$(RM) $(LDIR)/*.o $(LDIR)/*.a $(LDIR)/complex/*

deletemanpages:
	$(RM) -f man/man*/*

