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
	-@$(OMAKE) BOPT=$(BOPT) PARCH=$(PARCH) ACTION=libfast  tree 

ranlib:
	$(RANLIB) $(LDIR)/$(COMPLEX)/*.a

deletelibs:
	-$(RM) $(LDIR)/*.o $(LDIR)/*.a $(LDIR)/complex/*

deletemanpages:
	$(RM) -f $(PETSCLIB)/docs/man/man*/*

deletewwwpages:
	$(RM) -f $(PETSCLIB)/docs/www/man*/*

deletelatexpages:
	$(RM) -f $(PETSCLIB)/docs/rsum/*sum*.tex
