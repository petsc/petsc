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
DIRS     = is vec ksp sys pc mat sles options

include $(ITOOLSDIR)/bmake/$(PARCH)

all:
	-cd sys; make BOPT=$(BOPT) PARCH=$(PARCH) workers
	-@$(MAKE) BOPT=$(BOPT) PARCH=$(PARCH) libfasttree 


deletelibs:
	-$(RM) $(LDIR)/*.o $(LDIR)/*.a $(LDIR)/complex/*

deletemanpages:
	$(RM) -f man/man*/*

