ITOOLSDIR = ../

CFLAGS   = $(OPT) -I$(ITOOLSDIR)/include -I.. -I$(ITOOLSDIR) $(CONF)
SOURCEC  =
SOURCEF  =
WSOURCEC = 
SOURCEH  = comm.h draw.h is.h ksp.h mat.h options.h pc.h petsc.h sys.h \
           sysio.h vec.h
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)
DIRS     = 

include $(ITOOLSDIR)/bmake/$(PARCH)

all:
	-cd sys; make BOPT=$(BOPT) PARCH=$(PARCH) workers
	-@$(MAKE) BOPT=$(BOPT) PARCH=$(PARCH) libfasttree 

ranlib:
	$(RANLIB) $(LDIR)/$(COMPLEX)/*.a

deletelibs:
	-$(RM) $(LDIR)/*.o $(LDIR)/*.a $(LDIR)/complex/*

deletemanpages:
	$(RM) -f man/man*/*

