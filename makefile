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
DIRS     = src include pinclude

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)

all:
	-@$(OMAKE) BOPT=$(BOPT) PARCH=$(PARCH) ACTION=libfast  tree 
	$(RANLIB) $(LDIR)/*.a

ranlib:
	$(RANLIB) $(LDIR)/*.a

deletelibs:
	-$(RM) $(LDIR)/*.o $(LDIR)/*.a $(LDIR)/complex/*

deletemanpages:
	$(RM) -f $(PETSCLIB)/docs/man/man*/*

deletewwwpages:
	$(RM) -f $(PETSCLIB)/docs/www/man*/*

deletelatexpages:
	$(RM) -f $(PETSCLIB)/docs/rsum/*sum*.tex

#  to access the tags in emacs type esc-x visit-tags-table 
#  then esc . to find a function
etags:
	$(RM) -f TAGS
	etags -f TAGS src/*/impls/*/*.h src/*/impls/*/*/*.h src/*/examples/*.c
	etags -a -f TAGS src/*/*.h */*.c src/*/src/*.c src/*/impls/*/*.c 
	etags -a -f TAGS src/*/impls/*/*/*.c
	etags -a -f TAGS docs/design.tex
