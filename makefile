IPETSCDIR = .

CFLAGS   = $(OPT) -I$(IPETSCDIR)/include -I.. -I$(IPETSCDIR) $(CONF)
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

include $(IPETSCDIR)/bmake/$(PARCH)/$(PARCH)

all: chkpetsclib
	-@if [ ! -d $(LDIR) ]; then \
          echo $(LDIR) ; mkdir -p $(LDIR) ; fi
	-$(RM) -f $(LDIR)/*.a
	-@$(OMAKE) BOPT=$(BOPT) PARCH=$(PARCH) COMPLEX=$(COMPLEX) \
           ACTION=libfast  tree 
	$(RANLIB) $(LDIR)/*.a

ranlib:
	$(RANLIB) $(LDIR)/*.a

deletelibs:
	-$(RM) -f $(LDIR)/*.a $(LDIR)/complex/*

deletemanpages:
	$(RM) -f $(PETSCLIB)/Keywords $(PETSCLIB)/docs/man/man*/*

deletewwwpages:
	$(RM) -f $(PETSCLIB)/docs/www/man*/* $(PETSCLIB)/docs/www/www.cit

deletelatexpages:
	$(RM) -f $(PETSCLIB)/docs/tex/rsum/*sum*.tex

#  to access the tags in emacs type esc-x visit-tags-table 
#  then esc . to find a function
etags:
	$(RM) -f TAGS
	etags -f TAGS    src/*/impls/*/*.h src/*/impls/*/*/*.h src/*/examples/*.c
	etags -a -f TAGS src/*/*.h */*.c src/*/src/*.c src/*/impls/*/*.c 
	etags -a -f TAGS src/*/impls/*/*/*.c src/*/utils/*.c
	etags -a -f TAGS docs/tex/manual.tex src/sys/error/*.c
	etags -a -f TAGS include/*.h pinclude/*.h
	etags -a -f TAGS src/*/impls/*.c src/sys/*.c
	chmod g+w TAGS

runexamples:
