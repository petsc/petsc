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
	$(RM) -f $(PETSCLIB)/docs/man/man*/*

deletewwwpages:
	$(RM) -f $(PETSCLIB)/docs/www/man*/*

deletelatexpages:
	$(RM) -f $(PETSCLIB)/docs/tex/rsum/*sum*.tex

#  to access the tags in emacs type esc-x visit-tags-table 
#  then esc . to find a function
etags:
	$(RM) -f TAGS
	etags -f TAGS    src/*/impls/*/*.h src/*/impls/*/*/*.h src/*/examples/*.c
	etags -a -f TAGS src/*/*.h */*.c src/*/src/*.c src/*/impls/*/*.c 
	etags -a -f TAGS src/*/impls/*/*/*.c src/*/utils/*.c
	etags -a -f TAGS docs/tex/design.tex src/sys/error/*.c
	etags -a -f TAGS include/*.h pinclude/*.h
	etags -a -f TAGS src/*/impls/*.c
	chmod g+w TAGS

keywords:
	$(RM) -f keywords
	grep Keywords src/*/src/*.c src/*/impls/*.c src/*/impls/*/*.c > key1
	cut -f1 -d: key1 > key2
	cut -f3 -d: key1 > key3
	paste key3 key2 > Keywords
	$(RM) -f key1 key2 key3
	chmod g+w Keywords

runexamples:
