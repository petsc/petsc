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
	-@$(OMAKE) BOPT=$(BOPT) PARCH=$(PARCH) COMPLEX=$(COMPLEX) \
           ACTION=libfast  tree 
	$(RANLIB) $(LDIR)/*.a

ranlib:
	$(RANLIB) $(LDIR)/*.a

deletelibs:
	-$(RM) $(LDIR)/*.a $(LDIR)/complex/*

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
	etags -f TAGS    src/*/impls/*/*.h src/*/impls/*/*/*.h src/*/examples/*.c
	etags -a -f TAGS src/*/*.h */*.c src/*/src/*.c src/*/impls/*/*.c 
	etags -a -f TAGS src/*/impls/*/*/*.c src/*/utils/*.c
	etags -a -f TAGS docs/design.tex src/sys/error/*.c
	etags -a -f TAGS include/*.h pinclude/*.h

keywords:
	$(RM) -f keywords
	grep Keywords src/*/src/*.c src/*/impls/*.c src/*/impls/*/*.c > key1
	cut -f1 -d: key1 > key2
	cut -f3 -d: key1 > key3
	paste key3 key2 > Keywords
	$(RM) -f key1 key2 key3

runexamples:
