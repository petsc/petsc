IPETSCDIR = .

CFLAGS   = $(OPT) -I$(IPETSCDIR)/include -I.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =
SOURCEF  =
WSOURCEC = 
SOURCEH  = Changes Machines Readme maint/addlinks maint/buildtest \
           maint/builddist FAQ Installation\
           maint/buildlinks maint/wwwman maint/xclude maint/crontab
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)
DIRS     = src include pinclude docs

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

all: chkpetsc_dir
	-$(RM) -f $(PDIR)/*.a
	-@$(OMAKE) BOPT=$(BOPT) PETSC_ARCH=$(PETSC_ARCH) \
           ACTION=libfast  tree 
	$(RANLIB) $(PDIR)/*.a

ranlib:
	$(RANLIB) $(PDIR)/*.a

deletelibs:
	-$(RM) -f $(PDIR)/*.a $(PDIR)/complex/* $(PDIR)/c++/*

deletemanpages:
	$(RM) -f $(PETSC_DIR)/Keywords $(PETSC_DIR)/docs/man/man*/*

deletewwwpages:
	$(RM) -f $(PETSC_DIR)/docs/www/man*/* $(PETSC_DIR)/docs/www/www.cit

deletelatexpages:
	$(RM) -f $(PETSC_DIR)/docs/tex/rsum/*sum*.tex

allmanpages: deletemanpages deletewwwpages deletelatexpages
	-make ACTION=manpages tree
	-make ACTION=wwwpages tree
	-make ACTION=latexpages tree
	-maint/wwwman

#  To access the tags in emacs, type M-x visit-tags-table and specify
#  the file petsc/TAGS.  Then, to move to where a PETSc function is
#  defined, enter M-. and the function name.  To search for a string
#  and move to the first occurrence, use M-x tags-search and the string.
#  To locate later occurrences, use M-,

etags:
	-@$(OMAKE)  ACTION=C2f77delete  tree
	$(RM) TAGS
	etags -f TAGS    src/*/impls/*/*.h src/*/impls/*/*/*.h 
	etags -a -f TAGS src/*/examples/*.c
	etags -a -f TAGS src/*/*.h src/*/src/*.c src/*/impls/*/*.c 
	etags -a -f TAGS src/*/impls/*/*/*.c src/*/impls/*/*/*/*.c 
	etags -a -f TAGS include/*.h pinclude/*.h bmake/common
	etags -a -f TAGS src/*/impls/*.c src/*/utils/*.c
	etags -a -f TAGS makefile src/*/src/makefile src/makefile 
	etags -a -f TAGS src/*/impls/makefile src/*/impls/*/makefile
	etags -a -f TAGS src/*/utils/makefile src/*/examples/makefile
	etags -a -f TAGS src/*/makefile src/*/impls/*/*/makefile
	etags -a -f TAGS bmake/common bmake/sun4/sun4* bmake/rs6000/rs6000* 
	etags -a -f TAGS bmake/IRIX/IRIX* bmake/freebsd/freebsd*
	etags -a -f TAGS bmake/hpux/hpux* bmake/alpha/alpha*
	etags -a -f TAGS bmake/t3d/t3d* bmake/paragon/paragon*
	etags -a -f TAGS docs/tex/routin.tex  docs/tex/manual.tex
	etags -a -f TAGS docs/tex/intro.tex  docs/tex/part1.tex
	chmod g+w TAGS
