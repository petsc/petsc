#
# This makefile contains some basic commands for building PETSc.
# See bmake/common for additional commands.
#

PETSC_DIR = .

CFLAGS   =  -I$(PETSC_DIR)/include -I.. -I$(PETSC_DIR) $(CONF) $(PCONF)
SOURCEC  =
SOURCEF  =
SOURCEH  = Changes Machines Readme maint/addlinks maint/buildtest \
           maint/builddist FAQ Installation Performance BugReporting\
           maint/buildlinks maint/wwwman maint/xclude maint/crontab\
           bmake/common bmake/*/* maint/autoftp
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = src include docs 

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

# Builds PETSc libraries for a given BOPT and architecture
all: chkpetsc_dir
	-$(RM) -f $(PDIR)/*.a
	-@echo "Beginning to compile libraries in all directories"
	-@echo "Using $(CC) $(PETSC_INCLUDE) $(CONF) $(PCONF) $(BASEOPT)"
	-@echo "------------------------------------------"
	-@$(OMAKE) BOPT=$(BOPT) PETSC_ARCH=$(PETSC_ARCH) \
           ACTION=libfast  tree 
	$(RANLIB) $(PDIR)/*.a
	-@echo "Completed building libraries"
	-@echo "------------------------------------------"

# Builds PETSc test examples for a given BOPT and architecture
testexamples: chkpetsc_dir
	-@echo "Beginning to compile and run test examples"
	-@echo "Using compiler: $(CC) $(PETSC_INCLUDE) $(PCONF) $(BASEOPT)"
	-@echo "Using linker: $(CLINKER)"
	-@echo "Using libraries: $(PETSC_LIB)"
	-@echo "------------------------------------------"
	-@echo "Due to different numerical round-off on   "
	-@echo "certain machines some of the numbers may not  "
	-@echo "match exactly."
	-@echo "------------------------------------------"
	-@$(OMAKE) BOPT=$(BOPT) PETSC_ARCH=$(PETSC_ARCH) \
           ACTION=testexamples_1  tree 
	-@echo "Completed compiling and running test examples"
	-@echo "------------------------------------------"

# Builds PETSc test examples for a given BOPT and architecture
testfortran: chkpetsc_dir
	-@echo "Beginning to compile and run Fortran test examples"
	-@echo "Using compiler: $(FC) $(BASEOPTF)"
	-@echo "Using linker: $(FLINKER)"
	-@echo "Using libraries: $(PETSC_FORTRAN_LIB)  $(PETSC_LIB)"
	-@echo "------------------------------------------"
	-@$(OMAKE) BOPT=$(BOPT) PETSC_ARCH=$(PETSC_ARCH) \
           ACTION=testexamples_3  tree 
	-@echo "Completed compiling and running Fortran test examples"
	-@echo "------------------------------------------"

#
# Builds PETSc Fortran interface libary
# Note:  libfast cannot run on .F files on certain machines, so we
# use lib and check for errors here.
fortran: chkpetsc_dir
	-$(RM) -f $(PDIR)/libpetscfortran.a
	-@echo "Beginning to compile Fortran interface library"
	-@echo "------------------------------------------"
	-@cd src/fortran/custom; \
          $(OMAKE) BOPT=$(BOPT) PETSC_ARCH=$(PETSC_ARCH) lib > trashz 2>&1; \
          grep -v clog trashz | grep -v "information sections" | \
          egrep -i '(Error|warning|Can)' >> /dev/null;\
          if [ "$$?" != 1 ]; then \
          cat trashz ; fi; $(RM) trashz
	-@cd src/fortran/auto; \
          $(OMAKE) BOPT=$(BOPT) PETSC_ARCH=$(PETSC_ARCH) libfast
	$(RANLIB) $(PDIR)/libpetscfortran.a
	-@echo "Completed compiling Fortran interface library"
	-@echo "------------------------------------------"
    
ranlib:
	$(RANLIB) $(PDIR)/*.a

# Deletes PETSc libraries
deletelibs:
	-$(RM) -f $(PDIR)/*.a $(PDIR)/complex/* $(PDIR)/c++/*

# Deletes man pages (xman version)
deletemanpages:
	$(RM) -f $(PETSC_DIR)/Keywords $(PETSC_DIR)/docs/man/man*/*

# Deletes man pages (HTML version)
deletewwwpages:
	$(RM) -f $(PETSC_DIR)/docs/www/man*/* $(PETSC_DIR)/docs/www/www.cit

# Deletes man pages (LaTeX version)
deletelatexpages:
	$(RM) -f $(PETSC_DIR)/docs/tex/rsum/*sum*.tex

# To access the tags in emacs, type M-x visit-tags-table and specify
# the file petsc/TAGS.  Then, to move to where a PETSc function is
# defined, enter M-. and the function name.  To search for a string
# and move to the first occurrence, use M-x tags-search and the string.
# To locate later occurrences, use M-,

# Builds all etags files
alletags:
	-make etags
	-make etags_noexamples
	-make etags_makefiles

# Builds the basic etags file.  This should be employed by most users.
etags:
	$(RM) TAGS
	etags -f TAGS    src/*/impls/*/*.h src/*/impls/*/*/*.h 
	etags -a -f TAGS src/*/examples/*.c src/*/examples/*/*.c
	etags -a -f TAGS src/*/*.h src/*/*/*.h src/*/interface/*.c 
	etags -a -f TAGS src/*/src/*.c src/*/impls/*/*.c 
	etags -a -f TAGS src/*/impls/*/*/*.c 
	etags -a -f TAGS include/*.h include/*include/*.h bmake/common
	etags -a -f TAGS src/*/impls/*.c src/*/utils/*.c
	etags -a -f TAGS makefile src/*/src/makefile
	etags -a -f TAGS src/*/interface/makefile src/makefile 
	etags -a -f TAGS src/*/impls/makefile src/*/impls/*/makefile
	etags -a -f TAGS src/*/utils/makefile src/*/examples/makefile
	etags -a -f TAGS src/*/examples/*/makefile
	etags -a -f TAGS src/*/makefile src/*/impls/*/*/makefile
	etags -a -f TAGS src/fortran/makefile src/fortran/auto/makefile 
	etags -a -f TAGS src/fortran/custom/makefile
	etags -a -f TAGS include/makefile include/*/makefile docs/makefile
	etags -a -f TAGS bmake/common bmake/sun4/sun4* bmake/rs6000/rs6000* 
	etags -a -f TAGS bmake/solaris/solaris*
	etags -a -f TAGS bmake/IRIX/IRIX* bmake/freebsd/freebsd*
	etags -a -f TAGS bmake/hpux/hpux* bmake/alpha/alpha*
	etags -a -f TAGS bmake/t3d/t3d* bmake/paragon/paragon*
	etags -a -f TAGS docs/tex/routin.tex  docs/tex/manual.tex docs/tex/manual_tex.tex
	etags -a -f TAGS docs/tex/intro.tex  docs/tex/part1.tex docs/tex/part2.tex
	etags -a -f TAGS src/fortran/custom/*.c src/fortran/auto/*.c
	etags -a -f TAGS src/*/examples/*.F
	chmod g+w TAGS

# Builds the etags file that excludes the examples directories
etags_noexamples:
	$(RM) TAGS_NO_EXAMPLES
	etags -f TAGS_NO_EXAMPLES src/*/impls/*/*.h src/*/impls/*/*/*.h 
	etags -a -f TAGS_NO_EXAMPLES src/*/*.h src/*/*/*.h src/*/interface/*.c 
	etags -a -f TAGS_NO_EXAMPLES src/*/src/*.c  src/*/impls/*/*.c 
	etags -a -f TAGS_NO_EXAMPLES src/*/impls/*/*/*.c 
	etags -a -f TAGS_NO_EXAMPLES include/*.h include/*include/*.h
	etags -a -f TAGS_NO_EXAMPLES bmake/common
	etags -a -f TAGS_NO_EXAMPLES src/*/impls/*.c src/*/utils/*.c
	etags -a -f TAGS_NO_EXAMPLES makefile src/*/src/makefile 
	etags -a -f TAGS_NO_EXAMPLES src/*/interface/makefile src/makefile 
	etags -a -f TAGS_NO_EXAMPLES src/*/impls/makefile src/*/impls/*/makefile
	etags -a -f TAGS_NO_EXAMPLES src/*/utils/makefile
	etags -a -f TAGS_NO_EXAMPLES src/*/makefile src/*/impls/*/*/makefile
	etags -a -f TAGS_NO_EXAMPLES src/fortran/makefile src/fortran/auto/makefile 
	etags -a -f TAGS_NO_EXAMPLES src/fortran/custom/makefile
	etags -a -f TAGS_NO_EXAMPLES include/makefile include/*/makefile docs/makefile
	etags -a -f TAGS_NO_EXAMPLES bmake/common bmake/sun4/sun4* 
	etags -a -f TAGS_NO_EXAMPLES bmake/rs6000/rs6000* 
	etags -a -f TAGS_NO_EXAMPLES bmake/solaris/solaris*
	etags -a -f TAGS_NO_EXAMPLES bmake/IRIX/IRIX* bmake/freebsd/freebsd*
	etags -a -f TAGS_NO_EXAMPLES bmake/hpux/hpux* bmake/alpha/alpha*
	etags -a -f TAGS_NO_EXAMPLES bmake/t3d/t3d* bmake/paragon/paragon*
	etags -a -f TAGS_NO_EXAMPLES docs/tex/routin.tex  docs/tex/manual.tex
	etags -a -f TAGS_NO_EXAMPLES docs/tex/intro.tex  docs/tex/part1.tex
	etags -a -f TAGS_NO_EXAMPLES src/fortran/auto/*.c
	etags -a -f TAGS_NO_EXAMPLES src/fortran/custom/*.c 
	chmod g+w TAGS_NO_EXAMPLES

# Builds the etags file for makefiles
etags_makefiles:
	$(RM) TAGS_MAKEFILES
	etags -a -f TAGS_MAKEFILES bmake/common
	etags -a -f TAGS_MAKEFILES makefile src/*/src/makefile 
	etags -a -f TAGS_MAKEFILES src/*/interface/makefile src/makefile 
	etags -a -f TAGS_MAKEFILES src/*/impls/makefile src/*/impls/*/makefile
	etags -a -f TAGS_MAKEFILES src/*/utils/makefile src/*/interface/makefile
	etags -a -f TAGS_MAKEFILES src/*/makefile src/*/impls/*/*/makefile
	etags -a -f TAGS_MAKEFILES src/*/examples/makefile src/*/examples/*/makefile
	etags -a -f TAGS_MAKEFILES src/fortran/makefile src/fortran/auto/makefile 
	etags -a -f TAGS_MAKEFILES src/fortran/custom/makefile
	etags -a -f TAGS_MAKEFILES include/makefile include/*/makefile docs/makefile
	etags -a -f TAGS_MAKEFILES bmake/common bmake/sun4/sun4* 
	etags -a -f TAGS_MAKEFILES bmake/rs6000/rs6000* 
	etags -a -f TAGS_MAKEFILES bmake/solaris/solaris*
	etags -a -f TAGS_MAKEFILES bmake/IRIX/IRIX* bmake/freebsd/freebsd*
	etags -a -f TAGS_MAKEFILES bmake/hpux/hpux* bmake/alpha/alpha*
	etags -a -f TAGS_MAKEFILES bmake/t3d/t3d* bmake/paragon/paragon*
	chmod g+w TAGS_MAKEFILES

# ------------------------------------------------------------------
#
# All remaining actions are intended for PETSc developers only.
# PETSc users should not generally need to use these commands.
#

# Builds all versions of the man pages
allmanpages: deletemanpages deletewwwpages deletelatexpages
	-make ACTION=manpages tree
	-make ACTION=wwwpages tree
	-make ACTION=latexpages tree
	-maint/wwwman

alllatexpages: deletelatexpages
	-make ACTION=latexpages tree

# Builds Fortran stub files
allfortranstubs:
	-@$(RM) $(PETSC_DIR)/fortran/auto/*.c
	-make ACTION=fortranstubs tree

