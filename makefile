#
# This is the makefile for installing PETSc. See the file
# Installation for directions on installing PETSc.
# See also bmake/common for additional commands.
#

PETSC_DIR = .

CFLAGS	 =
SOURCEC	 =
SOURCEF	 =
SOURCEH	 = Changes Machines Readme maint/addlinks \
	   maint/builddist FAQ Installation Performance BugReporting\
	   maint/buildlinks maint/wwwman maint/xclude maint/crontab\
	   bmake/common bmake/*/base* maint/autoftp docs/www/sec/*
OBJSC	 =
OBJSF	 =
LIBBASE	 = libpetscvec
DIRS	 = src include docs 

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

# Builds PETSc libraries for a given BOPT and architecture
all: chkpetsc_dir
	-$(RM) -f $(PDIR)/*.a
	-@echo "Beginning to compile libraries in all directories"
	-@echo "Using compiler: $(CC) $(CFLAGS) $(COPTFLAGS)"
	-@echo "Using PETSc flags: $(PETSCFLAGS) $(PCONF)"
	-@echo "Using configuration flags: $(CONF)"
	-@echo "Using include paths: $(PETSC_INCLUDE)"
	-@echo "------------------------------------------"
	-@$(OMAKE) BOPT=$(BOPT) PETSC_ARCH=$(PETSC_ARCH) \
	   ACTION=libfast  tree 
	-@cd $(PETSC_DIR)/src/sys/src ; $(OMAKE) PETSC_ARCH=$(PETSC_ARCH) rs6000_time
	$(RANLIB) $(PDIR)/*.a
	-@echo "Completed building libraries"
	-@echo "------------------------------------------"

# Builds PETSc test examples for a given BOPT and architecture
testexamples: chkpetsc_dir
	-@echo "Beginning to compile and run test examples"
	-@echo "Using compiler: $(CC) $(CFLAGS) $(COPTFLAGS)"
	-@echo "Using PETSc flags: $(PETSCFLAGS) $(PCONF)"
	-@echo "Using include paths: $(PETSC_INCLUDE)"
	-@echo "Using linker: $(CLINKER)"
	-@echo "Using libraries: $(PETSC_LIB)"
	-@echo "------------------------------------------"
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines some of the numbers may not match exactly."
	-@echo "------------------------------------------"
	-@$(OMAKE) BOPT=$(BOPT) PETSC_ARCH=$(PETSC_ARCH) \
	   ACTION=testexamples_1  tree 
	-@echo "Completed compiling and running test examples"
	-@echo "------------------------------------------"

# Builds PETSc test examples for a given BOPT and architecture
testexamples_uni: chkpetsc_dir
	-@echo "Beginning to compile and run uniprocessor test examples"
	-@echo "Using compiler: $(CC) $(CFLAGS) $(COPTFLAGS)"
	-@echo "Using linker: $(CLINKER)"
	-@echo "Using PETSc flags: $(PETSCFLAGS) $(PCONF)"
	-@echo "Using include paths: $(PETSC_INCLUDE)"
	-@echo "Using libraries: $(PETSC_LIB)"
	-@echo "------------------------------------------"
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines some of the numbers may not match exactly."
	-@echo "------------------------------------------"
	-@$(OMAKE) BOPT=$(BOPT) PETSC_ARCH=$(PETSC_ARCH) \
	   ACTION=testexamples_4  tree 
	-@echo "Completed compiling and running uniprocessor test examples"
	-@echo "------------------------------------------"

# Builds PETSc test examples for a given BOPT and architecture
testfortran: chkpetsc_dir
	-@echo "Beginning to compile and run Fortran test examples"
	-@echo "Using compiler: $(FC) $(FFLAGS) $(FOPTFLAGS)"
	-@echo "Using linker: $(FLINKER)"
	-@echo "Using PETSc flags: $(PETSCFLAGS) $(PCONF)"
	-@echo "Using libraries: $(PETSC_FORTRAN_LIB)  $(PETSC_LIB)"
	-@echo "------------------------------------------"
	-@$(OMAKE) BOPT=$(BOPT) PETSC_ARCH=$(PETSC_ARCH) \
	   ACTION=testexamples_3  tree 
	-@echo "Completed compiling and running Fortran test examples"
	-@echo "------------------------------------------"

#
# Builds PETSc Fortran interface libary
# Note:	 libfast cannot run on .F files on certain machines, so we
# use lib and check for errors here.
fortran: chkpetsc_dir
	-$(RM) -f $(PDIR)/libpetscfortran.a
	-@echo "Beginning to compile Fortran interface library"
	-@echo "Using Fortran compiler: $(FC) $(FFLAGS) $(FOPTFLAGS)"
	-@echo "Using C/C++ compiler: $(CC) $(CFLAGS) $(COPTFLAGS)"
	-@echo "Using PETSc flags: $(PETSCFLAGS) $(PCONF)"
	-@echo "Using configuration flags: $(CONF)"
	-@echo "Using include paths: $(PETSC_INCLUDE)"
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
	$(RM) -f $(PETSC_DIR)/docs/www/man*/* $(PETSC_DIR)/docs/www/www.cit \
	         $(PETSC_DIR)/docs/www/man*.html

# Deletes man pages (LaTeX version)
deletelatexpages:
	$(RM) -f $(PETSC_DIR)/docs/tex/rsum/*sum*.tex

# To access the tags in emacs, type M-x visit-tags-table and specify
# the file petsc/TAGS.	Then, to move to where a PETSc function is
# defined, enter M-. and the function name.  To search for a string
# and move to the first occurrence, use M-x tags-search and the string.
# To locate later occurrences, use M-,

# Builds all etags files
alletags:
	-make etags_complete
	-make etags
	-make etags_noexamples
	-make etags_makefiles

# Builds the basic etags file.	This should be employed by most users.
etags:
	$(RM) TAGS
	etags -f TAGS	 src/*/impls/*/*.h src/*/impls/*/*/*.h 
	etags -a -f TAGS src/*/examples/*.c src/*/examples/*/*.c
	etags -a -f TAGS src/*/*.h src/*/*/*.h src/*/interface/*.c 
	etags -a -f TAGS src/*/src/*.c src/*/impls/*/*.c 
	etags -a -f TAGS src/*/impls/*/*/*.c  src/benchmarks/*.c
	etags -a -f TAGS src/contrib/*/*.c src/contrib/*/src/*.c 
	etags -a -f TAGS src/contrib/*/examples/*.c
	etags -a -f TAGS src/contrib/*/src/*.h src/contrib/*/examples/*.F
	etags -a -f TAGS include/*.h include/pinclude/*.h bmake/common
	etags -a -f TAGS include/FINCLUDE/*.h 
	etags -a -f TAGS src/*/impls/*.c src/*/utils/*.c
	etags -a -f TAGS makefile src/*/src/makefile
	etags -a -f TAGS src/*/interface/makefile src/makefile 
	etags -a -f TAGS src/*/impls/makefile src/*/impls/*/makefile
	etags -a -f TAGS src/*/utils/makefile src/*/examples/makefile
	etags -a -f TAGS src/*/examples/*/makefile src/*/examples/*/*/makefile
	etags -a -f TAGS src/*/makefile src/*/impls/*/*/makefile
	etags -a -f TAGS src/contrib/*/makefile src/contrib/*/src/makefile 
	etags -a -f TAGS src/fortran/makefile src/fortran/auto/makefile 
	etags -a -f TAGS src/fortran/custom/makefile
	etags -a -f TAGS include/makefile include/*/makefile 
	etags -a -f TAGS bmake/common bmake/*/base*
	etags -a -f TAGS src/fortran/custom/*.c src/fortran/auto/*.c 
	etags -a -f TAGS src/benchmarks/*.c src/fortran/custom/*.F
	etags -a -f TAGS src/*/examples/*.F src/*/examples/*.f 
	etags -a -f TAGS src/*/examples/*/*.F src/*/examples/*/*.f
	chmod g+w TAGS

# Builds complete etags list; only for PETSc developers.
etags_complete:
	$(RM) TAGS_COMPLETE
	etags -f TAGS_COMPLETE	 src/*/impls/*/*.h src/*/impls/*/*/*.h 
	etags -a -f TAGS_COMPLETE src/*/examples/*.c src/*/examples/*/*.c
	etags -a -f TAGS_COMPLETE src/*/*.h src/*/*/*.h src/*/interface/*.c 
	etags -a -f TAGS_COMPLETE src/*/src/*.c src/*/impls/*/*.c 
	etags -a -f TAGS_COMPLETE src/*/impls/*/*/*.c  src/benchmarks/*.c
	etags -a -f TAGS_COMPLETE src/contrib/*/*.c src/contrib/*/src/*.c 
	etags -a -f TAGS_COMPLETE src/contrib/*/examples/*.c
	etags -a -f TAGS_COMPLETE src/contrib/*/src/*.h 
	etags -a -f TAGS_COMPLETE src/contrib/*/examples/*.F
	etags -a -f TAGS_COMPLETE include/*.h include/pinclude/*.h bmake/common
	etags -a -f TAGS_COMPLETE include/FINCLUDE/*.h 
	etags -a -f TAGS_COMPLETE src/*/impls/*.c src/*/utils/*.c
	etags -a -f TAGS_COMPLETE makefile src/*/src/makefile
	etags -a -f TAGS_COMPLETE src/*/interface/makefile src/makefile 
	etags -a -f TAGS_COMPLETE src/*/impls/makefile src/*/impls/*/makefile
	etags -a -f TAGS_COMPLETE src/*/utils/makefile src/*/examples/makefile
	etags -a -f TAGS_COMPLETE src/*/examples/*/makefile 
	etags -a -f TAGS_COMPLETE src/*/examples/*/*/makefile
	etags -a -f TAGS_COMPLETE src/*/makefile src/*/impls/*/*/makefile
	etags -a -f TAGS_COMPLETE src/contrib/*/makefile 
	etags -a -f TAGS_COMPLETE src/contrib/*/src/makefile 
	etags -a -f TAGS_COMPLETE src/fortran/makefile src/fortran/auto/makefile 
	etags -a -f TAGS_COMPLETE src/fortran/custom/makefile
	etags -a -f TAGS_COMPLETE include/makefile include/*/makefile 
	etags -a -f TAGS_COMPLETE bmake/common bmake/*/base*
	etags -a -f TAGS_COMPLETE src/fortran/custom/*.c src/fortran/auto/*.c 
	etags -a -f TAGS_COMPLETE src/benchmarks/*.c
	etags -a -f TAGS_COMPLETE src/*/examples/*.F src/*/examples/*.f 
	etags -a -f TAGS_COMPLETE src/fortran/custom/*.F 
	etags -a -f TAGS_COMPLETE src/*/examples/*/*.F src/*/examples/*/*.f
	etags -a -f TAGS_COMPLETE docs/tex/manual/routin.tex 
	etags -a -f TAGS_COMPLETE docs/tex/manual/manual.tex
	etags -a -f TAGS_COMPLETE docs/tex/manual/manual_tex.tex
	etags -a -f TAGS_COMPLETE docs/tex/manual/intro.tex 
	etags -a -f TAGS_COMPLETE docs/tex/manual/part1.tex
	etags -a -f TAGS_COMPLETE docs/tex/manual/part2.tex
	etags -a -f TAGS_COMPLETE docs/tex/manual/intro.tex docs/makefile
	chmod g+w TAGS_COMPLETE

# Builds the etags file that excludes the examples directories
etags_noexamples:
	$(RM) TAGS_NO_EXAMPLES
	etags -f TAGS_NO_EXAMPLES src/*/impls/*/*.h src/*/impls/*/*/*.h 
	etags -a -f TAGS_NO_EXAMPLES src/*/*.h src/*/*/*.h src/*/interface/*.c 
	etags -a -f TAGS_NO_EXAMPLES src/*/src/*.c  src/*/impls/*/*.c 
	etags -a -f TAGS_NO_EXAMPLES src/*/impls/*/*/*.c 
	etags -a -f TAGS_NO_EXAMPLES src/contrib/*/*.c src/contrib/*/src/*.c 
	etags -a -f TAGS_NO_EXAMPLES src/contrib/*/src/*.h
	etags -a -f TAGS_NO_EXAMPLES include/*.h include/pinclude/*.h
	etags -a -f TAGS_NO_EXAMPLES include/FINCLUDE/*.h
	etags -a -f TAGS_NO_EXAMPLES bmake/common
	etags -a -f TAGS_NO_EXAMPLES src/*/impls/*.c src/*/utils/*.c
	etags -a -f TAGS_NO_EXAMPLES makefile src/*/src/makefile 
	etags -a -f TAGS_NO_EXAMPLES src/*/interface/makefile src/makefile 
	etags -a -f TAGS_NO_EXAMPLES src/*/impls/makefile src/*/impls/*/makefile
	etags -a -f TAGS_NO_EXAMPLES src/*/utils/makefile
	etags -a -f TAGS_NO_EXAMPLES src/*/makefile src/*/impls/*/*/makefile
	etags -a -f TAGS_NO_EXAMPLES src/contrib/*/makefile 
	etags -a -f TAGS_NO_EXAMPLES src/contrib/*/src/makefile 
	etags -a -f TAGS_NO_EXAMPLES src/fortran/makefile 
	etags -a -f TAGS_NO_EXAMPLES src/fortran/auto/makefile 
	etags -a -f TAGS_NO_EXAMPLES src/fortran/custom/makefile
	etags -a -f TAGS_NO_EXAMPLES include/makefile include/*/makefile 
	etags -a -f TAGS_NO_EXAMPLES bmake/common bmake/*/base*
	etags -a -f TAGS_NO_EXAMPLES src/fortran/auto/*.c
	etags -a -f TAGS_NO_EXAMPLES src/fortran/custom/*.c 
	etags -a -f TAGS_NO_EXAMPLES src/fortran/custom/*.F
	etags -a -f TAGS_NO_EXAMPLES docs/tex/manual/routin.tex 
	etags -a -f TAGS_NO_EXAMPLES docs/tex/manual/manual.tex
	etags -a -f TAGS_NO_EXAMPLES docs/tex/manual/intro.tex
	etags -a -f TAGS_NO_EXAMPLES docs/tex/manual/part1.tex 
	etags -a -f TAGS_NO_EXAMPLES docs/tex/manual/part2.tex 
	etags -a -f TAGS_NO_EXAMPLES docs/makefile
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
	etags -a -f TAGS_MAKEFILES src/*/examples/makefile 
	etags -a -f TAGS_MAKEFILES src/*/examples/*/makefile
	etags -a -f TAGS_MAKEFILES src/*/examples/*/*/makefile
	etags -a -f TAGS_MAKEFILES src/fortran/makefile 
	etags -a -f TAGS_MAKEFILES src/fortran/auto/makefile 
	etags -a -f TAGS_MAKEFILES src/contrib/*/makefile 
	etags -a -f TAGS_MAKEFILES src/contrib/*/src/makefile 
	etags -a -f TAGS_MAKEFILES src/fortran/custom/makefile
	etags -a -f TAGS_MAKEFILES include/makefile include/*/makefile
	etags -a -f TAGS_MAKEFILES bmake/common bmake/*/base*
	etags -a -f TAGS_MAKEFILES docs/makefile
	chmod g+w TAGS_MAKEFILES

# ------------------------------------------------------------------
#
# All remaining actions are intended for PETSc developers only.
# PETSc users should not generally need to use these commands.
#

# Builds all versions of the man pages
allmanpages: deletemanpages allwwwpages alllatexpages
	-make ACTION=manpages tree
	-cd src/fortran/custom; make manpages
	-cd docs/man; catman -W .
allwwwpages: deletewwwpages
	-make ACTION=wwwpages_buildcite tree
	-cd src/fortran/custom; make wwwpages_buildcite
	-cd src/fortran/custom; make wwwpages
	-make ACTION=wwwpages tree
	-maint/wwwman
	-maint/examplesindex.tcl
alllatexpages: deletelatexpages
	-make ACTION=latexpages tree
	-cd src/fortran/custom; make latexpages

# Builds Fortran stub files
allfortranstubs:
	-@include/finclude/generateincludes
	-@$(RM) -f $(PETSC_DIR)/src/fortran/auto/*.c
	-make ACTION=fortranstubs tree
	chmod g+w $(PETSC_DIR)/src/fortran/auto/*.c



