#
# This is the makefile for installing PETSc. See the file
# Installation for directions on installing PETSc.
# See also bmake/common for additional commands.
#

CFLAGS	 =
SOURCEC	 =
SOURCEF	 =
DOCS	 = Changes Machines Readme maint/addlinks \
	   maint/builddist FAQ Installation BugReporting\
	   maint/buildlinks maint/wwwman maint/xclude maint/crontab\
	   bmake/common bmake/*/base* maint/autoftp docs/www/sec/* \
           include/finclude/generateincludes bin/petscviewinfo.text \
           bin/petscoptsinfo.text
OBJSC	 =
OBJSF	 =
LIBBASE	 = libpetscvec
DIRS	 = src include docs 

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

# Builds PETSc libraries for a given BOPT and architecture
all: chkpetsc_dir
	-$(RM) -f $(PDIR)/*
	-@echo "Beginning to compile libraries in all directories"
	-@echo On `date` on `hostname`
	-@echo "Using compiler: $(CC) $(COPTFLAGS)"
	-@echo "-----------------------------------------"
	-@echo "Using PETSc flags: $(PETSCFLAGS) $(PCONF)"
	-@echo "-----------------------------------------"
	-@echo "Using configuration flags: $(CONF)"
	-@echo "-----------------------------------------"
	-@echo "Using include paths: $(PETSC_INCLUDE)"
	-@echo "-----------------------------------------"
	-@echo "Using PETSc directory: $(PETSC_DIR)"
	-@echo "Using PETSc arch: $(PETSC_ARCH)"
	-@echo "========================================="
	-@$(OMAKE) BOPT=$(BOPT) PETSC_ARCH=$(PETSC_ARCH) \
	   ACTION=libfast  tree 
	-@cd $(PETSC_DIR)/src/sys/src ; \
	$(OMAKE) BOPT=$(BOPT) PETSC_ARCH=$(PETSC_ARCH) rs6000_time
	$(RANLIB) $(PDIR)/*.a
	-@chmod g+w  $(PDIR)/*.a
	-@echo "Completed building libraries"
	-@echo "========================================="

# Builds PETSc test examples for a given BOPT and architecture
testexamples: chkopts
	-@echo "Beginning to compile and run test examples"
	-@echo On `date` on `hostname`
	-@echo "Using compiler: $(CC) $(COPTFLAGS)"
	-@echo "-----------------------------------------"
	-@echo "Using PETSc flags: $(PETSCFLAGS) $(PCONF)"
	-@echo "-----------------------------------------"
	-@echo "Using include paths: $(PETSC_INCLUDE)"
	-@echo "-----------------------------------------"
	-@echo "Using PETSc directory: $(PETSC_DIR)"
	-@echo "Using PETSc arch: $(PETSC_ARCH)"
	-@echo "------------------------------------------"
	-@echo "Using linker: $(CLINKER)"
	-@echo "Using libraries: $(PETSC_LIB)"
	-@echo "------------------------------------------"
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines some of the numbers may not match exactly."
	-@echo "========================================="
	-@$(OMAKE) BOPT=$(BOPT) PETSC_ARCH=$(PETSC_ARCH) \
	   ACTION=testexamples_1  tree 
	-@echo "Completed compiling and running test examples"
	-@echo "========================================="

# Builds PETSc test examples for a given BOPT and architecture
testexamples_uni: chkopts
	-@echo "Beginning to compile and run uniprocessor test examples"
	-@echo On `date` on `hostname`
	-@echo "Using compiler: $(CC) $(COPTFLAGS)"
	-@echo "Using linker: $(CLINKER)"
	-@echo "------------------------------------------"
	-@echo "Using PETSc flags: $(PETSCFLAGS) $(PCONF)"
	-@echo "------------------------------------------"
	-@echo "Using include paths: $(PETSC_INCLUDE)"
	-@echo "------------------------------------------"
	-@echo "Using PETSc directory: $(PETSC_DIR)"
	-@echo "Using PETSc arch: $(PETSC_ARCH)"
	-@echo "------------------------------------------"
	-@echo "Using linker: $(CLINKER)"
	-@echo "Using libraries: $(PETSC_LIB)"
	-@echo "------------------------------------------"
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines some of the numbers may not match exactly."
	-@echo "========================================="
	-@$(OMAKE) BOPT=$(BOPT) PETSC_ARCH=$(PETSC_ARCH) \
	   ACTION=testexamples_4  tree 
	-@echo "Completed compiling and running uniprocessor test examples"
	-@echo "========================================="

#
# Builds PETSc Fortran interface libary
# Note:	 libfast cannot run on .F files on certain machines, so we
# use lib and check for errors here.
fortran: chkpetsc_dir
	-$(RM) -f $(PDIR)/libpetscfortran.*
	-@echo "Beginning to compile Fortran interface library"
	-@echo On `date` on `hostname`
	-@echo "Using Fortran compiler: $(FC) $(FFLAGS) $(FOPTFLAGS)"
	-@echo "Using C/C++ compiler: $(CC) $(COPTFLAGS)"
	-@echo "------------------------------------------"
	-@echo "Using PETSc flags: $(PETSCFLAGS) $(PCONF)"
	-@echo "------------------------------------------"
	-@echo "Using configuration flags: $(CONF)"
	-@echo "------------------------------------------"
	-@echo "Using include paths: $(PETSC_INCLUDE)"
	-@echo "------------------------------------------"
	-@echo "Using PETSc directory: $(PETSC_DIR)"
	-@echo "Using PETSc arch: $(PETSC_ARCH)"
	-@echo "========================================="
	-@cd src/fortran/custom; \
	  $(OMAKE) BOPT=$(BOPT) PETSC_ARCH=$(PETSC_ARCH) lib > trashz 2>&1; \
	  grep -v clog trashz | grep -v "information sections" | \
	  egrep -i '(Error|warning|Can)' >> /dev/null;\
	  if [ "$$?" != 1 ]; then \
	  cat trashz ; fi; $(RM) trashz
	-@cd src/fortran/auto; \
	  $(OMAKE) BOPT=$(BOPT) PETSC_ARCH=$(PETSC_ARCH) libfast
	$(RANLIB) $(PDIR)/libpetscfortran.a
	-@chmod g+w  $(PDIR)/*.a
	-@echo "Completed compiling Fortran interface library"
	-@echo "========================================="

# Builds PETSc test examples for a given BOPT and architecture
testfortran: chkopts
	-@echo "Beginning to compile and run Fortran test examples"
	-@echo On `date` on `hostname`
	-@echo "Using compiler: $(FC) $(FFLAGS) $(FOPTFLAGS)"
	-@echo "Using linker: $(FLINKER)"
	-@echo "Using PETSc flags: $(PETSCFLAGS) $(PCONF)"
	-@echo "------------------------------------------"
	-@echo "Using PETSc directory: $(PETSC_DIR)"
	-@echo "Using PETSc arch: $(PETSC_ARCH)"
	-@echo "------------------------------------------"
	-@echo "Using linker: $(FLINKER)"
	-@echo "Using libraries: $(PETSC_FORTRAN_LIB) $(PETSC_LIB)"
	-@echo "========================================="
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines or the way Fortran formats numbers"
	-@echo "some of the results may not match exactly."
	-@echo "========================================="
	-@echo "On some machines you may get messages of the form"
	-@echo "PetscScalarAddressToFortran:C and Fortran arrays are"
	-@echo "not commonly aligned or are too far apart to be indexed" 
	-@echo "by an integer. Locations: C xxxc Fortran xxxf"
	-@echo "Locations/sizeof(Scalar): C yyc Fortran yyf"
	-@echo "This indicates that you may not be able to use the"
	-@echo "PETSc routines VecGetArray() and MatGetArray() from Fortran" 
	-@echo "========================================="
	-@$(OMAKE) BOPT=$(BOPT) PETSC_ARCH=$(PETSC_ARCH) \
	   ACTION=testexamples_3  tree 
	-@echo "Completed compiling and running Fortran test examples"
	-@echo "========================================="
    
#
# Builds PETSc Fortran90 interface libary
# Note:	 libfast cannot run on .F files on certain machines, so we
# use lib and check for errors here.
# Note: F90 interface currently only supported in NAG F90 compiler
fortran90: chkpetsc_dir fortran
	-@echo "Beginning to compile Fortran90 interface library"
	-@echo On `date` on `hostname`
	-@echo "Using Fortran compiler: $(FC) $(FFLAGS) $(FOPTFLAGS)"
	-@echo "Using C/C++ compiler: $(CC) $(COPTFLAGS)"
	-@echo "------------------------------------------"
	-@echo "Using PETSc flags: $(PETSCFLAGS) $(PCONF)"
	-@echo "------------------------------------------"
	-@echo "Using configuration flags: $(CONF)"
	-@echo "------------------------------------------"
	-@echo "Using include paths: $(PETSC_INCLUDE)"
	-@echo "------------------------------------------"
	-@echo "Using PETSc directory: $(PETSC_DIR)"
	-@echo "Using PETSc arch: $(PETSC_ARCH)"
	-@echo "========================================="
	-@cd src/fortran/f90; \
	  $(OMAKE) BOPT=$(BOPT) PETSC_ARCH=$(PETSC_ARCH) lib > trashz 2>&1; \
	  grep -v clog trashz | grep -v "information sections" | \
	  egrep -i '(Error|warning|Can)' >> /dev/null;\
	  if [ "$$?" != 1 ]; then \
	  cat trashz ; fi; $(RM) trashz
	$(RANLIB) $(PDIR)/libpetscfortran.a
	-@chmod g+w  $(PDIR)/*.a
	-@echo "Completed compiling Fortran90 interface library"
	-@echo "========================================="

# Builds noise routines (not yet publically available)
# Note:	 libfast cannot run on .F files on certain machines, so we
# use lib and check for errors here.
noise: chkpetsc_dir
	-@echo "Beginning to compile noise routines"
	-@echo "Using Fortran compiler: $(FC) $(FFLAGS) $(FOPTFLAGS)"
	-@echo "Using C/C++ compiler: $(CC) $(COPTFLAGS)"
	-@echo "------------------------------------------"
	-@echo "Using PETSc flags: $(PETSCFLAGS) $(PCONF)"
	-@echo "------------------------------------------"
	-@echo "Using configuration flags: $(CONF)"
	-@echo "------------------------------------------"
	-@echo "Using include paths: $(PETSC_INCLUDE)"
	-@echo "------------------------------------------"
	-@echo "Using PETSc directory: $(PETSC_DIR)"
	-@echo "Using PETSc arch: $(PETSC_ARCH)"
	-@echo "========================================="
	-@cd src/snes/interface/noise; \
	  $(OMAKE) BOPT=$(BOPT) PETSC_ARCH=$(PETSC_ARCH) lib > trashz 2>&1; \
	  grep -v clog trashz | grep -v "information sections" | \
	  egrep -i '(Error|warning|Can)' >> /dev/null;\
	  if [ "$$?" != 1 ]; then \
	  cat trashz ; fi; $(RM) trashz
	$(RANLIB) $(PDIR)/libpetscsnes.a
	-@chmod g+w  $(PDIR)/libpetscsnes.a
	-@echo "Completed compiling noise routines"
	-@echo "========================================="


ranlib:
	$(RANLIB) $(PDIR)/*.a

# Deletes PETSc libraries
deletelibs:
	-$(RM) -f $(PDIR)/*.a $(PDIR)/complex/* $(PDIR)/c++/*

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

TAGS_INCLUDE_FILES  = include/*.h include/pinclude/*.h include/FINCLUDE/*.h 
TAGS_BMAKE_FILES    = bmake/common bmake/*/base*
TAGS_EXAMPLE_FILES  = src/*/examples/*/*.[c,h,F,f] src/*/examples/*/*/*.[c,h,F,f] \
                      src/benchmarks/*.c src/contrib/*/examples/*/*.[c,h,F,f]
TAGS_DOC_FILES      = docs/tex/manual/routin.tex docs/tex/manual/manual.tex \
                      docs/tex/manual/manual_tex.tex docs/tex/manual/intro.tex \
                      docs/tex/manual/part1.tex docs/tex/manual/part2.tex
TAGS_SRC_FILES      = src/*/*.[c,h] src/*/interface/*.[c,h] src/*/src/*.[c,h] \
                      src/*/utils/*.[c,h] \
                      src/*/impls/*.[c,h] src/*/impls/*/*.[c,h] src/*/impls/*/*/*.[c,h] \
                      src/gvec/impls/*/*/*/*/*.[c,h] src/contrib/*/*.[c,h] \
                      src/contrib/*/src/*.[c,h] src/fortran/custom/*.[c,h,F]
TAGS_MAKEFILE_FILES = include/makefile include/*/makefile \
                      makefile \
                      src/makefile src/*/makefile src/*/src/makefile \
                      src/*/interface/makefile \
                      src/*/utils/makefile \
                      src/*/impls/makefile src/*/impls/*/makefile src/*/impls/*/*/makefile \
                      src/*/examples/makefile src/*/examples/*/makefile src/*/examples/*/*/makefile \
                      src/gvec/impls/*/*/*/*/makefile src/gvec/impls/*/*/*/makefile \
                      src/fortran/*/makefile \
                      src/contrib/*/makefile src/contrib/*/src/makefile \
                      src/contrib/*/examples/makefile src/contrib/*/examples/*/makefile \
                      docs/makefile

# Builds all etags files
alletags:
	-make etags_complete
	-make etags
	-make etags_noexamples
	-make etags_makefiles

# Builds the basic etags file.	This should be employed by most users.
etags:
	$(RM) TAGS
	etags -f TAGS $(TAGS_INCLUDE_FILES) 
	etags -a -f TAGS $(TAGS_SRC_FILES) 
	etags -a -f TAGS $(TAGS_EXAMPLE_FILES) 
	etags -a -f TAGS $(TAGS_MAKEFILE_FILES) 
	etags -a -f TAGS $(TAGS_BMAKE_FILES) 
	chmod g+w TAGS

# Builds complete etags list; only for PETSc developers.
etags_complete:
	$(RM) TAGS_COMPLETE
	etags -f TAGS_COMPLETE $(TAGS_SRC_FILES) 
	etags -a -f TAGS_COMPLETE $(TAGS_INCLUDE_FILES) 
	etags -a -f TAGS_COMPLETE $(TAGS_EXAMPLE_FILES)
	etags -a -f TAGS_COMPLETE $(TAGS_MAKEFILE_FILES) 
	etags -a -f TAGS_COMPLETE $(TAGS_BMAKE_FILES) 
	etags -a -f TAGS_COMPLETE $(TAGS_DOC_FILES)
	chmod g+w TAGS_COMPLETE

# Builds the etags file that excludes the examples directories
etags_noexamples:
	$(RM) TAGS_NO_EXAMPLES
	etags -f TAGS_NO_EXAMPLES $(TAGS_SRC_FILES)
	etags -a -f TAGS_NO_EXAMPLES $(TAGS_INCLUDE_FILES) 
	etags -a -f TAGS_NO_EXAMPLES $(TAGS_MAKEFILE_FILES) 
	etags -a -f TAGS_NO_EXAMPLES $(TAGS_BMAKE_FILES) 
	etags -a -f TAGS_NO_EXAMPLES $(TAGS_DOC_FILES)
	chmod g+w TAGS_NO_EXAMPLES

# Builds the etags file for makefiles
etags_makefiles: 
	$(RM) TAGS_MAKEFILES
	etags -f TAGS_MAKEFILES $(TAGS_MAKEFILE_FILES) 
	etags -a -f TAGS_MAKEFILES $(TAGS_BMAKE_FILES) 
	chmod g+w TAGS_MAKEFILES

#
# ctags builds the tags file required for VI.
# To use the tags file do the following:
# 1. within vi invole the command - :set tags=/home/bsmith/petsc/tags
#    or add  the command to your ~/.exrc file - set tags=/home/bsmith/petsc/tags
# 2. now to go to a tag do - :tag TAGNAME for eg - :tag MatCreate
# 
ctags:  
	$(RM) tags
	ctags -w -f tags $(TAGS_INCLUDE_FILES) 
	ctags -w -a -f tags $(TAGS_SRC_FILES) 
	ctags -w -a -f tags $(TAGS_EXAMPLE_FILES)
	ctags -w -a -f tags $(TAGS_MAKEFILE_FILES) 
	ctags -w -a -f tags $(TAGS_BMAKE_FILES)
	chmod g+w tags

# ------------------------------------------------------------------
#
# All remaining actions are intended for PETSc developers only.
# PETSc users should not generally need to use these commands.
#

# Builds all versions of the man pages
allmanpages: allwwwpages alllatexpages
allwwwpages: deletewwwpages
	-make ACTION=wwwpages_buildcite tree
	-cd src/fortran/custom; make wwwpages_buildcite
	-cd src/fortran/custom; make wwwpages
	-make ACTION=wwwpages tree
	-maint/wwwman
	-maint/examplesindex.tcl -www
	-maint/htmlkeywords.tcl
	-@chmod g+w docs/www/man*/*

#This is similar to allwwwpages except -www -> -wwwhome
#The wwwmanpages built this way can pe placed at PETSc Home Page
allwwwhomepages: deletewwwpages
	-make ACTION=wwwpages_buildcite tree
	-cd src/fortran/custom; make wwwpages_buildcite
	-cd src/fortran/custom; make wwwpages
	-make ACTION=wwwpages tree
	-maint/wwwman
	-maint/examplesindex.tcl -wwwhome
	-maint/htmlkeywords.tcl -wwwhome
	-@chmod g+w docs/www/man*/*

alllatexpages: deletelatexpages
	-make ACTION=latexpages tree
	-cd src/fortran/custom; make latexpages
	-@chmod g+w docs/tex/rsum/*

# Builds Fortran stub files
allfortranstubs:
	-@include/finclude/generateincludes
	-@$(RM) -f src/fortran/auto/*.c
	-make ACTION=fortranstubs tree
	chmod g+w src/fortran/auto/*.c

allci: 
	-@cd src/fortran/custom ; $(OMAKE) BOPT=$(BOPT) ci
	-@$(OMAKE) BOPT=$(BOPT) PETSC_ARCH=$(PETSC_ARCH) ACTION=ci  tree 

allco: 
	-@cd src/fortran/custom ; $(OMAKE) BOPT=$(BOPT) co
	-@$(OMAKE) BOPT=$(BOPT) PETSC_ARCH=$(PETSC_ARCH) ACTION=co  tree 

#
#   The commands below are for generating ADIC versions of the code;
# they are not currently used.
#
CFLAGS   =  $(CPPFLAGS) $(CONF)
alladicignore:
	-@$(RM) $(PDIR)/adicignore
	-@$(OMAKE) BOPT=$(BOPT) PETSC_ARCH=$(PETSC_ARCH) ACTION=adicignore  tree 

alladic:
	-@echo "Beginning to compile ADIC source code in all directories"
	-@echo "Using ADIC compiler: $(ADIC_CC) $(CFLAGS)"
	-@echo "========================================="
	-@cd include ; \
           $(ADIC_CC) -s -f 1 $(CFLAGS) petsc.h 
	-@$(OMAKE) BOPT=$(BOPT) PETSC_ARCH=$(PETSC_ARCH) ACTION=adic  tree 
	-@cd src/inline ; \
            $(OMAKE) BOPT=$(BOPT) PETSC_ARCH=$(PETSC_ARCH) adic
	-@cd src/adic/blas ; \
            $(OMAKE) BOPT=$(BOPT) PETSC_ARCH=$(PETSC_ARCH) adic
	-@cd src/adic/lapack ; \
            $(OMAKE) BOPT=$(BOPT) PETSC_ARCH=$(PETSC_ARCH) adic

alladiclib:
	-@echo "Beginning to compile ADIC libraries in all directories"
	-@echo "Using compiler: $(CC) $(COPTFLAGS)"
	-@echo "-----------------------------------------"
	-@echo "Using PETSc flags: $(PETSCFLAGS) $(PCONF)"
	-@echo "-----------------------------------------"
	-@echo "Using configuration flags: $(CONF)"
	-@echo "-----------------------------------------"
	-@echo "Using include paths: $(PETSC_INCLUDE)"
	-@echo "-----------------------------------------"
	-@echo "Using PETSc directory: $(PETSC_DIR)"
	-@echo "Using PETSc arch: $(PETSC_ARCH)"
	-@echo "========================================="
	-@$(RM) -f  $(PDIR)/*adic.a
	-@$(OMAKE) BOPT=$(BOPT) PETSC_ARCH=$(PETSC_ARCH) ACTION=adiclib  tree
	-@cd src/adic/blas ; \
            $(OMAKE) BOPT=$(BOPT) PETSC_ARCH=$(PETSC_ARCH) adiclib
	-@cd src/adic/lapack ; \
            $(OMAKE) BOPT=$(BOPT) PETSC_ARCH=$(PETSC_ARCH) adiclib
	-@cd src/adic/src ; \
            $(OMAKE) BOPT=$(BOPT) PETSC_ARCH=$(PETSC_ARCH) lib






