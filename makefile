# $Id: makefile,v 1.250 1998/11/06 22:03:43 bsmith Exp bsmith $ 
#
# This is the makefile for installing PETSc. See the file
# Installation for directions on installing PETSc.
# See also bmake/common for additional commands.
#
ALL: all

DIRS	 = src include docs 

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

#
# Basic targets to build PETSc libraries.
# all     : builds the c, fortran,f90 libraries
# fortran : builds the fortran libary
# f90     : builds the fortran and the f90 libraries.
#
all       : info chkpetsc_dir deletelibs build_c build_fortrankernels \
	    build_fortran build_fortran90 shared
fortran   : info chkpetsc_dir build_fortran
fortran90 : fortran build_fortran90

#
#  Prints information about the system and PETSc being compiled
#
info:
	-@echo "=========================================="
	-@echo On `date` on `hostname`
	-@echo Machine characteristics: `uname -a`
	-@echo "-----------------------------------------"
	-@echo "Using C compiler: ${CC} ${COPTFLAGS} ${CCPPFLAGS}"
	-@if [ -n "${CCV}" -a "${CCV}" != "unknown" ] ; then \
	  echo "Compiler version:" `${CCV}` ; fi
	-@echo "Using Fortran compiler: ${FC} ${FOPTFLAGS} ${FCPPFLAGS}"
	-@echo "-----------------------------------------"
	-@grep PETSC_VERSION_NUMBER include/petsc.h | sed "s/........//"
	-@echo "-----------------------------------------"
	-@echo "Using PETSc flags: ${PETSCFLAGS} ${PCONF}"
	-@echo "-----------------------------------------"
	-@echo "Using configuration flags:"
	-@grep "define " bmake/${PETSC_ARCH}/petscconf.h
	-@echo "-----------------------------------------"
	-@echo "Using include paths: ${PETSC_INCLUDE}"
	-@echo "-----------------------------------------"
	-@echo "Using PETSc directory: ${PETSC_DIR}"
	-@echo "Using PETSc arch: ${PETSC_ARCH}"
	-@echo "------------------------------------------"
	-@echo "Using C linker: ${CLINKER}"
	-@echo "Using libraries: ${PETSC_LIB}"
	-@echo "Using Fortran linker: ${FLINKER}"
	-@echo "Using Fortran libraries: ${PETSC_FORTRAN_LIB}"
	-@echo "=========================================="

#
# Build the PETSc libraries
#
build_c:
	-@echo "BEGINNING TO COMPILE LIBRARIES IN ALL DIRECTORIES"
	-@echo "========================================="
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} ACTION=libfast  tree 
	-@cd ${PETSC_DIR}/src/sys/src/time ; \
	${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} rs6000_time
	${RANLIB} ${PDIR}/*.a
	-@chmod g+w  ${PDIR}/*.a
	-@echo "Completed building libraries"
	-@echo "========================================="

#
# Builds PETSc Fortran interface libary
# Note:	 libfast cannot run on .F files on certain machines, so we
# use lib and check for errors here.

build_fortran:
	-@echo "BEGINNING TO COMPILE FORTRAN INTERFACE LIBRARY"
	-@echo "========================================="
	-${RM} -f ${PDIR}/libpetscfortran.*
	-@cd src/fortran/auto; \
	  ${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} libfast
	-@cd src/fortran/custom; \
	  ${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} lib > trashz 2>&1; \
	  grep -v clog trashz | grep -v "information sections" | \
	  egrep -i '(Error|warning|Can)' >> /dev/null;\
	  if [ "$$?" != 1 ]; then \
	  cat trashz ; fi; ${RM} trashz
	${RANLIB} ${PDIR}/libpetscfortran.a
	-@chmod g+w  ${PDIR}/*.a
	-@echo "Completed compiling Fortran interface library"
	-@echo "========================================="

#
# Builds PETSc Fortran90 interface libary
# Note: F90 interface currently supported in NAG, IRIX, IBM F90 compilers.
#
build_fortran90: 
	-@echo "BEGINNING TO COMPILE FORTRAN90 INTERFACE LIBRARY"
	-@echo "========================================="
	-@cd src/fortran/f90; \
	  ${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} lib > trashz 2>&1; \
	  grep -v clog trashz | grep -v "information sections" | \
	  egrep -i '(Error|warning|Can)' >> /dev/null;\
	  if [ "$$?" != 1 ]; then \
	  cat trashz ; fi; ${RM} trashz
	${RANLIB} ${PDIR}/libpetscfortran.a
	-@chmod g+w  ${PDIR}/*.a
	-@echo "Completed compiling Fortran90 interface library"
	-@echo "========================================="

#
# Builds PETSc Fortran kernels; some numerical kernels have
# a Fortran version that may give better performance on certain 
# machines. These always provide better performance for complex numbers.
#
build_fortrankernels: chkpetsc_dir 
	-@echo "BEGINNING TO COMPILE FORTRAN KERNELS LIBRARY"
	-@echo "========================================="
	-@cd src/fortran/kernels; \
	  ${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} lib
	-@chmod g+w  ${PDIR}/*.a
	-@echo "Completed compiling Fortran kernels library"
	-@echo "========================================="

petscblas: info chkpetsc_dir
	-${RM} -f ${PDIR}/libpetscblas.*
	-@echo "BEGINNING TO COMPILE C VERSION OF BLAS AND LAPACK"
	-@echo "========================================="
	-@cd src/blaslapack; \
	  ${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} ACTION=libfast tree
	${RANLIB} ${PDIR}/libpetscblas.a
	-@chmod g+w  ${PDIR}/*.a
	-@echo "Completed compiling C version of BLAS and LAPACK"
	-@echo "========================================="


# Builds PETSc test examples for a given BOPT and architecture
testexamples: info chkopts
	-@echo "BEGINNING TO COMPILE AND RUN TEST EXAMPLES"
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines some of the numbers may not match exactly."
	-@echo "========================================="
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} \
	   ACTION=testexamples_1  tree 
	-@echo "Completed compiling and running test examples"
	-@echo "========================================="

# Builds PETSc test examples for a given BOPT and architecture
testfortran: info chkopts
	-@echo "BEGINNING TO COMPILE AND RUN FORTRAN TEST EXAMPLES"
	-@echo "========================================="
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines or the way Fortran formats numbers"
	-@echo "some of the results may not match exactly."
	-@echo "========================================="
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} \
	   ACTION=testexamples_3  tree 
	-@echo "Completed compiling and running Fortran test examples"
	-@echo "========================================="

# Builds PETSc test examples for a given BOPT and architecture
testexamples_uni: info chkopts
	-@echo "BEGINNING TO COMPILE AND RUN TEST UNI-PROCESSOR EXAMPLES"
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines some of the numbers may not match exactly."
	-@echo "========================================="
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} \
	   ACTION=testexamples_4  tree 
	-@echo "Completed compiling and running uniprocessor test examples"
	-@echo "========================================="
testfortran_uni: info chkopts
	-@echo "BEGINNING TO COMPILE AND RUN TEST UNI-PROCESSOR FORTRAN EXAMPLES"
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines some of the numbers may not match exactly."
	-@echo "========================================="
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} \
	   ACTION=testexamples_9  tree 
	-@echo "Completed compiling and running uniprocessor fortran test examples"
	-@echo "========================================="

# Ranlib on the libraries
ranlib:
	${RANLIB} ${PDIR}/*.a

# Deletes PETSc libraries
deletelibs: chkopts_basic
	-${RM} -f ${PDIR}/*


# ------------------------------------------------------------------
#
# All remaining actions are intended for PETSc developers only.
# PETSc users should not generally need to use these commands.
#


# To access the tags in EMACS, type M-x visit-tags-table and specify
# the file petsc/TAGS.	
# 1) To move to where a PETSc function is defined, enter M-. and the
#     function name.
# 2) To search for a string and move to the first occurrence,
#     use M-x tags-search and the string.
#     To locate later occurrences, use M-,

TAGS_INCLUDE_FILES  = include/*.h include/pinclude/*.h bmake/*/petscconf.h \
                      include/finclude/*.h 
TAGS_BMAKE_FILES    = bmake/common bmake/*/base*
TAGS_EXAMPLE_FILES  = src/*/examples/*/*.[c,h,F,f] src/*/examples/*/*/*.[c,h,F,f] \
                      src/benchmarks/*.c src/contrib/*/examples/*/*.[c,h,F,f]\
		      src/fortran/f90/tests/*.[c,h,F,f]
TAGS_FEXAMPLE_FILES = src/*/examples/*/*.[F,f] src/*/examples/*/*/*.[F,f] \
                      src/contrib/*/examples/*/*.[F,f]\
		      src/fortran/f90/tests/*.[F,f]
TAGS_DOC_FILES      = docs/tex/manual/routin.tex docs/tex/manual/manual.tex \
                      docs/tex/manual/manual_tex.tex docs/tex/manual/intro.tex \
                      docs/tex/manual/part1.tex docs/tex/manual/developer.tex docs/tex/manual/part2.tex
TAGS_SRC_FILES      = src/sys/src/*/*.c src/*/*.[c,h] src/*/interface/*.[c,h] src/*/src/*.[c,h] \
                      src/*/utils/*.[c,h] src/snes/mf/*.[c,h] \
                      src/*/impls/*.[c,h] src/*/impls/*/*.[c,h] src/*/impls/*/*/*.[c,h] \
                      src/snes/interface/noise/*.[c,F,h] \
		      src/contrib/*/*.[c,h] \
                      src/contrib/*/src/*.[c,h] src/fortran/custom/*.[c,h,F] \
		      src/fortran/kernels/*.[c,h,F] \
		      src/fortran/f90/*.[c,h,F] src/fortran/f90/*/*.[c,h,F] \
		      src/blaslapack/blas/*.c src/blaslapack/lapack/src[1,2,3]/*.c \
                      src/contrib/pc/*/*.c
TAGS_MAKEFILE_FILES = include/makefile include/*/makefile \
                      makefile src/sys/src/*/makefile \
                      src/makefile src/*/makefile src/*/src/makefile \
                      src/*/interface/makefile \
                      src/*/utils/makefile \
                      src/*/impls/makefile src/*/impls/*/makefile src/*/impls/*/*/makefile \
                      src/snes/interface/noise/makefile src/*/examples/makefile \
		      src/*/examples/*/makefile src/*/examples/*/*/makefile \
                      src/fortran/*/makefile src/fortran/f90/*/makefile \
                      src/contrib/*/makefile src/contrib/*/src/makefile \
                      src/contrib/*/examples/makefile src/contrib/*/examples/*/makefile \
                      src/contrib/sif/*/makefile docs/makefile src/adic/*/makefile \
                      src/contrib/pc/*/makefile src/contrib/makefile  

# Builds all etags files
alletags:
	-${OMAKE} etags_complete
	-${OMAKE} etags
	-${OMAKE} etags_noexamples
	-${OMAKE} etags_examples
	-${OMAKE} etags_makefiles
	-${OMAKE} ctags

# Builds the basic etags file.	This should be employed by most users.
etags:
	-${RM} TAGS
	-etags -f TAGS ${TAGS_INCLUDE_FILES} 
	-etags -a -f TAGS ${TAGS_SRC_FILES} 
	-etags -a -f TAGS ${TAGS_EXAMPLE_FILES} 
	-etags -a -f TAGS ${TAGS_MAKEFILE_FILES} 
	-etags -a -f TAGS ${TAGS_BMAKE_FILES} 
	-chmod g+w TAGS

# Builds complete etags list; only for PETSc developers.
etags_complete:
	-${RM} TAGS_COMPLETE
	-etags -f TAGS_COMPLETE ${TAGS_SRC_FILES} 
	-etags -a -f TAGS_COMPLETE ${TAGS_INCLUDE_FILES} 
	-etags -a -f TAGS_COMPLETE ${TAGS_EXAMPLE_FILES}
	-etags -a -f TAGS_COMPLETE ${TAGS_MAKEFILE_FILES} 
	-etags -a -f TAGS_COMPLETE ${TAGS_BMAKE_FILES} 
	-etags -a -f TAGS_COMPLETE ${TAGS_DOC_FILES}
	-chmod g+w TAGS_COMPLETE

# Builds the etags file that excludes the examples directories
etags_noexamples:
	-${RM} TAGS_NO_EXAMPLES
	-etags -f TAGS_NO_EXAMPLES ${TAGS_SRC_FILES}
	-etags -a -f TAGS_NO_EXAMPLES ${TAGS_INCLUDE_FILES} 
	-etags -a -f TAGS_NO_EXAMPLES ${TAGS_MAKEFILE_FILES} 
	-etags -a -f TAGS_NO_EXAMPLES ${TAGS_BMAKE_FILES} 
	-etags -a -f TAGS_NO_EXAMPLES ${TAGS_DOC_FILES}
	-chmod g+w TAGS_NO_EXAMPLES

# Builds the etags file for makefiles
etags_makefiles: 
	-${RM} TAGS_MAKEFILES
	-etags -f TAGS_MAKEFILES ${TAGS_MAKEFILE_FILES} 
	-etags -a -f TAGS_MAKEFILES ${TAGS_BMAKE_FILES} 
	-chmod g+w TAGS_MAKEFILES

# Builds the etags file for examples
etags_examples: 
	-${RM} TAGS_EXAMPLES
	-etags -f TAGS_EXAMPLES ${TAGS_EXAMPLE_FILES} 
	-chmod g+w TAGS_EXAMPLES
etags_fexamples: 
	-${RM} TAGS_FEXAMPLES
	-etags -f TAGS_FEXAMPLES ${TAGS_FEXAMPLE_FILES} 
	-chmod g+w TAGS_FEXAMPLES

#
# To use the tags file from VI do the following:
# 1. within vi invoke the command - :set tags=/home/bsmith/petsc/vitags
#    or add  the command to your ~/.exrc file - set tags=/home/bsmith/petsc/vitags
# 2. now to go to a tag do - :tag TAGNAME for eg - :tag MatCreate
# 
ctags:  
	-${RM} vitags
	-ctags -w -f vitags ${TAGS_INCLUDE_FILES} 
	-ctags -w -a -f vitags ${TAGS_SRC_FILES} 
	-ctags -w -a -f vitags ${TAGS_EXAMPLE_FILES}
	-ctags -w -a -f vitags ${TAGS_MAKEFILE_FILES} 
	-ctags -w -a -f vitags ${TAGS_BMAKE_FILES}
	-chmod g+w vitags
#
# These are here for the target allci and allco
#

DOCS	 = maint/addlinks maint/builddist \
	   maint/buildlinks maint/wwwman maint/xclude maint/crontab\
	   bmake/common bmake/*/base* maint/autoftp docs/manualpages/sec/* \
           include/foldinclude/generateincludes bin/petscviewinfo.text \
           bin/petscoptsinfo.text bmake/*/petscconf.h

# Deletes man pages (HTML version)
deletemanualpages:
	${RM} -f ${PETSC_DIR}/docs/manualpages/man*/* \
                 ${PETSC_DIR}/docs/manualpages/man?.html \
                 ${PETSC_DIR}/docs/manualpages/manualpages.cit 

# Deletes man pages (LaTeX version)
deletelatexpages:
	${RM} -f ${PETSC_DIR}/docs/tex/rsum/*sum*.tex

# Builds all versions of the man pages
allmanpages: allmanualpages alllatexpages
allmanualpages: deletemanualpages
	-${OMAKE} ACTION=manualpages_buildcite tree
	-cd src/fortran/custom; ${OMAKE} manualpages_buildcite
	-cd src/fortran/custom; ${OMAKE} manualpages
	-${OMAKE} ACTION=manualpages tree
	-maint/wwwman ${PETSC_DIR}
	-maint/examplesindex.tcl
	-maint/htmlkeywords.tcl
	-@chmod g+w docs/manualpages/man*/*

alllatexpages: deletelatexpages
	-${OMAKE} ACTION=latexpages tree
	-cd src/fortran/custom; ${OMAKE} latexpages
	-@chmod g+w docs/tex/rsum/*

# Builds Fortran stub files
allfortranstubs:
	-@include/foldinclude/generateincludes
	-@${RM} -f src/fortran/auto/*.c
	-${OMAKE} ACTION=fortranstubs tree
	-@cd src/fortran/auto; ${OMAKE} -f makefile fixfortran
	chmod g+w src/fortran/auto/*.c

allci: 
	-@cd src/fortran/custom ; ${OMAKE} BOPT=${BOPT} ci
	-@cd src/fortran/f90 ; ${OMAKE} BOPT=${BOPT} ci
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} ACTION=ci  tree 

allco: 
	-@cd src/fortran/custom ; ${OMAKE} BOPT=${BOPT} co
	-@cd src/fortran/f90 ; ${OMAKE} BOPT=${BOPT} co
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} ACTION=co  tree 

#
#   The commands below are for generating ADIC versions of the code;
# they are not currently used.
#
alladicignore:
	-@${RM} ${PDIR}/adicignore
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} ACTION=adicignore  tree 

alladic:
	-@echo "Beginning to compile ADIC source code in all directories"
	-@echo "Using ADIC compiler: ${ADIC_CC} ${CCPPFLAGS}"
	-@echo "========================================="
	-@cd include ; \
           ${ADIC_CC} -s -f 1 ${CCPPFLAGS} petsc.h 
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} ACTION=adic  tree 
	-@cd src/inline ; \
            ${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} adic
	-@cd src/blaslapack ; \
            ${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} ACTION=adic  tree

alladiclib:
	-@echo "Beginning to compile ADIC libraries in all directories"
	-@echo "Using compiler: ${CC} ${COPTFLAGS}"
	-@echo "-----------------------------------------"
	-@echo "Using PETSc flags: ${PETSCFLAGS} ${PCONF}"
	-@echo "-----------------------------------------"
	-@echo "Using configuration flags:"
	-@grep "define " bmake/${PETSC_ARCH}/petscconf.h
	-@echo "-----------------------------------------"
	-@echo "Using include paths: ${PETSC_INCLUDE}"
	-@echo "-----------------------------------------"
	-@echo "Using PETSc directory: ${PETSC_DIR}"
	-@echo "Using PETSc arch: ${PETSC_ARCH}"
	-@echo "========================================="
	-@${RM} -f  ${PDIR}/*adic.a
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} ACTION=adiclib  tree
	-@cd src/blaslapack ; \
            ${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} ACTION=adiclib  tree
	-@cd src/adic/src ; \
            ${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} lib

# -------------------------------------------------------------------------------
#
# Some macros to check if the fortran interface is up-to-date.
#
countfortranfunctions: 
	-@cd ${PETSC_DIR}/src/fortran; egrep '^void' custom/*.c auto/*.c | \
	cut -d'(' -f1 | tr -s '' ' ' | cut -d' ' -f2 | uniq | egrep -v "(^$$|Petsc)" | \
	sed "s/_$$//" | sort > /tmp/countfortranfunctions

countcfunctions:
	-@ grep extern ${PETSC_DIR}/include/*.h *.h | grep "(" | tr -s '' ' ' | \
	cut -d'(' -f1 | cut -d' ' -f3 | grep -v "\*" | tr -s '' '\012' |  \
	tr 'A-Z' 'a-z' |  sort > /tmp/countcfunctions

difffortranfunctions: countfortranfunctions countcfunctions
	-@echo -------------- Functions missing in the fortran interface ---------------------
	-@diff /tmp/countcfunctions /tmp/countfortranfunctions | grep "^<" | cut -d' ' -f2
	-@echo ----------------- Functions missing in the C interface ------------------------
	-@diff /tmp/countcfunctions /tmp/countfortranfunctions | grep "^>" | cut -d' ' -f2
	-@${RM}  /tmp/countcfunctions /tmp/countfortranfunctions

checkbadfortranstubs:
	-@echo "========================================="
	-@echo "Functions with MPI_Comm as an Argument"
	-@echo "========================================="
	-@cd ${PETSC_DIR}/src/fortran/auto; grep '^void' *.c | grep 'MPI_Comm' | \
	tr -s '' ' ' | tr -s ':' ' ' |cut -d'(' -f1 | cut -d' ' -f1,3
	-@echo "========================================="
	-@echo "Functions with a String as an Argument"
	-@echo "========================================="
	-@cd ${PETSC_DIR}/src/fortran/auto; grep '^void' *.c | grep 'char \*' | \
	tr -s '' ' ' | tr -s ':' ' ' |cut -d'(' -f1 | cut -d' ' -f1,3
	-@echo "========================================="
	-@echo "Functions with Pointers to PETSc Objects as Argument"
	-@echo "========================================="
	-@cd ${PETSC_DIR}/src/fortran/auto; \
	_p_OBJ=`grep _p_ ${PETSC_DIR}/include/*.h | tr -s '' ' ' | \
	cut -d' ' -f 3 | tr -s '' '\012' | grep -v '{' | cut -d'*' -f1 | \
	sed "s/_p_//g" | tr -s '\012 ' ' *|' ` ; \
	for OBJ in $$_p_OBJ; do \
	grep "$$OBJ \*" *.c | tr -s '' ' ' | tr -s ':' ' ' | \
	cut -d'(' -f1 | cut -d' ' -f1,3; \
	done 
# Builds noise routines (not yet publically available)
# Note:	 libfast cannot run on .F files on certain machines, so we
# use lib and check for errors here.
noise: info chkpetsc_dir
	-@echo "Beginning to compile noise routines"
	-@echo "========================================="
	-@cd src/snes/interface/noise; \
	  ${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} lib > trashz 2>&1; \
	  grep -v clog trashz | grep -v "information sections" | \
	  egrep -i '(Error|warning|Can)' >> /dev/null;\
	  if [ "$$?" != 1 ]; then \
	  cat trashz ; fi; ${RM} trashz
	${RANLIB} ${PDIR}/libpetscsnes.a
	-@chmod g+w  ${PDIR}/libpetscsnes.a
	-@echo "Completed compiling noise routines"
	-@echo "========================================="

