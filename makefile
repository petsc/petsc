# $Id: makefile,v 1.222 1998/04/09 04:04:17 bsmith Exp balay $ 
#
# This is the makefile for installing PETSc. See the file
# Installation for directions on installing PETSc.
# See also bmake/common for additional commands.
#
ALL: all

CFLAGS	 =
SOURCEC	 =
SOURCEF	 =
DOCS	 = maint/addlinks maint/builddist \
	   maint/buildlinks maint/wwwman maint/xclude maint/crontab\
	   bmake/common bmake/*/base* maint/autoftp docs/manualpages/sec/* \
           include/finclude/generateincludes bin/petscviewinfo.text \
           bin/petscoptsinfo.text
OBJSC	 =
OBJSF	 =
LIBBASE	 = libpetscvec
DIRS	 = src include docs 

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base


#
#  Prints information about the system and PETSc being compiled
#
info:
	-@echo "=========================================="
	-@echo On `date` on `hostname`
	-@echo Machine characteristics: `uname -a`
	-@echo "-----------------------------------------"
	-@echo "Using C compiler: ${CC} ${COPTFLAGS}"
	-@if [ -n "${CCV}" -a "${CCV}" != "unknown" ] ; then \
	  echo "Compiler version:" `${CCV}` ; fi
	-@echo "Using Fortran compiler: ${FC} ${FFLAGS} ${FOPTFLAGS}"
	-@echo "-----------------------------------------"
	-@grep PETSC_VERSION_NUMBER include/petsc.h | sed "s/........//"
	-@echo "-----------------------------------------"
	-@echo "Using PETSc flags: ${PETSCFLAGS} ${PCONF}"
	-@echo "-----------------------------------------"
	-@echo "Using configuration flags: ${CONF}"
	-@if [ -r bmake/${PETSC_ARCH}/petscconf.h ] ; then \
	  grep "define " bmake/${PETSC_ARCH}/petscconf.h ; fi
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

# Builds PETSc libraries for a given BOPT and architecture
all: info chkpetsc_dir deletelibs build_kernels build_c shared build_fortran build_fortran90

build_c:
	-@echo "BEGINNING TO COMPILE LIBRARIES IN ALL DIRECTORIES"
	-@echo "========================================="
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} \
	   ACTION=libfast  tree 
	-@cd ${PETSC_DIR}/src/sys/src ; \
	${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} rs6000_time
	${RANLIB} ${PDIR}/*.a
	-@chmod g+w  ${PDIR}/*.a
	-@echo "Completed building libraries"
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
testexamples_uni: info chkopts
	-@echo "BEGINNING TO COMPILE AND RUN TEST UNI-PROCESSOR EXAMPLES"
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines some of the numbers may not match exactly."
	-@echo "========================================="
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} \
	   ACTION=testexamples_4  tree 
	-@echo "Completed compiling and running uniprocessor test examples"
	-@echo "========================================="

#
# Builds PETSc Fortran interface libary
# Note:	 libfast cannot run on .F files on certain machines, so we
# use lib and check for errors here.
fortran: info chkpetsc_dir build_fortran

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
# Note:	 libfast cannot run on .F files on certain machines, so we
# use lib and check for errors here.
# Note: F90 interface currently only supported in NAG F90 compiler
fortran90: fortran build_fortran90

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
fortrankernels: info chkpetsc_dir 
	-${RM} -f ${PDIR}/libpetsckernels.*
	-@echo "BEGINNING TO COMPILE FORTRAN KERNELS LIBRARY"
	-@echo "========================================="
	-@cd src/fortran/kernels; \
	  ${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} lib
	-@chmod g+w  ${PDIR}/*.a
	-@echo "Completed compiling Fortran kernels library"
	-@echo "========================================="

# Builds PETSc test examples for a given BOPT and architecture
testfortran: info chkopts
	-@echo "BEGINNING TO COMPILE AND RUN FORTRAN TEST EXAMPLES"
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
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} \
	   ACTION=testexamples_3  tree 
	-@echo "Completed compiling and running Fortran test examples"
	-@echo "========================================="
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

petscblas: info chkpetsc_dir
	-${RM} -f ${PDIR}/libpetscblas.*
	-@echo "BEGINNING TO COMPILE C VERSION OF BLAS AND LAPACK"
	-@echo "========================================="
	-@cd src/adic/blas; \
	  ${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} libfast
	-@cd src/adic/lapack; \
	  ${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} ACTION=libfast tree
	${RANLIB} ${PDIR}/libpetscblas.a
	-@chmod g+w  ${PDIR}/*.a
	-@echo "Completed compiling C version of BLAS and LAPACK"
	-@echo "========================================="

# If USE_DYNAMIC_LIBRARIES flag is set, build shared libs
build_shared:
	-@shared_var=`echo "${PETSCFLAGS}" | sed 's/-DUSE_DYNAMIC_LIBRARIES//g'`; \
	if [ "${PETSCFLAGS}" != "$$shared_var" ] ; then \
	${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} shared; fi

# If fortrankernels are used, build them.
build_kernels:
	-@kernel_var=`echo "${PETSCFLAGS}" | sed 's/-DUSE_FORTRAN_KERNELS//g'`; \
	if [ "${PETSCFLAGS}" != "$$kernel_var" ] ; then \
	${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} fortrankernels; fi

# Ranlib on the libraries
ranlib:
	${RANLIB} ${PDIR}/*.a

# Deletes PETSc libraries
deletelibs: chkopts_basic
	-${RM} -f ${PDIR}/*

# Deletes man pages (HTML version)
deletemanualpages:
	${RM} -f ${PETSC_DIR}/docs/manualpages/man*/* ${PETSC_DIR}/docs/manualpages/www.cit \
	         ${PETSC_DIR}/docs/manualpages/man*.html

# Deletes man pages (LaTeX version)
deletelatexpages:
	${RM} -f ${PETSC_DIR}/docs/tex/rsum/*sum*.tex

# To access the tags in EMACS, type M-x visit-tags-table and specify
# the file petsc/TAGS.	
# 1) To move to where a PETSc function is defined, enter M-. and the
#     function name.
# 2) To search for a string and move to the first occurrence,
#     use M-x tags-search and the string.
#     To locate later occurrences, use M-,

TAGS_INCLUDE_FILES  = include/*.h include/pinclude/*.h include/FINCLUDE/*.h 
TAGS_BMAKE_FILES    = bmake/common bmake/*/base*
TAGS_EXAMPLE_FILES  = src/*/examples/*/*.[c,h,F,f] src/*/examples/*/*/*.[c,h,F,f] \
                      src/benchmarks/*.c src/contrib/*/examples/*/*.[c,h,F,f]\
		      src/fortran/f90/tests
TAGS_DOC_FILES      = docs/tex/manual/routin.tex docs/tex/manual/manual.tex \
                      docs/tex/manual/manual_tex.tex docs/tex/manual/intro.tex \
                      docs/tex/manual/part1.tex docs/tex/manual/part2.tex
TAGS_SRC_FILES      = src/*/*.[c,h] src/*/interface/*.[c,h] src/*/src/*.[c,h] \
                      src/*/utils/*.[c,h] \
                      src/*/impls/*.[c,h] src/*/impls/*/*.[c,h] src/*/impls/*/*/*.[c,h] \
                      src/snes/interface/noise/*.[c,F,h] src/gvec/impls/*/*/*/*/*.[c,h] \
		      src/contrib/*/*.[c,h] \
                      src/contrib/*/src/*.[c,h] src/fortran/custom/*.[c,h,F] \
		      src/fortran/kernels/*.[c,h,F] \
		      src/fortran/f90/*.[c,h,F] src/fortran/f90/*/*.[c,h,F] \
		      src/adic/blas/*.c src/lapack/src[1,2,3]/*.c
TAGS_MAKEFILE_FILES = include/makefile include/*/makefile \
                      makefile \
                      src/makefile src/*/makefile src/*/src/makefile \
                      src/*/interface/makefile \
                      src/*/utils/makefile \
                      src/*/impls/makefile src/*/impls/*/makefile src/*/impls/*/*/makefile \
                      src/snes/interface/noise/makefile src/*/examples/makefile \
		      src/*/examples/*/makefile src/*/examples/*/*/makefile \
                      src/gvec/impls/*/*/*/*/makefile src/gvec/impls/*/*/*/makefile \
                      src/fortran/*/makefile src/fortran/f90/*/makefile \
                      src/contrib/*/makefile src/contrib/*/src/makefile \
                      src/contrib/*/examples/makefile src/contrib/*/examples/*/makefile \
                      src/contrib/sif/*/makefile docs/makefile src/adic/*/makefile \
		      src/adic/lapack/*/makefile 

# Builds all etags files
alletags:
	-make etags_complete
	-make etags
	-make etags_noexamples
	-make etags_examples
	-make etags_makefiles

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

#
# To use the tags file from VI do the following:
# 1. within vi invoke the command - :set tags=/home/bsmith/petsc/tags
#    or add  the command to your ~/.exrc file - set tags=/home/bsmith/petsc/tags
# 2. now to go to a tag do - :tag TAGNAME for eg - :tag MatCreate
# 
ctags:  
	-${RM} tags
	-ctags -w -f tags ${TAGS_INCLUDE_FILES} 
	-ctags -w -a -f tags ${TAGS_SRC_FILES} 
	-ctags -w -a -f tags ${TAGS_EXAMPLE_FILES}
	-ctags -w -a -f tags ${TAGS_MAKEFILE_FILES} 
	-ctags -w -a -f tags ${TAGS_BMAKE_FILES}
	-chmod g+w tags

# ------------------------------------------------------------------
#
# All remaining actions are intended for PETSc developers only.
# PETSc users should not generally need to use these commands.
#

# Builds all versions of the man pages
allmanpages: allmanualpages alllatexpages
allmanualpages: deletemanualpages
	-make ACTION=manualpages_buildcite tree
	-cd src/fortran/custom; make manualpages_buildcite
	-cd src/fortran/custom; make manualpages
	-make ACTION=manualpages tree
	-maint/wwwman
	-maint/examplesindex.tcl
	-maint/htmlkeywords.tcl
	-@chmod g+w docs/manualpages/man*/*

alllatexpages: deletelatexpages
	-make ACTION=latexpages tree
	-cd src/fortran/custom; make latexpages
	-@chmod g+w docs/tex/rsum/*

# Builds Fortran stub files
allfortranstubs:
	-@include/finclude/generateincludes
	-@${RM} -f src/fortran/auto/*.c
	-make ACTION=fortranstubs tree
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
CFLAGS   =  ${CPPFLAGS} ${CONF}
alladicignore:
	-@${RM} ${PDIR}/adicignore
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} ACTION=adicignore  tree 

alladic:
	-@echo "Beginning to compile ADIC source code in all directories"
	-@echo "Using ADIC compiler: ${ADIC_CC} ${CFLAGS}"
	-@echo "========================================="
	-@cd include ; \
           ${ADIC_CC} -s -f 1 ${CFLAGS} petsc.h 
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} ACTION=adic  tree 
	-@cd src/inline ; \
            ${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} adic
	-@cd src/adic/blas ; \
            ${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} adic
	-@cd src/adic/lapack ; \
            ${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} ACTION=adic  tree

alladiclib:
	-@echo "Beginning to compile ADIC libraries in all directories"
	-@echo "Using compiler: ${CC} ${COPTFLAGS}"
	-@echo "-----------------------------------------"
	-@echo "Using PETSc flags: ${PETSCFLAGS} ${PCONF}"
	-@echo "-----------------------------------------"
	-@echo "Using configuration flags: ${CONF}"
	-@echo "-----------------------------------------"
	-@echo "Using include paths: ${PETSC_INCLUDE}"
	-@echo "-----------------------------------------"
	-@echo "Using PETSc directory: ${PETSC_DIR}"
	-@echo "Using PETSc arch: ${PETSC_ARCH}"
	-@echo "========================================="
	-@${RM} -f  ${PDIR}/*adic.a
	-@${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} ACTION=adiclib  tree
	-@cd src/adic/blas ; \
            ${OMAKE} BOPT=${BOPT} PETSC_ARCH=${PETSC_ARCH} adiclib
	-@cd src/adic/lapack ; \
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
