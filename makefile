#
# This is the makefile for compiling PETSc. See 
# http://www.mcs.anl.gov/petsc/petsc-as/documentation/installation.html for directions on installing PETSc.
# See also bmake/common for additional commands.
#
ALL: all
LOCDIR	 = ./
DIRS	 = src include 
CFLAGS	 = 
FFLAGS	 = 

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

#
# Basic targets to build PETSc libraries.
# all: builds the c, fortran, and f90 libraries
all: 
	@${OMAKE}  PETSC_ARCH=${PETSC_ARCH}  chkpetsc_dir
	-@${OMAKE} all_build 2>&1 | tee make_log_${PETSC_ARCH}
	-@egrep -i "( error | error:)" make_log_${PETSC_ARCH} > /dev/null; if [ "$$?" = "0" ]; then \
           echo "********************************************************************"; \
           echo "  Error during compile, check make_log_${PETSC_ARCH}"; \
           echo "  Send it and configure.log to petsc-maint@mcs.anl.gov";\
           echo "********************************************************************"; \
           exit 1; fi

all_build: chk_petsc_dir chklib_dir info info_h deletelibs  build shared
#
# Prints information about the system and version of PETSc being compiled
#
info:
	-@echo "=========================================="
	-@echo " "
	-@echo "See docs/faq.html and docs/bugreporting.html"
	-@echo "for help with installation problems. Please send EVERYTHING"
	-@echo "printed out below when reporting problems"
	-@echo " "
	-@echo "To subscribe to the PETSc users mailing list, send mail to "
	-@echo "majordomo@mcs.anl.gov with the message: "
	-@echo "subscribe petsc-announce"
	-@echo " "
	-@echo "=========================================="
	-@echo On `date` on `hostname`
	-@echo Machine characteristics: `uname -a`
	-@echo "-----------------------------------------"
	-@echo "Using PETSc directory: ${PETSC_DIR}"
	-@echo "Using PETSc arch: ${PETSC_ARCH}"
	-@echo "-----------------------------------------"
	-@grep "define PETSC_VERSION" ${PETSC_DIR}/include/petscversion.h | ${SED} "s/........//"
	-@echo "-----------------------------------------"
	-@echo "Using configure Options: ${CONFIGURE_OPTIONS}"
	-@echo "Using configuration flags:"
	-@grep "\#define " ${PETSC_DIR}/bmake/${PETSC_ARCH}/petscconf.h
	-@echo "-----------------------------------------"
	-@echo "Using include paths: ${PETSC_INCLUDE}"
	-@echo "------------------------------------------"
	-@echo "Using C/C++ compiler: ${CC} ${CC_FLAGS} ${COPTFLAGS} ${CFLAGS}"
	-@echo "C/C++ Compiler version: " `${CCV}`
	-@if [ "${FC}" != "" ]; then \
	   echo "Using Fortran compiler: ${FC} ${FC_FLAGS} ${FFLAGS} ${FPP_FLAGS}";\
	   echo "Fortran Compiler version: " `${FCV}`;\
         fi
	-@echo "-----------------------------------------"
	-@echo "Using C/C++ linker: ${CC_LINKER}"
	-@if [ "${FC}" != "" ]; then \
	   echo "Using Fortran linker: ${FC_LINKER}";\
         fi
	-@echo "-----------------------------------------"
	-@echo "Using libraries: ${PETSC_LIB}"
	-@echo "------------------------------------------"
	-@echo "Using mpirun: ${MPIRUN}"
	-@echo "=========================================="
#
#
MINFO = ${PETSC_DIR}/bmake/${PETSC_ARCH}/petscmachineinfo.h
info_h:
	-@$(RM) -f ${MINFO} MINFO
	-@echo  "static const char *petscmachineinfo = \"\__n__\"" >> MINFO
	-@echo  "\"-----------------------------------------\__n__\"" >> MINFO
	-@echo  "\"Libraries compiled on `date` on `hostname` \__n__\"" >> MINFO
	-@echo  "\"Machine characteristics: `uname -a` \__n__\"" >> MINFO
	-@echo  "\"Using PETSc directory: ${PETSC_DIR}\__n__\"" >> MINFO
	-@echo  "\"Using PETSc arch: ${PETSC_ARCH}\__n__\"" >> MINFO
	-@echo  "\"-----------------------------------------\"; " >> MINFO
	-@echo  "static const char *petsccompilerinfo = \"\__n__\"" >> MINFO
	-@echo  "\"Using C compiler: ${CC} ${CC_FLAGS} ${COPTFLAGS} ${CFLAGS}\__n__\"" >> MINFO
	-@echo  "\"Using Fortran compiler: ${FC} ${FC_FLAGS} ${FFLAGS} ${FPP_FLAGS}\__n__\"" >> MINFO
	-@echo  "\"-----------------------------------------\"; " >> MINFO
	-@echo  "static const char *petsccompilerflagsinfo = \"\__n__\"" >> MINFO
	-@echo  "\"Using include paths: ${PETSC_INCLUDE}\__n__\"" >> MINFO
	-@echo  "\"------------------------------------------\"; " >> MINFO
	-@echo  "static const char *petsclinkerinfo = \"\__n__\"" >> MINFO
	-@echo  "\"Using C linker: ${CLINKER}\__n__\"" >> MINFO
	-@echo  "\"Using Fortran linker: ${FLINKER}\__n__\"" >> MINFO
	-@echo  "\"Using libraries: ${PETSC_LIB} \__n__\"" >> MINFO
	-@echo  "\"------------------------------------------\"; " >> MINFO
	-@cat MINFO | ${SED} -e 's/\\ /\\\\ /g' | ${SED} -e 's/__n__/n/g' > ${MINFO}
	-@ if [ -f /usr/bin/cygcheck.exe ]; then /usr/bin/dos2unix ${MINFO} 2> /dev/null; fi
	-@$(RM) -f MINFO

#
# Builds the PETSc libraries
# This target also builds fortran77 and f90 interface
# files and compiles .F files
#
build:
	-@echo "BEGINNING TO COMPILE LIBRARIES IN ALL DIRECTORIES"
	-@echo "========================================="
	-@${OMAKE}  PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} ACTION=libfast tree
	-@${RANLIB} ${PETSC_LIB_DIR}/*.${AR_LIB_SUFFIX}
	-@echo "Completed building libraries"
	-@echo "========================================="
#
# Builds the Python wrappers
python:
	-@if [ -d "${PETSC_DIR}/src/python/PETSc" ]; then \
	  echo "COMPILING PYTHON WRAPPERS"; \
	  echo "========================================="; \
	  PYTHONPATH=${PYTHONPATH}:${PETSC_DIR}/python/BuildSystem ./make.py --with-petsc-arch=${PETSC_ARCH} --ignoreWarnings 0; \
	  echo "Completed building Python wrappers"; \
	  echo "========================================="; \
	fi
#
# Builds PETSc test examples for a given architecture
#
test: 
	-@echo "Running test examples to verify correct installation"
	@cd src/snes/examples/tutorials; ${OMAKE} PETSC_ARCH=${PETSC_ARCH}  PETSC_DIR=${PETSC_DIR} testex19
	@if [ "${FC}" != "" ]; then cd src/snes/examples/tutorials; ${OMAKE} PETSC_ARCH=${PETSC_ARCH}  PETSC_DIR=${PETSC_DIR} testex5f; fi;
	-@echo "Completed test examples"

testexamples: info 
	-@echo "BEGINNING TO COMPILE AND RUN TEST EXAMPLES"
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines some of the numbers may not match exactly."
	-@echo "========================================="
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH}  PETSC_DIR=${PETSC_DIR} ACTION=testexamples_C  tree 
	-@echo "Completed compiling and running test examples"
	-@echo "========================================="
testfortran: info 
	-@echo "BEGINNING TO COMPILE AND RUN FORTRAN TEST EXAMPLES"
	-@echo "========================================="
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines or the way Fortran formats numbers"
	-@echo "some of the results may not match exactly."
	-@echo "========================================="
	-@if [ "${FC}" != "" ]; then \
            ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} ACTION=testexamples_Fortran tree; \
            echo "Completed compiling and running Fortran test examples"; \
          else \
            echo "Error: No FORTRAN compiler available"; \
          fi
	-@echo "========================================="
testexamples_uni: info 
	-@echo "BEGINNING TO COMPILE AND RUN TEST UNI-PROCESSOR EXAMPLES"
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines some of the numbers may not match exactly."
	-@echo "========================================="
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} ACTION=testexamples_C_X11_MPIUni  tree 
	-@echo "Completed compiling and running uniprocessor test examples"
	-@echo "========================================="
testfortran_uni: info 
	-@echo "BEGINNING TO COMPILE AND RUN TEST UNI-PROCESSOR FORTRAN EXAMPLES"
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines some of the numbers may not match exactly."
	-@echo "========================================="
	-@if [ "${FC}" != "" ]; then \
            ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} ACTION=testexamples_Fortran_MPIUni  tree; \
            echo "Completed compiling and running uniprocessor fortran test examples"; \
          else \
            echo "Error: No FORTRAN compiler available"; \
          fi
	-@
	-@echo "========================================="

# Ranlib on the libraries
ranlib:
	${RANLIB} ${PETSC_LIB_DIR}/*.${AR_LIB_SUFFIX}

# Deletes PETSc libraries
deletelibs: 
	-${RM} -f ${PETSC_LIB_DIR}/lib*.*

# Cleans up build
allclean: deletelibs
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} ACTION=clean tree


#
# Check if PETSC_DIR variable specified is valid
#
chk_petsc_dir:
	@if [ ! -f ${PETSC_DIR}/include/petscversion.h ]; then \
	  echo "Incorrect PETSC_DIR specified: ${PETSC_DIR}!"; \
	  echo "You need to use / to separate directories, not \\!"; \
	  echo "Aborting build"; \
	  false; fi
#
#
install:
	-@if [ "${INSTALL_DIR}" = "${PETSC_DIR}" ]; then \
	  echo "Install directory is current directory; nothing needs to be done";\
        else \
	  echo Installing PETSc at ${INSTALL_DIR};\
          if [ ! -d `dirname ${INSTALL_DIR}` ]; then \
	    ${MKDIR} `dirname ${INSTALL_DIR}` ; \
          fi;\
          if [ ! -d ${INSTALL_DIR} ]; then \
	    ${MKDIR} ${INSTALL_DIR} ; \
          fi;\
          cp -fr include ${INSTALL_DIR};\
          if [ ! -d ${INSTALL_DIR}/bmake ]; then \
	    ${MKDIR} ${INSTALL_DIR}/bmake ; \
          fi;\
          cp -f bmake/adic* bmake/petscconf ${INSTALL_DIR}/bmake ; \
          cp -fr bmake/common ${INSTALL_DIR}/bmake;\
          cp -fr bmake/${PETSC_ARCH} ${INSTALL_DIR}/bmake;\
          cp -fr bin ${INSTALL_DIR};\
          if [ ! -d ${INSTALL_DIR}/lib ]; then \
	    ${MKDIR} ${INSTALL_DIR}/lib ; \
          fi;\
          if [ -d lib/${PETSC_ARCH} ]; then \
            cp -fr lib/${PETSC_ARCH} ${INSTALL_DIR}/lib;\
            ${RANLIB} ${INSTALL_DIR}/lib/${PETSC_ARCH}/*.a ;\
            ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${INSTALL_DIR} shared; \
          fi;\
          echo "sh/bash: PETSC_DIR="${INSTALL_DIR}"; export PETSC_DIR";\
          echo "csh/tcsh: setenv PETSC_DIR "${INSTALL_DIR} ;\
          echo "Then do make test to verify correct install";\
        fi;

install_src:
	-@if [ "${INSTALL_DIR}" = "${PETSC_DIR}" ]; then \
	  echo "You did not set a directory to install to";\
        else \
	  echo Installing PETSc source at ${INSTALL_DIR};\
          if [ ! -d `dirname ${INSTALL_DIR}` ]; then \
	    ${MKDIR} `dirname ${INSTALL_DIR}` ; \
          fi;\
          if [ ! -d ${INSTALL_DIR} ]; then \
	    ${MKDIR} ${INSTALL_DIR} ; \
          fi;\
          cp -fr src ${INSTALL_DIR};\
        fi;

install_docs:
	-@if [ "${INSTALL_DIR}" = "${PETSC_DIR}" ]; then \
	  echo "You did not set a directory to install to";\
        else \
	  echo Installing PETSc documentation at ${INSTALL_DIR};\
          if [ ! -d `dirname ${INSTALL_DIR}` ]; then \
	    ${MKDIR} `dirname ${INSTALL_DIR}` ; \
          fi;\
          if [ ! -d ${INSTALL_DIR} ]; then \
	    ${MKDIR} ${INSTALL_DIR} ; \
          fi;\
          cp -fr docs ${INSTALL_DIR};\
          ${RM} -fr docs/tex;\
        fi;
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
# Builds all etags files
alletags:
	-@maint/generateetags.py
	-@find python -type f -name "*.py" |grep -v SCCS | xargs etags -o TAGS_PYTHON

allfortranstubs:
	-@maint/generatefortranstubs.py ${BFORT}
#
# These are here for the target allci and allco, and etags
#

BMAKEFILES = bmake/common/base bmake/common/test bmake/adic.init bmake/adicmf.init
DOCS	   = bmake/readme
SCRIPTS    = maint/builddist  maint/wwwman maint/xclude maint/bugReport.py maint/buildconfigtest maint/builddistlite \
             maint/buildtest maint/checkBuilds.py maint/copylognightly maint/copylognightly.tao maint/countfiles maint/findbadfiles \
             maint/fixinclude maint/getexlist maint/getpdflabels.py maint/helpindex.py maint/hosts.local maint/hosts.solaris  \
             maint/lex.py  maint/mapnameslatex.py maint/startnightly maint/startnightly.tao maint/submitPatch.py \
             maint/update-docs.py  maint/wwwindex.py maint/xcludebackup maint/xcludecblas maint/zap maint/zapall \
             python/PETSc/Configure.py python/PETSc/Options.py \
             python/PETSc/packages/*.py python/PETSc/utilities/*.py

chk_loc:
	@if [ ${LOC}foo = foo ] ; then \
	  echo "*********************** ERROR ************************" ; \
	  echo " Please specify LOC variable for eg: make allmanualpages LOC=/sandbox/petsc"; \
	  echo "******************************************************";  false; fi
	@${MKDIR} ${LOC}/docs/manualpages

# Builds all the documentation - should be done every night
alldoc: alldoc1 alldoc2

# Build everything that goes into 'doc' dir except html sources
alldoc1: chk_loc deletemanualpages chk_concepts_dir
	-${OMAKE} ACTION=manualpages_buildcite tree_basic LOC=${LOC}
	-@sed -e s%man+../%man+manualpages/% ${LOC}/docs/manualpages/manualpages.cit > ${LOC}/docs/manualpages/htmlmap
	-@cat ${PETSC_DIR}/src/docs/mpi.www.index >> ${LOC}/docs/manualpages/htmlmap
	cd src/docs/tex/manual; ${OMAKE} manual.pdf LOC=${LOC}
	-${OMAKE} ACTION=manualpages tree_basic LOC=${LOC}
	-maint/wwwindex.py ${PETSC_DIR} ${LOC}
	-${OMAKE} ACTION=manexamples tree_basic LOC=${LOC}
	-${OMAKE} manconcepts LOC=${LOC}
	-${OMAKE} ACTION=getexlist tree_basic LOC=${LOC}
	-${OMAKE} ACTION=exampleconcepts tree_basic LOC=${LOC}
	-maint/helpindex.py ${PETSC_DIR} ${LOC}
	-grep -h Polymorphic include/*.h | grep -v '#define ' | sed "s?PetscPolymorphic[a-zA-Z]*(??g" | cut -f1 -d"{" > tmppoly
	-maint/processpoly.py ${PETSC_DIR} ${LOC}
	-${RM} tmppoly

# Builds .html versions of the source
# html overwrites some stuff created by update-docs - hence this is done later.
alldoc2: chk_loc
	-${OMAKE} ACTION=html PETSC_DIR=${PETSC_DIR} alltree LOC=${LOC}
	-maint/update-docs.py ${PETSC_DIR} ${LOC}

alldocclean: deletemanualpages allcleanhtml

# Deletes man pages (HTML version)
deletemanualpages: chk_loc
	-@if [ -d ${LOC} -a -d ${LOC}/docs/manualpages ]; then \
          find ${LOC}/docs/manualpages -type f -name "*.html" -exec ${RM} {} \; ;\
          ${RM} ${LOC}/docs/exampleconcepts ;\
          ${RM} ${LOC}/docs/manconcepts ;\
          ${RM} ${LOC}/docs/manualpages/manualpages.cit ;\
          maint/update-docs.py ${PETSC_DIR} ${LOC} clean;\
        fi

allcleanhtml: 
	-${RM} include/adic/*.h.html 
	-${OMAKE} ACTION=cleanhtml PETSC_DIR=${PETSC_DIR} alltree

chk_concepts_dir: chk_loc
	@if [ ! -d "${LOC}/docs/manualpages/concepts" ]; then \
	  echo Making directory ${LOC}/docs/manualpages/concepts for library; ${MKDIR} ${LOC}/docs/manualpages/concepts; fi
#
#  makes .lines files for all source code
# 
allgcov: 
	-@${RM} -rf /tmp/gcov
	-@mkdir /tmp/gcov
	-${OMAKE} ACTION=gcov PETSC_DIR=${PETSC_DIR} tree

# usage make allrcslabel NEW_RCS_LABEL=v_2_0_28
allrcslabel: 
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH} NEW_RCS_LABEL=${NEW_RCS_LABEL} ACTION=rcslabel  alltree 
#
#   The commands below are for generating ADIC versions of the code;
# they are not currently used.
#
alladicignore:
	-@${RM} ${INSTALL_LIB_DIR}/adicignore
	-@${OMAKE}  PETSC_ARCH=${PETSC_ARCH} ACTION=adicignore  tree 

alladic:
	-@echo "Beginning to compile ADIC source code in all directories"
	-@echo "Using ADIC compiler: ${ADIC_CC} ${CCPPFLAGS}"
	-@echo "========================================="
	-@cd include ; \
           ${ADIC_CC} -s -f 1 ${CCPPFLAGS} petsc.h 
	-@${OMAKE}  PETSC_ARCH=${PETSC_ARCH} ACTION=adic  tree 
	-@cd src/inline ; \
            ${OMAKE} PETSC_ARCH=${PETSC_ARCH} adic

alladiclib:
	-@echo "Beginning to compile ADIC libraries in all directories"
	-@echo "Using compiler: ${CC} ${COPTFLAGS}"
	-@echo "-----------------------------------------"
	-@echo "Using PETSc flags: ${PETSCFLAGS} ${PCONF}"
	-@echo "-----------------------------------------"
	-@echo "Using configuration flags:"
	-@grep "define " bmake/${INLUDE_ARCH}/petscconf.h
	-@echo "-----------------------------------------"
	-@echo "Using include paths: ${PETSC_INCLUDE}"
	-@echo "-----------------------------------------"
	-@echo "Using PETSc directory: ${PETSC_DIR}"
	-@echo "Using PETSc arch: ${PETSC_ARCH}"
	-@echo "========================================="
	-@${RM} -f  ${INSTALL_LIB_DIR}/*adic.${AR_LIB_SUFFIX}
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH} ACTION=adiclib  tree
	-@cd src/adic/src ; \
            ${OMAKE} PETSC_ARCH=${PETSC_ARCH} lib

# -------------------------------------------------------------------------------
#
# Some macros to check if the fortran interface is up-to-date.
#
countfortranfunctions: 
	-@cd ${PETSC_DIR}/src/fortran; egrep '^void' custom/*.c auto/*.c | \
	cut -d'(' -f1 | tr -s  ' ' | cut -d' ' -f2 | uniq | egrep -v "(^$$|Petsc)" | \
	sed "s/_$$//" | sort > /tmp/countfortranfunctions

countcfunctions:
	-@ grep extern ${PETSC_DIR}/include/*.h *.h | grep "(" | tr -s ' ' | \
	cut -d'(' -f1 | cut -d' ' -f3 | grep -v "\*" | tr -s '\012' |  \
	tr 'A-Z' 'a-z' |  sort > /tmp/countcfunctions

difffortranfunctions: countfortranfunctions countcfunctions
	-@echo -------------- Functions missing in the fortran interface ---------------------
	-@${DIFF} /tmp/countcfunctions /tmp/countfortranfunctions | grep "^<" | cut -d' ' -f2
	-@echo ----------------- Functions missing in the C interface ------------------------
	-@${DIFF} /tmp/countcfunctions /tmp/countfortranfunctions | grep "^>" | cut -d' ' -f2
	-@${RM}  /tmp/countcfunctions /tmp/countfortranfunctions

checkbadfortranstubs:
	-@echo "========================================="
	-@echo "Functions with MPI_Comm as an Argument"
	-@echo "========================================="
	-@cd ${PETSC_DIR}/src/fortran/auto; grep '^void' *.c | grep 'MPI_Comm' | \
	tr -s ' ' | tr -s ':' ' ' |cut -d'(' -f1 | cut -d' ' -f1,3
	-@echo "========================================="
	-@echo "Functions with a String as an Argument"
	-@echo "========================================="
	-@cd ${PETSC_DIR}/src/fortran/auto; grep '^void' *.c | grep 'char \*' | \
	tr -s ' ' | tr -s ':' ' ' |cut -d'(' -f1 | cut -d' ' -f1,3
	-@echo "========================================="
	-@echo "Functions with Pointers to PETSc Objects as Argument"
	-@echo "========================================="
	-@cd ${PETSC_DIR}/src/fortran/auto; \
	_p_OBJ=`grep _p_ ${PETSC_DIR}/include/*.h | tr -s ' ' | \
	cut -d' ' -f 3 | tr -s '\012' | grep -v '{' | cut -d'*' -f1 | \
	sed "s/_p_//g" | tr -s '\012 ' ' *|' ` ; \
	for OBJ in $$_p_OBJ; do \
	grep "$$OBJ \*" *.c | tr -s ' ' | tr -s ':' ' ' | \
	cut -d'(' -f1 | cut -d' ' -f1,3; \
	done 
#
# Automatically generates PETSc exercises in html from the tutorial examples.
#
# The introduction for each section is obtained from docs/manualpages/header_${MANSEC} is under RCS and may be edited
#  (used also in introductions to the manual pages)
# The overall introduction is in docs/exercises/introduction.html and is under RCS and may be edited
# The list of exercises is from TUTORIALS in each directory's makefile
#
# DO NOT EDIT the pageform.txt or *.htm files generated since they will be automatically replaced.
# The pagemaker rule is in the file bmake/common (at the bottom)
#
# Eventually the line below will replace the two cd in the rule below, it is just this way now for speed
#	-@${OMAKE} PETSC_DIR=${PETSC_DIR} pagemaker
#
exercises:
	-@echo "========================================="
	-@echo "Generating HTML tutorial exercises"
	-@${RM} docs/pageform.txt
	-@echo "title=\"PETSc Exercises\""                >  docs/pageform.txt 
	-@echo "access_title=Exercise Sections"              >>  docs/pageform.txt 
	-@echo "access_format=short"                        >> docs/pageform.txt
	-@echo "startpage=../exercises/introduction.htm"  >> docs/pageform.txt
	-@echo "NONE title=\"Introduction\" command=link src=../exercises/introduction.htm" >> docs/pageform.txt
	-@echo "Generating HTML for individual directories"
	-@echo "========================================="
	-@${OMAKE} PETSC_DIR=${PETSC_DIR} ACTION=pagemaker tree
	-@echo "Completed HTML for individual directories"
	-@echo "NONE title=\"<HR>\" " >> docs/pageform.txt; 
	-@echo "NONE title=\"PETSc Documentation\" command=link src=../index.html target=replace" >> docs/pageform.txt
	/home/MPI/class/mpiexmpl/maint/makepage.new -pageform=docs/pageform.txt -access_extra=/dev/null -outdir=docs/exercises
	-@echo "========================================="

# Make a tarball of all the Python code
#   This is currently used to release to the Teragrid
petscPython.tgz:
	@tar cvzf $@ --exclude SCCS --exclude BitKeeper --dereference python/
	-@scp $@ tg-login2.uc.teragrid.org:./

.PHONY: info info_h all all_build build testexamples testfortran testexamples_uni testfortran_uni ranlib deletelibs allclean update chk_petsc_dir \
        alletags etags etags_complete etags_noexamples etags_makefiles etags_examples etags_fexamples alldoc allmanualpages \
        allhtml allcleanhtml  allci allco allrcslabel alladicignore alladic alladiclib countfortranfunctions \
        start_configure configure_petsc configure_clean python

