#
# This is the makefile for installing PETSc. See 
# http://www.mcs.anl.gov/petsc/petsc-2/documentation/installation.html for directions on installing PETSc.
# See also bmake/common for additional commands.
#
ALL: all
LOCDIR = . 
DIRS   = src include 

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

#
# Basic targets to build PETSc libraries.
# all: builds the c, fortran, and f90 libraries
all: 
	@${OMAKE}  PETSC_ARCH=${PETSC_ARCH}  chkpetsc_dir
	-@${OMAKE} all_build 2>&1 | tee make_log_${PETSC_ARCH}
all_build: chk_petsc_dir chklib_dir info info_h deletelibs  build shared
#
# Prints information about the system and version of PETSc being compiled
#
info:
	-@echo "=========================================="
	-@echo " "
	-@echo "See docs/troubleshooting.html and docs/bugreporting.html"
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
	-@echo "Using configuration flags:"
	-@grep "\#define " ${PETSC_DIR}/bmake/${PETSC_ARCH}/petscconf.h
	-@echo "-----------------------------------------"
	-@echo "Using include paths: ${PETSC_INCLUDE}"
	-@echo "Using PETSc flags: ${PETSCFLAGS} ${PCONF}"
	-@echo "------------------------------------------"
	-@echo "Using C/C++ compiler: ${CC} ${COPTFLAGS} ${CPPFLAGS}"
	-@echo "C/C++ Compiler version: " `${CCV}`
	-@if [ "${FC}" != "" ]; then \
	   echo "Using Fortran compiler: ${FC} ${FOPTFLAGS} ${FPPFLAGS}";\
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
	-@$(RM) -f MINFO ${MINFO}
	-@echo  "static const char *petscmachineinfo = \"  " >> MINFO
	-@echo  "Libraries compiled on `date` on `hostname` " >> MINFO
	-@echo  Machine characteristics: `uname -a` "" >> MINFO
	-@echo  "Using PETSc directory: ${PETSC_DIR}" >> MINFO
	-@echo  "Using PETSc arch: ${PETSC_ARCH}" >> MINFO
	-@echo  "-----------------------------------------\"; " >> MINFO
	-@echo  "static const char *petsccompilerinfo = \"  " >> MINFO
	-@echo  "Using C compiler: ${CC} ${COPTFLAGS} ${CCPPFLAGS} " >> MINFO
	-@echo  "C Compiler version:"  >> MINFO ; ${C_CCV} >> MINFO 2>&1 ; true
	-@echo  "C++ Compiler version:"  >> MINFO; ${CXX_CCV} >> MINFO 2>&1 ; true
	-@echo  "Using Fortran compiler: ${FC} ${FOPTFLAGS} ${FCPPFLAGS}" >> MINFO
	-@echo  "Fortran Compiler version:" >> MINFO ; ${FCV} >> MINFO 2>&1 ; true
	-@echo  "-----------------------------------------\"; " >> MINFO
	-@echo  "static const char *petsccompilerflagsinfo = \"  " >> MINFO
	-@echo  "Using PETSc flags: ${PETSCFLAGS} ${PCONF}" >> MINFO
	-@echo  "-----------------------------------------" >> MINFO
	-@echo  "Using configuration flags:" >> MINFO
	-@echo  "-----------------------------------------" >> MINFO
	-@echo  "Using include paths: ${PETSC_INCLUDE}" >> MINFO
	-@echo  "------------------------------------------\"; " >> MINFO
	-@echo  "static const char *petsclinkerinfo = \"  " >> MINFO
	-@echo  "Using C linker: ${CLINKER}" >> MINFO
	-@echo  "Using Fortran linker: ${FLINKER}" >> MINFO
	-@echo  "Using libraries: ${PETSC_LIB} \"; " >> MINFO
	-@cat MINFO | ${SED} -e 's/\^M//g' | ${SED} -e 's/\\/\\\\/g' | ${SED} -e 's/$$/ \\n\\/' | sed -e 's/\;  \\n\\/\;/'> MINFO_
	-@cat MINFO_ | ${SED} -e 's/\//g'  > /dev/null; foobar=$$?; \
          if [ "$$foobar" = "0" ]; then \
	    cat MINFO_ | ${SED} -e 's/\//g' > ${MINFO}; \
          else cat MINFO | ${SED} -e 's/\^M//g' | ${SED} -e 's/\\/\\\\/g' | ${SED} -e 's/$$/ \\n\\/' | sed -e 's/\;  \\n\\/\;/'> ${MINFO}; \
          fi
	-@$(RM) MINFO MINFO_

#
# Builds the PETSc libraries
# This target also builds fortran77 and f90 interface
# files and compiles .F files
#
build:
	-@echo "BEGINNING TO COMPILE LIBRARIES IN ALL DIRECTORIES"
	-@echo "========================================="
	-@${OMAKE}  PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} ACTION=libfast tree
	@grep -i " error " make_log_${PETSC_ARCH} > /dev/null; if [ "$$?" = 0 ]; then \
           echo "Error during compile, check " make_log_${PETSC_ARCH}; \
           echo "Send it and configure.log to petsc-maint@mcs.anl.gov"; exit 1; fi
	-@${RANLIB} ${PETSC_LIB_DIR}/*.${AR_LIB_SUFFIX}
	-@echo "Completed building libraries"
	-@echo "========================================="
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
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH}  PETSC_DIR=${PETSC_DIR} ACTION=testexamples_1  tree 
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
            ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} ACTION=testexamples_3 tree; \
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
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} ACTION=testexamples_4  tree 
	-@echo "Completed compiling and running uniprocessor test examples"
	-@echo "========================================="
testfortran_uni: info 
	-@echo "BEGINNING TO COMPILE AND RUN TEST UNI-PROCESSOR FORTRAN EXAMPLES"
	-@echo "Due to different numerical round-off on certain"
	-@echo "machines some of the numbers may not match exactly."
	-@echo "========================================="
	-@if [ "${FC}" != "" ]; then \
            ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} ACTION=testexamples_9  tree; \
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
	-${RM} -f ${PETSC_LIB_DIR}/*

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
	-@if [ "${INSTALL_DIR}" == "${PETSC_DIR}" ]; then \
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
          cp -f bmake/adic* bmake/variables ${INSTALL_DIR}/bmake ; \
          cp -fr bmake/common ${INSTALL_DIR}/bmake;\
          cp -fr bmake/${PETSC_ARCH} ${INSTALL_DIR}/bmake;\
          cp -fr bin ${INSTALL_DIR};\
          if [ ! -d ${INSTALL_DIR}/lib ]; then \
	    ${MKDIR} ${INSTALL_DIR}/lib ; \
          fi;\
          for i in lib/lib*; do \
            if [ ! -d ${INSTALL_DIR}/$${i} ]; then \
              ${MKDIR} ${INSTALL_DIR}/$${i};\
            fi; \
            if [ -d $${i}/${PETSC_ARCH} ]; then \
              cp -fr $${i}/${PETSC_ARCH} ${INSTALL_DIR}/$${i};\
              ${RANLIB}  ${INSTALL_DIR}/$${i}/*.a > /dev/null 2>&1 ;\
              ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${INSTALL_DIR} shared; \
            fi;\
          done;\
          echo "sh/bash: PETSC_DIR="${INSTALL_DIR}"; export PETSC_DIR";\
          echo "csh/tcsh: setenv PETSC_DIR "${INSTALL_DIR} ;\
          echo "The do make test to verify correct install";\
        fi;

install_src:
	-@if [ "${INSTALL_DIR}" == "${PETSC_DIR}" ]; then \
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
	-@if [ "${INSTALL_DIR}" == "${PETSC_DIR}" ]; then \
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
	cd src/docs/tex/manual; ${OMAKE} build_manual.pdf LOC=${LOC}
	-${OMAKE} ACTION=manualpages tree_basic LOC=${LOC}
	-maint/wwwindex.py ${PETSC_DIR} ${LOC}
	-${OMAKE} ACTION=manexamples tree_basic LOC=${LOC}
	-${OMAKE} manconcepts LOC=${LOC}
	-${OMAKE} ACTION=getexlist tree_basic LOC=${LOC}
	-${OMAKE} ACTION=exampleconcepts tree_basic LOC=${LOC}
	-maint/helpindex.py ${PETSC_DIR} ${LOC}

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

allci: 
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH} ACTION=ci  alltree 

allco: 
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH} ACTION=co  alltree 

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
        start_configure configure_petsc configure_clean petscPython.tgz

