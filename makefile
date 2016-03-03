#
# This is the makefile for compiling PETSc. See
# http://www.mcs.anl.gov/petsc/documentation/installation.html for directions on installing PETSc.
# See also conf for additional commands.
#
ALL: all
LOCDIR	 = ./
DIRS	 = src include tutorials interfaces
CFLAGS	 =
FFLAGS	 =
CPPFLAGS =
FPPFLAGS =

# next line defines PETSC_DIR and PETSC_ARCH if they are not set
include ././${PETSC_ARCH}/lib/petsc/conf/petscvariables
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

#
# Basic targets to build PETSc libraries.
# all: builds the c, fortran, and f90 libraries
all: chk_makej
	@${OMAKE}  PETSC_ARCH=${PETSC_ARCH}  PETSC_DIR=${PETSC_DIR} chk_petscdir chk_upgrade | tee ${PETSC_ARCH}/lib/petsc/conf/make.log
	@ln -sf ${PETSC_ARCH}/lib/petsc/conf/make.log make.log
	@if [ "${MAKE_IS_GNUMAKE}" != "" ]; then \
	   ${OMAKE_PRINTDIR} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} all-gnumake-local 2>&1 | tee -a ${PETSC_ARCH}/lib/petsc/conf/make.log; \
	elif [ "${PETSC_BUILD_USING_CMAKE}" != "" ]; then \
	   ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} all-cmake-local 2>&1 | tee -a ${PETSC_ARCH}/lib/petsc/conf/make.log \
		| egrep -v '( --check-build-system |cmake -E | -o CMakeFiles/petsc[[:lower:]]*.dir/| -o lib/libpetsc|CMakeFiles/petsc[[:lower:]]*\.dir/(build|depend|requires)|-f CMakeFiles/Makefile2|Dependee .* is newer than depender |provides\.build. is up to date)'; \
	 else \
	   ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} all-legacy-local 2>&1 | tee -a ${PETSC_ARCH}/lib/petsc/conf/make.log \
                | ${GREP} -v "has no symbols"; \
	 fi
	@egrep -i "( error | error: |no such file or directory)" ${PETSC_ARCH}/lib/petsc/conf/make.log | tee ${PETSC_ARCH}/lib/petsc/conf/error.log > /dev/null
	@if test -s ${PETSC_ARCH}/lib/petsc/conf/error.log; then \
           printf ${PETSC_TEXT_HILIGHT}"**************************ERROR*************************************\n" 2>&1 | tee -a ${PETSC_ARCH}/lib/petsc/conf/make.log; \
           echo "  Error during compile, check ${PETSC_ARCH}/lib/petsc/conf/make.log" 2>&1 | tee -a ${PETSC_ARCH}/lib/petsc/conf/make.log; \
           echo "  Send it and ${PETSC_ARCH}/lib/petsc/conf/configure.log to petsc-maint@mcs.anl.gov" 2>&1 | tee -a ${PETSC_ARCH}/lib/petsc/conf/make.log;\
           printf "********************************************************************"${PETSC_TEXT_NORMAL}"\n" 2>&1 | tee -a ${PETSC_ARCH}/lib/petsc/conf/make.log;\
	 else \
	  ${OMAKE} shared_install PETSC_ARCH=${PETSC_ARCH}  PETSC_DIR=${PETSC_DIR} 2>&1 | tee -a ${PETSC_ARCH}/lib/petsc/conf/make.log ;\
        fi #solaris make likes to print the whole command that gave error. So split this up into the smallest chunk below
	@echo "Finishing at: `date`" >> ${PETSC_ARCH}/lib/petsc/conf/make.log
	@if test -s ${PETSC_ARCH}/lib/petsc/conf/error.log; then exit 1; fi

all-gnumake:
	@if [ "${MAKE_IS_GNUMAKE}" != "" ]; then \
          ${OMAKE_PRINTDIR}  PETSC_ARCH=${PETSC_ARCH}  PETSC_DIR=${PETSC_DIR} PETSC_BUILD_USING_CMAKE="" all;\
        else printf ${PETSC_TEXT_HILIGHT}"Build not configured for GNUMAKE. Quiting"${PETSC_TEXT_NORMAL}"\n"; exit 1; fi

all-cmake:
	@if [ "${PETSC_BUILD_USING_CMAKE}" != "" ]; then \
          ${OMAKE}  PETSC_ARCH=${PETSC_ARCH}  PETSC_DIR=${PETSC_DIR} MAKE_IS_GNUMAKE="" all;\
        else printf ${PETSC_TEXT_HILIGHT}"Build not configured for CMAKE. Quiting"${PETSC_TEXT_NORMAL}"\n"; exit 1; fi

all-legacy:
	@${OMAKE}  PETSC_ARCH=${PETSC_ARCH}  PETSC_DIR=${PETSC_DIR} PETSC_BUILD_USING_CMAKE="" MAKE_IS_GNUMAKE="" all

all-gnumake-local: chk_makej info gnumake matlabbin mpi4py-build petsc4py-build

all-cmake-local: chk_makej info cmakegen cmake matlabbin mpi4py-build petsc4py-build

all-legacy-local: chk_makej chklib_dir info deletelibs deletemods build matlabbin shared_nomesg mpi4py-build petsc4py-build
#
# Prints information about the system and version of PETSc being compiled
#
info: chk_makej
	-@echo "=========================================="
	-@echo " "
	-@echo "See documentation/faq.html and documentation/bugreporting.html"
	-@echo "for help with installation problems.  Please send EVERYTHING"
	-@echo "printed out below when reporting problems.  Please check the"
	-@echo "mailing list archives and consider subscribing."
	-@echo " "
	-@echo "  http://www.mcs.anl.gov/petsc/miscellaneous/mailing-lists.html"
	-@echo " "
	-@echo "=========================================="
	-@echo Starting on `hostname` at `date`
	-@echo Machine characteristics: `uname -a`
	-@echo "-----------------------------------------"
	-@echo "Using PETSc directory: ${PETSC_DIR}"
	-@echo "Using PETSc arch: ${PETSC_ARCH}"
	-@echo "-----------------------------------------"
	-@grep "define PETSC_VERSION" ${PETSC_DIR}/include/petscversion.h | ${SED} "s/........//"
	-@echo "-----------------------------------------"
	-@echo "Using configure Options: ${CONFIGURE_OPTIONS}"
	-@echo "Using configuration flags:"
	-@grep "\#define " ${PETSC_DIR}/${PETSC_ARCH}/include/petscconf.h
	-@echo "-----------------------------------------"
	-@echo "Using C/C++ compile: ${PETSC_COMPILE}"
	-@if [ "${PETSC_LANGUAGE}" = "CONLY" -a "${MPICC_SHOW}" != "" ]; then \
             printf  "mpicc -show: %b\n" "${MPICC_SHOW}"; \
	  elif [ "${PETSC_LANGUAGE}" = "CXXONLY" -a "${MPICXX_SHOW}" != "" ]; then \
             printf "mpicxx -show: %b\n" "${MPICXX_SHOW}"; \
          fi;
	-@if [ "${FC}" != "" ]; then \
	   echo "Using Fortran compile: ${PETSC_FCOMPILE}";\
           if [ "${MPIFC_SHOW}" != "" ]; then \
             printf "mpif90 -show: %b\n" "${MPIFC_SHOW}"; \
           fi; \
         fi
	-@if [ "${CUDAC}" != "" ]; then \
	   echo "Using CUDA compile: ${PETSC_CUCOMPILE}";\
         fi
	-@echo "-----------------------------------------"
	-@echo "Using C/C++ linker: ${PCC_LINKER}"
	-@echo "Using C/C++ flags: ${PCC_LINKER_FLAGS}"
	-@if [ "${FC}" != "" ]; then \
	   echo "Using Fortran linker: ${FC_LINKER}";\
	   echo "Using Fortran flags: ${FC_LINKER_FLAGS}";\
         fi
	-@echo "-----------------------------------------"
	-@echo "Using libraries: ${PETSC_LIB}"
	-@echo "------------------------------------------"
	-@echo "Using mpiexec: ${MPIEXEC}"
	-@echo "=========================================="

#
# Builds the PETSc libraries
# This target also builds fortran77 and f90 interface
# files and compiles .F files
#
build: chk_makej
	-@echo "BEGINNING TO COMPILE LIBRARIES IN ALL DIRECTORIES"
	-@echo "========================================="
	-@${OMAKE}  PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} ACTION=libfast tree
	-@${RANLIB} ${PETSC_LIB_DIR}/*.${AR_LIB_SUFFIX}  > tmpf 2>&1 ; ${GREP} -v "has no symbols" tmpf; ${RM} tmpf;
	-@echo "Completed building libraries"
	-@echo "========================================="
#
# Build MatLab binaries
#
matlabbin:
	-@if [ "${MATLAB_MEX}" != "" -a "${PETSC_SCALAR}" = "real" -a "${PETSC_PRECISION}" = "double" ]; then \
          echo "BEGINNING TO COMPILE MATLAB INTERFACE"; \
            if [ ! -d "${PETSC_DIR}/${PETSC_ARCH}/lib/petsc" ] ; then ${MKDIR}  ${PETSC_DIR}/${PETSC_ARCH}/lib/petsc; fi; \
            if [ ! -d "${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/matlab" ] ; then ${MKDIR}  ${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/matlab; fi; \
            cd src/sys/classes/viewer/impls/socket/matlab && ${OMAKE} matlabcodes PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR}; \
            echo "========================================="; \
        fi
#
# Builds PETSc test examples for a given architecture
#
check: test
test:
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH}  PETSC_DIR=${PETSC_DIR} test_build 2>&1 | tee ./${PETSC_ARCH}/lib/petsc/conf/test.log
	-@if [ "${PETSC_WITH_BATCH}" = "" ]; then \
          printf "=========================================\n"; \
          printf "Now to evaluate the computer systems you plan use - do:\n"; \
          printf "make PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH} streams\n"; \
        fi
testx:
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH}  PETSC_DIR=${PETSC_DIR} testx_build 2>&1 | tee ./${PETSC_ARCH}/lib/petsc/conf/testx.log
test_build:
	-@echo "Running test examples to verify correct installation"
	-@echo "Using PETSC_DIR=${PETSC_DIR} and PETSC_ARCH=${PETSC_ARCH}"
	@cd src/snes/examples/tutorials >/dev/null; ${OMAKE} PETSC_ARCH=${PETSC_ARCH}  PETSC_DIR=${PETSC_DIR} clean
	@cd src/snes/examples/tutorials >/dev/null; ${OMAKE} PETSC_ARCH=${PETSC_ARCH}  PETSC_DIR=${PETSC_DIR} testex19
	@if [ "${FC}" != "" ]; then \
          egrep "^#define PETSC_USE_FORTRAN_DATATYPES 1" ${PETSC_DIR}/${PETSC_ARCH}/include/petscconf.h | tee .ftn-dtype.log > /dev/null; \
          if test -s .ftn-dtype.log; then F90TEST="testex5f90t"; else F90TEST="testex5f"; fi; ${RM} .ftn-dtype.log; \
          cd src/snes/examples/tutorials >/dev/null; ${OMAKE} PETSC_ARCH=${PETSC_ARCH}  PETSC_DIR=${PETSC_DIR} $${F90TEST}; \
         fi;
	@cd src/snes/examples/tutorials >/dev/null; ${OMAKE} PETSC_ARCH=${PETSC_ARCH}  PETSC_DIR=${PETSC_DIR} clean
	-@echo "Completed test examples"
testx_build:
	-@echo "Running graphics test example to verify correct X11 installation"
	-@echo "Using PETSC_DIR=${PETSC_DIR} and PETSC_ARCH=${PETSC_ARCH}"
	@cd src/snes/examples/tutorials; ${OMAKE} PETSC_ARCH=${PETSC_ARCH}  PETSC_DIR=${PETSC_DIR} clean
	@cd src/snes/examples/tutorials; ${OMAKE} PETSC_ARCH=${PETSC_ARCH}  PETSC_DIR=${PETSC_DIR} testxex19
	@cd src/snes/examples/tutorials; ${OMAKE} PETSC_ARCH=${PETSC_ARCH}  PETSC_DIR=${PETSC_DIR} clean
	-@echo "Completed graphics test example"

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
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} ACTION=testexamples_C_X_MPIUni  tree
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
deletelibs: chk_makej
	-${RM} -rf ${PETSC_LIB_DIR}/libpetsc*.*
deletemods: chk_makej
	-${RM} -f ${PETSC_DIR}/${PETSC_ARCH}/include/petsc*.mod

# Cleans up build
allclean-legacy: deletelibs deletemods
	-@${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} ACTION=clean-legacy tree
allclean-cmake:
	-@cd ${PETSC_ARCH} && ${OMAKE} clean
allclean-gnumake:
	-@${OMAKE} -f gmakefile clean

allclean:
	@if [ "${MAKE_IS_GNUMAKE}" != "" ]; then \
	   ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} allclean-gnumake; \
	elif [ "${PETSC_BUILD_USING_CMAKE}" != "" ]; then \
	   ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} allclean-cmake; \
	else \
	   ${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} allclean-legacy; \
	fi

clean:: allclean

distclean: chk_petscdir
	@if [ -f ${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/conf/reconfigure-${PETSC_ARCH}.py ]; then \
	  echo "*** Preserving ${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/conf/reconfigure-${PETSC_ARCH}.py in ${PETSC_DIR} ***"; \
          mv -f ${PETSC_DIR}/${PETSC_ARCH}/lib/petsc/conf/reconfigure-${PETSC_ARCH}.py ${PETSC_DIR}/; fi
	@echo "*** Deleting all build files in ${PETSC_DIR}/${PETSC_ARCH} ***"
	-${RM} -rf ${PETSC_DIR}/${PETSC_ARCH}/


#
reconfigure:
	@${PYTHON} ${PETSC_ARCH}/lib/petsc/conf/reconfigure-${PETSC_ARCH}.py
#
install:
	@${PYTHON} ./config/install.py -destDir=${DESTDIR}
	${OMAKE} PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} mpi4py-install petsc4py-install

newall:
	-@cd src/sys;  @${PYTHON} ${PETSC_DIR}/config/builder.py
	-@cd src/vec;  @${PYTHON} ${PETSC_DIR}/config/builder.py
	-@cd src/mat;  @${PYTHON} ${PETSC_DIR}/config/builder.py
	-@cd src/dm;   @${PYTHON} ${PETSC_DIR}/config/builder.py
	-@cd src/ksp;  @${PYTHON} ${PETSC_DIR}/config/builder.py
	-@cd src/snes; @${PYTHON} ${PETSC_DIR}/config/builder.py
	-@cd src/ts;   @${PYTHON} ${PETSC_DIR}/config/builder.py

streams:
	cd src/benchmarks/streams; ${OMAKE} PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH} streams

stream:
	cd src/benchmarks/streams; ${OMAKE} PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH} stream
# ------------------------------------------------------------------
#
# All remaining actions are intended for PETSc developers only.
# PETSc users should not generally need to use these commands.
#
#  See the users manual for how the tags files may be used from Emacs and Vi/Vim
#
alletags:
	-@${PYTHON} bin/maint/generateetags.py
	-@find config -type f -name "*.py" |grep -v SCCS | xargs etags -o TAGS_PYTHON

# obtain gtags from http://www.gnu.org/s/global/
allgtags:
	-@find ${PETSC_DIR}/include ${PETSC_DIR}/src ${PETSC_DIR}/bin -regex '\(.*makefile\|.*\.\(cc\|hh\|cpp\|C\|hpp\|c\|h\|cu\|m\)$$\)' | grep -v ftn-auto  | gtags -f -

allfortranstubs:
	-@${RM} -rf include/petsc/finclude/ftn-auto/*-tmpdir
	-@${PYTHON} bin/maint/generatefortranstubs.py ${BFORT}  ${VERBOSE}
	-@${PYTHON} bin/maint/generatefortranstubs.py -merge  ${VERBOSE}
	-@${RM} -rf include/petsc/finclude/ftn-auto/*-tmpdir
deletefortranstubs:
	-@find . -type d -name ftn-auto | xargs rm -rf
cmakegen:
	-@${PYTHON} config/cmakegen.py
#
# These are here for the target allci and allco, and etags
#

BMAKEFILES = conf/variables conf/rules conf/test 
SCRIPTS    = bin/maint/builddist  bin/maint/wwwman bin/maint/xclude bin/maint/bugReport.py bin/maint/buildconfigtest bin/maint/builddistlite \
             bin/maint/buildtest bin/maint/checkBuilds.py bin/maint/copylognightly bin/maint/copylognightly.tao bin/maint/countfiles bin/maint/findbadfiles \
             bin/maint/fixinclude bin/maint/getexlist bin/maint/getpdflabels.py bin/maint/helpindex.py bin/maint/hosts.local bin/maint/hosts.solaris  \
             bin/maint/lex.py  bin/maint/mapnameslatex.py bin/maint/startnightly bin/maint/startnightly.tao bin/maint/submitPatch.py \
             bin/maint/update-docs.py  bin/maint/wwwindex.py bin/maint/xcludebackup bin/maint/xcludecblas bin/maint/zap bin/maint/zapall \
             config/PETSc/Configure.py config/PETSc/Options.py \
             config/PETSc/utilities/*.py


# Builds all the documentation - should be done every night
alldoc: alldoc1 alldoc2 alldoc3 docsetdate

# Build everything that goes into 'doc' dir except html sources
alldoc1: chk_loc deletemanualpages chk_concepts_dir
	-${PYTHON} bin/maint/countpetsccits.py
	-${OMAKE} ACTION=manualpages_buildcite tree_basic LOC=${LOC}
	-@sed -e s%man+../%man+manualpages/% ${LOC}/docs/manualpages/manualpages.cit > ${LOC}/docs/manualpages/htmlmap
	-@cat ${PETSC_DIR}/src/docs/mpi.www.index >> ${LOC}/docs/manualpages/htmlmap
	-cd src/docs/tex/manual; ${OMAKE} manual.pdf LOC=${LOC}
	-cd src/docs/tex/manual; ${OMAKE} developers.pdf LOC=${LOC}
	-cd src/docs/tao_tex/manual; ${OMAKE} manual.pdf
	-${OMAKE} ACTION=manualpages tree_basic LOC=${LOC}
	-${PYTHON} bin/maint/wwwindex.py ${PETSC_DIR} ${LOC}
	-${OMAKE} ACTION=manexamples tree_basic LOC=${LOC}
	-${OMAKE} manconcepts LOC=${LOC}
	-${OMAKE} ACTION=getexlist tree_basic LOC=${LOC}
	-${OMAKE} ACTION=exampleconcepts tree_basic LOC=${LOC}
	-${PYTHON} bin/maint/helpindex.py ${PETSC_DIR} ${LOC}

# Builds .html versions of the source
# html overwrites some stuff created by update-docs - hence this is done later.
alldoc2: chk_loc
	-${OMAKE} ACTION=html PETSC_DIR=${PETSC_DIR} alltree LOC=${LOC}
	-${PYTHON} bin/maint/update-docs.py ${PETSC_DIR} ${LOC}
#
# Builds HTML versions of Matlab scripts
alldoc3: chk_loc
	if  [ "${MATLAB_COMMAND}" != "" ]; then\
          export MATLABPATH=${MATLABPATH}:${PETSC_DIR}/share/petsc/matlab; \
          cd ${PETSC_DIR}/share/petsc/matlab; ${MATLAB_COMMAND} -nodisplay -nodesktop -r "generatehtml;exit" ; \
        fi

#
# Makes links for all manual pages in $LOC/docs/manualpages/all
allman:
	@cd ${LOC}/docs/manualpages; rm -rf all ; mkdir all ; find *  -type d -wholename all -prune -o -name index.html -prune  -o -type f -name \*.html -exec ln -s  -f ../{} all \;

DOCSETDATE_PRUNE_LIST="-o -type f -wholename share/petsc/saws/linearsolveroptions.html -prune -o -type f -wholename tutorials/HandsOnExercise.html -prune -o -type f -wholename tutorials/TAOHandsOnExercise.html -prune"

# modify all generated html files and add in version number, date, canonical URL info.
docsetdate: chk_petscdir
	@echo "Updating generated html files with petsc version, date, canonical URL info";\
        version_release=`grep '^#define PETSC_VERSION_RELEASE ' include/petscversion.h |tr -s ' ' | cut -d ' ' -f 3`; \
        version_major=`grep '^#define PETSC_VERSION_MAJOR ' include/petscversion.h |tr -s ' ' | cut -d ' ' -f 3`; \
        version_minor=`grep '^#define PETSC_VERSION_MINOR ' include/petscversion.h |tr -s ' ' | cut -d ' ' -f 3`; \
        version_subminor=`grep '^#define PETSC_VERSION_SUBMINOR ' include/petscversion.h |tr -s ' ' | cut -d ' ' -f 3`; \
        if  [ $${version_release} = 0 ]; then \
          petscversion=petsc-master; \
          export petscversion; \
        elif [ $${version_release} = 1 ]; then \
          petscversion=petsc-$${version_major}.$${version_minor}.$${version_subminor}; \
          export petscversion; \
        else \
          echo "Unknown PETSC_VERSION_RELEASE: $${version_release}"; \
          exit; \
        fi; \
        datestr=`git log -1 --pretty=format:%ci | cut -d ' ' -f 1`; \
        export datestr; \
        gitver=`git describe`; \
        export gitver; \
        find * -type d -wholename src/docs/website -prune -o -type d -wholename src/benchmarks/results -prune -o \
          -type d -wholename config/BuildSystem/docs/website -prune -o -type d -wholename include/web -prune -o \
          -type d -wholename 'arch-*' -prune -o -type d -wholename src/tops -prune -o -type d -wholename externalpackages -prune ${DOCSETDATE_PRUNE_LIST} -o \
          -type f -name \*.html \
          -exec perl -pi -e 's^(<body.*>)^$$1\n   <div id=\"version\" align=right><b>$$ENV{petscversion} $$ENV{datestr}</b></div>\n   <div id="bugreport" align=right><a href="mailto:petsc-maint\@mcs.anl.gov?subject=Typo or Error in Documentation &body=Please describe the typo or error in the documentation: $$ENV{petscversion} $$ENV{gitver} {} "><small>Report Typos and Errors</small></a></div>^i' {} \; \
          -exec perl -pi -e 's^(<head>)^$$1 <link rel="canonical" href="http://www.mcs.anl.gov/petsc/petsc-current/{}" />^i' {} \; ; \
        echo "Done fixing version number, date, canonical URL info"

alldocclean: deletemanualpages allcleanhtml

# Deletes man pages (HTML version)
deletemanualpages: chk_loc
	-@if [ -d ${LOC} -a -d ${LOC}/docs/manualpages ]; then \
          find ${LOC}/docs/manualpages -type f -name "*.html" -exec ${RM} {} \; ;\
          ${RM} ${LOC}/docs/exampleconcepts ;\
          ${RM} ${LOC}/docs/manconcepts ;\
          ${RM} ${LOC}/docs/manualpages/manualpages.cit ;\
          ${PYTHON} bin/maint/update-docs.py ${PETSC_DIR} ${LOC} clean;\
        fi

allcleanhtml:
	-${OMAKE} ACTION=cleanhtml PETSC_DIR=${PETSC_DIR} alltree

chk_concepts_dir: chk_loc
	@if [ ! -d "${LOC}/docs/manualpages/concepts" ]; then \
	  echo Making directory ${LOC}/docs/manualpages/concepts for library; ${MKDIR} ${LOC}/docs/manualpages/concepts; fi

###########################################################
# targets to build distribution and update docs
###########################################################

# Creates ${HOME}/petsc.tar.gz [and petsc-lite.tar.gz]
dist:
	${PETSC_DIR}/bin/maint/builddist ${PETSC_DIR} master

# This target works only if you can do 'ssh petsc@login.mcs.anl.gov'
# also copy the file over to ftp site.
web-snapshot:
	@if [ ! -f "${HOME}/petsc-master.tar.gz" ]; then \
	    echo "~/petsc-master.tar.gz missing! cannot update petsc-master snapshot on mcs-web-site"; \
	  else \
            echo "updating petsc-master snapshot on mcs-web-site"; \
	    tmpdir=`mktemp -d -t petsc-doc.XXXXXXXX`; \
	    cd $${tmpdir}; tar -xzf ${HOME}/petsc-master.tar.gz; \
	    /usr/bin/rsync  -e ssh -az --delete $${tmpdir}/petsc-master/ \
              petsc@login.mcs.anl.gov:/mcs/web/research/projects/petsc/petsc-master ;\
	    /bin/cp -f /home/petsc/petsc-master.tar.gz /mcs/ftp/pub/petsc/petsc-master.tar.gz;\
	    ${RM} -rf $${tmpdir} ;\
	  fi

# build the tarfile - and then update petsc-master snapshot on mcs-web-site
update-web-snapshot: dist web-snapshot

# This target updates website main pages
update-web:
	@cd ${PETSC_DIR}/src/docs; make PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH} bib2html; \
	/usr/bin/rsync -az -C --exclude=documentation/index.html \
          --exclude=documentation/installation.html --exclude=download/index.html \
	  ${PETSC_DIR}/src/docs/website/ petsc@login.mcs.anl.gov:/mcs/web/research/projects/petsc
	@cd ${PETSC_DIR}/docs; /usr/bin/rsync -az developers.pdf petsc@login.mcs.anl.gov:/mcs/web/research/projects/petsc/developers/
	@cd ${PETSC_DIR}/src/docs/tex; /usr/bin/rsync -az petscapp.bib petsc.bib petsc@login.mcs.anl.gov:/mcs/web/research/projects/petsc/publications

#
#  builds a single list of files for each PETSc library so they may all be built in parallel
#  without a recursive set of make calls
createfastbuild:
	cd src/vec; ${RM} -f files; /bin/echo -n "SOURCEC = " > files; make tree ACTION=sourcelist BASE_DIR=${PETSC_DIR}/src/vec

###########################################################
#
#  See script for details
#
gcov:
	-@${PETSC_DIR}/bin/maint/gcov.py -run_gcov

mergegcov:
	-@${PETSC_DIR}/bin/maint/gcov.py -merge_gcov ${LOC} *.tar.gz

########################
#
# Create the include dependency graph (requires graphviz to be available)
#
includegraph:
	-@${PETSC_DIR}/src/contrib/style/include-graph.sh includegraph.pdf
	-@echo Include dependency graph written to includegraph.pdf

#
# -------------------------------------------------------------------------------
#
# Some macros to check if the fortran interface is up-to-date.
#
countfortranfunctions:
	-@cd ${PETSC_DIR}/src/fortran; egrep '^void' custom/*.c auto/*.c | \
	cut -d'(' -f1 | tr -s  ' ' | cut -d' ' -f2 | uniq | egrep -v "(^$$|Petsc)" | \
	sed "s/_$$//" | sort > /tmp/countfortranfunctions

countcfunctions:
	-@grep PETSC_EXTERN ${PETSC_DIR}/include/*.h  | grep "(" | tr -s ' ' | \
	cut -d'(' -f1 | cut -d' ' -f3 | grep -v "\*" | tr -s '\012' |  \
	tr 'A-Z' 'a-z' |  sort | uniq > /tmp/countcfunctions

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

checkpackagetests:
	-@echo "Missing package tests"
	-@cat config/examples/*.py > configexamples; pushd config/BuildSystem/config/packages/; packages=`ls *.py | sed "s/\\.py//g"`;popd; for i in $${packages}; do j=`echo $${i} | tr '[:upper:]' '[:lower:]'`; printf $${j} ; egrep "(with-$${j}|download-$${j})" configexamples | grep -v "=0" | wc -l ; done
	-@echo "Missing download package tests"
	-@cat config/examples/*.py > configexamples; pushd config/BuildSystem/config/packages/; packages=`grep -l "download " *.py  | sed "s/\\.py//g"`;popd; for i in $${packages}; do j=`echo $${i} | tr '[:upper:]' '[:lower:]'`; printf $${j} ; egrep "(download-$${j})" configexamples | grep -v "=0" | wc -l ; done

#
# Automatically generates PETSc exercises in html from the tutorial examples.
#
# The introduction for each section is obtained from docs/manualpages/header_${MANSEC} is under RCS and may be edited
#  (used also in introductions to the manual pages)
# The overall introduction is in docs/exercises/introduction.html and is under RCS and may be edited
# The list of exercises is from TUTORIALS in each directory's makefile
#
# DO NOT EDIT the pageform.txt or *.htm files generated since they will be automatically replaced.
# The pagemaker rule is in the file conf (at the bottom)
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

.PHONY: info info_h all build testexamples testfortran testexamples_uni testfortran_uni ranlib deletelibs allclean update \
        alletags etags etags_complete etags_noexamples etags_makefiles etags_examples etags_fexamples alldoc allmanualpages \
        allhtml allcleanhtml  allci allco allrcslabel countfortranfunctions \
        start_configure configure_petsc configure_clean matlabbin

petscao : petscmat petscao.f90.h
petscdm : petscksp petscdm.f90.h
petscdraw : petsc petscdraw.f90.h
petscis : petsc petscis.f90.h
petscksp : petscpc  petscksp.f90.h
petsclog : petsc petsclog.f90.h
petscmat : petscvec petscmat.f90.h
petscmg : petscksp petscmg.f90.h
petscpc : petscmat petscpc.f90.h
petscsnes : petscksp petscsnes.f90.h
petscsys : petsc petscsys.f90.h
petscts : petscsnes petscts.f90.h
petsc : petsc.f90.h
petscvec : petscis petscvec.f90.h
petscviewer : petsc petscviewer.f90.h
petscmesh : petsc petscmesh.f90.h
modules : petscao petscdm petscdraw petscis petscksp petsclog petscmat petscmg petscpc petscsnes petscsys petscts petsc petscvec petscviewer petscmesh
