
#
# This is the makefile for installing TAO. See the file
# docs/installation.html for directions on installing TAO.
# See also bmake/common for additional commands.
#
ALL: all

# Call make recursively in these directory
DIRS = src include docs tests

include ${PETSC_DIR}/${PETSC_ARCH}/conf/petscvariables
include ${TAO_DIR}/conf/tao_base

#
# Basic targets to build TAO libraries.
# all     : builds the C/C++ and Fortran libraries
all       : tao_info tao_chk_tao_dir tao_chk_lib_dir tao_deletelibs tao_build tao_shared 
#
# Prints information about the system and version of TAO being compiled
#
tao_info:
	-@echo "=========================================="
	-@echo " "
	-@echo "Please send EVERYTHING printed out below when"
	-@echo "reporting problems to tao-comments@mcs.anl.gov"
	-@echo " "
	-@echo "To subscribe to the TAO users mailing list, please "
	-@echo "visit https://lists.mcs.anl.gov/mailman/listinfo/tao-news"
	-@echo " "
	-@echo "=========================================="
	-@echo On `date` on `hostname`
	-@echo Machine characteristics: `uname -a`
	-@echo "-----------------------------------------"
	-@echo "Using PETSc directory: ${PETSC_DIR}"
	-@echo "Using PETSc arch: ${PETSC_ARCH}"
	-@echo "Using TAO directory: ${TAO_DIR}"
	-@echo "-----------------------------------------"
	-@grep -e "TAO_VERSION_MAJOR" ${TAO_DIR}/include/tao_version.h | grep -v "&&" | sed "s/........//"
	-@grep -e "TAO_VERSION_MINOR" ${TAO_DIR}/include/tao_version.h | grep -v "&&" | sed "s/........//"
	-@grep -e "TAO_VERSION_PATCH" ${TAO_DIR}/include/tao_version.h | grep -v "&&" | sed "s/........//"
	-@grep "define PETSC_VERSION" ${PETSC_DIR}/include/petscversion.h | ${SED} "s/........//" 
	-@echo "-----------------------------------------"
	-@echo "Using PETSc configure Options: ${CONFIGURE_OPTIONS}"
	-@echo "------------------------------------------"
	-@echo "Using C/C++ include paths: ${TAO_INCLUDE}"
	-@echo "Using C/C++ compiler: ${PCC} ${PCC_FLAGS} ${COPTFLAGS} ${CFLAGS}"
	-@if [ "${FC}" != "" ]; then \
	   echo "Using Fortran include/module paths: ${PETSC_FC_INCLUDES}";\
	   echo "Using Fortran compiler: ${FC} ${FC_FLAGS} ${FFLAGS} ${FPP_FLAGS}";\
         fi
	-@echo "-----------------------------------------"
	-@echo "Using C/C++ linker: ${PCC_LINKER}"
	-@echo "Using C/C++ flags: ${PCC_LINKER_FLAGS}"
	-@if [ "${FC}" != "" ]; then \
	   echo "Using Fortran linker: ${FC_LINKER}";\
	   echo "Using Fortran flags: ${FC_LINKER_FLAGS}";\
         fi
	-@echo "-----------------------------------------"
	-@echo "Using PETSc libraries: ${PETSC_LIB}"
	-@echo "Using TAO libraries: ${TAO_LIB}"
	-@echo "------------------------------------------"
	-@echo "Using mpiexec: ${MPIEXEC}"
	-@echo "=========================================="



#
# Builds the TAO libraries
# This target also builds fortran77 and f90 interface
# files and compiles .F files
#
tao_build:
	@echo "BEGINNING TO COMPILE LIBRARIES IN ALL DIRECTORIES"
	@echo "========================================="
	@${OMAKE}  PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} TAO_DIR=${TAO_DIR} ACTION=libfast tree
	@${RANLIB} ${TAO_LIB_DIR}/*.${AR_LIB_SUFFIX}
	@echo "Completed building libraries"
	@echo "========================================="

#

# Deletes TAO libraries
tao_deletelibs: 
	-${RM} -f ${TAO_LIB_DIR}/libtao*.*


tao_shared: shared

install: tao_chk_petsc_install 
	@${PYTHON} ./conf/install.py -destDir=${DESTDIR}

tao_alletags:
	-@maint/generateetags.py


tao_testexamples: 
	-@PYTHONPATH=${TAO_DIR}/maint ./maint/runTests.py -d -e c

tao_testexamples_uni:
	-@PYTHONPATH=${TAO_DIR}/maint ./maint/runTests.py -d -e c single

tao_testfortran: 
	-@PYTHONPATH=${TAO_DIR}/maint ./maint/runTests.py -d -e fortran

tao_testfortran_uni:
	-@PYTHONPATH=${TAO_DIR}/maint ./maint/runTests.py -d -e fortran single


tao_allfortranstubs:
	-@maint/generatefortranstubs.py ${BFORT}

tao_manual:
	cd docs/tex/manual; ${OMAKE} manual.pdf

tao_deletemanpages:
	${RM} -f ${TAO_DIR}/docs/manpages/*/*.html \
                 ${TAO_DIR}/docs/manpages/manpages.cit 

tao_allmanpages: tao_htmlpages tao_deletemanpages tao_pages tao_docsetdate

tao_pages:
	@mkdir -p ${TAO_DIR}/docs/manpages/taosolver
	@mkdir -p ${TAO_DIR}/docs/manpages/taolinesearch

	-${OMAKE} ACTION=tao_manpages_buildcite tree
	-${OMAKE} ACTION=tao_manpages tree
	-${OMAKE} ACTION=tao_manexamples tree LOC=${TAO_DIR}
	-maint/wwwindex.py ${TAO_DIR}


tao_htmlpages: 
	-${OMAKE} ACTION=tao_html TAO_DIR=${TAO_DIR} PETSC_DIR=${PETSC_DIR} alltree LOC=${TAO_DIR}


tao_docsetdate: 
	@echo "Updating generated html files with TAO version, date info";\
        version_release=`grep '^#define TAO_VERSION_RELEASE ' include/tao_version.h |tr -s ' ' | cut -d ' ' -f 3`; \
        version_major=`grep '^#define TAO_VERSION_MAJOR ' include/tao_version.h |tr -s ' ' | cut -d ' ' -f 3`; \
        version_minor=`grep '^#define TAO_VERSION_MINOR ' include/tao_version.h |tr -s ' ' | cut -d ' ' -f 3`; \
        version_patch=`grep '^#define TAO_VERSION_PATCH ' include/tao_version.h |tr -s ' ' | cut -d ' ' -f 3`; \
        if  [ $${version_release} = 0 ]; then \
          taoversion=tao-devel; \
          export taoversion; \
        elif [ $${version_release} = 1 ]; then \
          taoversion=tao-$${version_major}.$${version_minor}-p$${version_patch}; \
          export taoversion; \
        else \
          echo "Unknown TAO_VERSION_RELEASE: $${version_release}"; \
          exit; \
        fi; \
	datestr=`hg tip --template "{date|shortdate}"`; \
	export datestr; \
	find * -type d -wholename src/docs/website -prune -o \
	  -type f -name \*.html \
	  -exec perl -pi -e 's^(<body.*>)^$$1\n <div id=\"version\" align=right><b>$$ENV{taoversion} $$ENV{datestr}</b></div>^i' {} \; \
	  -exec perl -pi -e 's^(<head>)^$$1 <link rel="canonical" href="http://www.mcs.anl.gov/tao/docs/{}" />^i' {} \; ; \
	echo "Done fixing version number, date, canonical URL info"

tao_chk_lib_dir: chklib_dir
