#
# This is the makefile for installing TAO. See the file
# docs/installation.html for directions on installing TAO.
# See also bmake/common for additional commands.
#
ALL: all


# Call make recursively in these directory
DIRS = src include docs tests

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
	-@echo "See docs/troubleshooting.html and docs/bugreporting.html"
	-@echo "for help with installation problems. Please send EVERYTHING"
	-@echo "printed out below when reporting problems."
	-@echo " "
	-@echo "To subscribe to the TAO users mailing list, send mail to "
	-@echo "majordomo@mcs.anl.gov with the message: "
	-@echo "subscribe tao-news"
	-@echo " "
	-@echo "=========================================="
	-@echo On `date` on `hostname`
	-@echo Machine characteristics: `uname -a`
	-@echo "-----------------------------------------"
	-@echo "Using PETSc directory: ${PETSC_DIR}"
	-@echo "Using PETSc arch: ${PETSC_ARCH}"
	-@echo "Using TAO directory: ${TAO_DIR}"
	-@echo "-----------------------------------------"
	-@grep "define PETSC_VERSION" ${PETSC_DIR}/include/petscversion.h | ${SED} "s/........//"
	-@grep TAO_VERSION_NUMBER include/tao_version.h | sed "s/........//"
	-@echo "-----------------------------------------"
	-@echo "Using include paths: ${TAO_INCLUDE}"
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
	-@echo "Using libraries: ${TAO_LIB}"
	-@echo "------------------------------------------"
	-@echo "Using mpiexec: ${MPIEXEC}"
	-@echo "=========================================="

MINFO = ${PETSC_DIR}/${PETSC_ARCH}/include/petscmachineinfo.h
tao_info_h:
	-@$(RM) -f ${MINFO} MINFO
	-@echo  "static const char *petscmachineinfo = \"\__n__\"" >> MINFO
	-@echo  "\"-----------------------------------------\__n__\"" >> MINFO
	-@if [ -f /usr/bin/cygcheck.exe ]; then \
	  echo  "\"Libraries compiled on `date` on `hostname|/usr/bin/dos2unix` \__n__\"" >> MINFO; \
          else \
	  echo  "\"Libraries compiled on `date` on `hostname` \__n__\"" >> MINFO; \
          fi
	-@echo  "\"Machine characteristics: `uname -a` \__n__\"" >> MINFO
	-@echo  "\"Using PETSc directory: ${PETSC_DIR}\__n__\"" >> MINFO
	-@echo  "\"Using PETSc arch: ${PETSC_ARCH}\__n__\"" >> MINFO
	-@echo  "\"-----------------------------------------\"; " >> MINFO
	-@echo  "static const char *petsccompilerinfo = \"\__n__\"" >> MINFO
	-@echo  "\"Using C compiler: ${PCC} ${PCC_FLAGS} ${COPTFLAGS} ${CFLAGS}\__n__\"" >> MINFO
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
# Builds the TAO libraries
# This target also builds fortran77 and f90 interface
# files and compiles .F files
#
tao_build:
	-@echo "BEGINNING TO COMPILE LIBRARIES IN ALL DIRECTORIES"
	-@echo "========================================="
	-@${OMAKE}  PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} TAO_DIR=${TAO_DIR} ACTION=libfast tree
	-@${RANLIB} ${TAO_LIB_DIR}/*.${AR_LIB_SUFFIX}
	-@echo "Completed building libraries"
	-@echo "========================================="

#

# Deletes TAO libraries
tao_deletelibs: 
	-${RM} -f ${PETSC_LIB_DIR}/libtao*.*


tao_shared: shared


tao_alletags:
	-@maint/generateetags.py


tao_testexamples_c: 
	-@PYTHONPATH=${TAO_DIR}/maint ./maint/runExamples.py

tao_allfortranstubs:
	-@maint/generatefortranstubs.py ${BFORT}

tao_manual:
	cd docs/tex/manual; ${OMAKE} manual.dvi manual.pdf manual.html

tao_deletemanpages:
	${RM} -f ${TAO_DIR}/docs/manualpages/*/*.html \
                 ${TAO_DIR}/docs/manualpages/manualpages.cit 

tao_allmanpages: tao_deletemanpages
	-${OMAKE} ACTION=tao_manpages_buildcite tree
	-${OMAKE} ACTION=tao_manpages tree
	-maint/wwwindex.py ${TAO_DIR}

tao_htmlpages: 
	-${OMAKE} ACTION=tao_html TAO_DIR=${TAO_DIR} PETSC_DIR=${PETSC_DIR} alltree LOC=${TAO_DIR}


tao_chk_lib_dir: chklib_dir
