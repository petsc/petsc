#
# This is the makefile for installing TAO. See the file
# docs/installation.html for directions on installing TAO.
# See also bmake/common for additional commands.
#
ALL: all


# Call make recursively in these directory
DIRS = src include docs 

include ${TAO_DIR}/bmake/packages
include ${TAO_DIR}/bmake/tao_common

#
# Basic targets to build TAO libraries.
# all     : builds the C/C++ and Fortran libraries
all       : info tao_chkcxx chktao_dir tao_chklib_dir tao_deletelibs tao_build_c tao_build_fortran tao_shared 
#
# Prints information about the system and version of TAO being compiled
#
info:
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
	-@echo "Using mpirun: ${MPIEXEC}"
	-@echo "=========================================="

