#!/bin/sh
# $Id: solid.make,v 1.8 1997/12/06 00:42:57 balay Exp balay $ 

# Defaults
hme="/home/petsc/petsc-2.0.22"
src_dir=""
action="lib"

# process the command line arguments
for arg in "$@" ; do
#    echo procs sing arg $arg
    case "$arg" in 
        -echo)
        set -x
        ;;

        -help | -h)
        echo "Description: "
        echo " This program is used to build petsc.solid libraries on the variety."
        echo " of platforms it is built on."
        echo " "
        echo "Options:"
        echo "  PETSC_DIR=petsc_dir : the current installation of petsc"
        echo "  SRC_DIR=src_dir     : the petsc src dir where make should be invoked"
        echo "  ACTION=action       : defaults to \"lib\" "
        echo " "
        echo "Example Usage:"
        echo "  - To update the libraries with changes in src/sles/interface"
        echo "  solid.make PETSC_DIR=/home/petsc/petsc-2.0.22 SRC_DIR=src/sles/interface ACTION=lib"
        echo "  - To rebuild a new version of PETSC on all the machines"
        echo "  solid.make PETSC_DIR=/home/petsc/petsc-2.0.22 SRC_DIR=\"\" ACTION=\"all fortran\" "
        echo " "
        echo "Defaults:"
        echo "  PETSC_DIR=$hme SRC_DIR=$src_dir ACTION=$action"
        echo " "
        echo "Notes:"
        echo " It is better that this routine be invoked by ~petsc"
        exit 1
        ;;

        PETSC_DIR=*)
        hme=`echo $arg|sed 's/PETSC_DIR=//g'`
        ;;

        SRC_DIR=*)
        src_dir=`echo $arg|sed 's/SRC_DIR=//g'`
        ;;

        ACTION=*)
        action=`echo $arg|sed 's/ACTION=//g'`
        ;;

        *) 
        echo " ignoring option $arg"
        ;;
    esac
done


# IRIX
arch=IRIX
make="make PETSC_ARCH=$arch PETSC_DIR=$hme $action shared"
#make="make PETSC_ARCH=$arch PETSC_DIR=$hme $action"
rsh -n violet "cd $hme/$src_dir; $make BOPT=g"
rsh -n violet "cd $hme/$src_dir; $make BOPT=O"

# solaris
arch=solaris
make="make PETSC_ARCH=$arch PETSC_DIR=$hme $action shared"
#make="make PETSC_ARCH=$arch PETSC_DIR=$hme $action"
rsh -n fire "cd $hme/$src_dir; $make BOPT=g"
rsh -n fire "cd $hme/$src_dir; $make BOPT=O"



# rs6000
arch=rs6000
make="make PETSC_ARCH=$arch PETSC_DIR=$hme $action"
rsh -n ico09 "cd $hme/$src_dir; $make BOPT=g"
rsh -n ico09 "cd $hme/$src_dir; $make BOPT=O"
rsh -n ico09 "cd $hme/$src_dir; $make BOPT=g_c++"
rsh -n ico09 "cd $hme/$src_dir; $make BOPT=O_c++"
rsh -n ico09 "cd $hme/$src_dir; $make BOPT=g_complex"
rsh -n ico09 "cd $hme/$src_dir; $make BOPT=O_complex"


arch=IRIX64
make="make PETSC_ARCH=$arch PETSC_DIR=$hme $action"
rsh -n yukon "cd $hme/$src_dir; $make BOPT=g"
rsh -n yukon "cd $hme/$src_dir; $make BOPT=O"
rsh -n yukon "cd $hme/$src_dir; $make BOPT=g_complex"
rsh -n yukon "cd $hme/$src_dir; $make BOPT=O_complex"


# sun4
arch=sun4
make="make PETSC_ARCH=$arch PETSC_DIR=$hme $action"
rsh -n eagle "cd $hme/$src_dir; $make BOPT=g"
rsh -n eagle "cd $hme/$src_dir; $make BOPT=O"
#rsh -n eagle "cd $hme/$src_dir; $make BOPT=g_c++"
#rsh -n eagle "cd $hme/$src_dir; $make BOPT=O_c++"
#rsh -n maverick "cd $hme/$src_dir; $make BOPT=g_complex"


# rs6000_p4
arch=rs6000_p4
make="make PETSC_ARCH=$arch PETSC_DIR=$hme $action"
rsh -n doc "cd $hme/$src_dir; $make BOPT=g"
rsh -n doc "cd $hme/$src_dir; $make BOPT=O"

