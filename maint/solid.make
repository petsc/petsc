#!/bin/sh

# Defaults
hme="/home/petsc/petsc-2.0.17"
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
        echo "  solid.make PETSC_DIR=/home/petsc/petsc-2.0.17 SRC_DIR=src/sles/interface ACTION=lib"
        echo "  - To rebuild a new version of PETSC on all the machines"
        echo "  solid.make PETSC_DIR=/home/petsc/petsc-2.0.18 SRC_DIR=\"\" ACTION=\"all fortran\" "
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

# sun4
arch=sun4
make="make PETSC_ARCH=$arch PETSC_DIR=$hme $action shared"
rsh maverick "cd $hme/$src_dir; $make BOPT=g"
rsh maverick "cd $hme/$src_dir; $make BOPT=O"
rsh maverick "cd $hme/$src_dir; $make BOPT=g_c++"
rsh maverick "cd $hme/$src_dir; $make BOPT=O_c++"
rsh maverick "cd $hme/$src_dir; $make BOPT=g_complex"


# IRIX
arch=IRIX
make="make PETSC_ARCH=$arch PETSC_DIR=$hme $action shared"
rsh cyan "cd $hme/$src_dir; $make BOPT=g"
rsh cyan "cd $hme/$src_dir; $make BOPT=O"
rsh cyan "cd $hme/$src_dir; $make BOPT=g_c++"


# rs6000_p4
arch=rs6000_p4
make="make PETSC_ARCH=$arch PETSC_DIR=$hme $action"
rsh doc "cd $hme/$src_dir; $make BOPT=g"
rsh doc "cd $hme/$src_dir; $make BOPT=O"


# rs6000
arch=rs6000
make="make PETSC_ARCH=$arch PETSC_DIR=$hme $action"
rsh clyde "cd $hme/$src_dir; $make BOPT=O"
rsh clyde "cd $hme/$src_dir; $make BOPT=g_c++"
rsh clyde "cd $hme/$src_dir; $make BOPT=O_c++"




# rs6000_quad
arch=rs6000_quad
make="make PETSC_ARCH=$arch PETSC_DIR=$hme lib"
rsh quad "cd $hme/$src_dir; $make BOPT=g"
rsh quad "cd $hme/$src_dir; $make BOPT=O"
rsh quad "cd $hme/$src_dir; $make BOPT=g_c++"
rsh quad "cd $hme/$src_dir; $make BOPT=O_c++"

