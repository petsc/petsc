# $Id: makefile,v 1.11 2000/08/17 04:50:42 bsmith Exp balay $ 

ALL: lib 

CFLAGS    =
FFLAGS    =
CPPFLAGS  = -DPETSC_LIB_DIR='"${PETSC_LIB_DIR}"' -DPETSC_DIR='"${PETSC_DIR}"'
SOURCEC	  = ffpath.c ftest.c ghome.c mpiuopen.c rpath.c \
            fpath.c fwd.c grpath.c mprint.c sysio.c fretrieve.c smatlab.c
SOURCEF	  =
SOURCEH	  = mprint.h
OBJSC	  = ffpath.o ftest.o ghome.o mpiuopen.o rpath.o \
            fpath.o fwd.o grpath.o mprint.o sysio.o fretrieve.o smatlab.o
OBJSF	  =
MANSEC	  = Sys
LIBBASE	  = libpetsc
DIRS	  = 
LOCDIR	  = src/sys/src/fileio/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test



