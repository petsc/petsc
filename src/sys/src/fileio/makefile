# $Id: makefile,v 1.7 1999/01/12 23:14:14 bsmith Exp balay $ 

ALL: lib 

CFLAGS    =
FFLAGS    =
CPPFLAGS  = -DPETSC_LDIR='"${LDIR}"' -DPETSC_DIR='"${PETSC_DIR}"'
SOURCEC	  = ffpath.c ftest.c ghome.c mpiuopen.c rpath.c \
            fpath.c fwd.c grpath.c mprint.c sysio.c fretrieve.c
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = ffpath.o ftest.o ghome.o mpiuopen.o rpath.o \
            fpath.o fwd.o grpath.o mprint.o sysio.o fretrieve.o
OBJSF	  =
MANSEC	  = Sys
LIBBASE	  = libpetsc
DIRS	  = 
LOCDIR	  = src/sys/src/fileio/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base




