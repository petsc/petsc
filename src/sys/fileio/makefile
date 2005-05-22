
ALL: lib 

CFLAGS    = 
FFLAGS    =
CPPFLAGS  = -DPETSC_LIB_DIR='"${PETSC_LIB_DIR}"'
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

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test



