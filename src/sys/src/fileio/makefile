# $Id: makefile,v 1.2 1998/05/08 16:12:44 bsmith Exp bsmith $ 

ALL: lib 
CFLAGS	  = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' -DPETSC_BOPT='"${BOPT}"' \
            -DPETSC_LDIR='"${LDIR}"' -DPETSC_DIR='"${PETSC_DIR}"' -DPETSC_SLSUFFIX='"${SLSUFFIX}"'

SOURCEC	  = ffpath.c     ftest.c      ghome.c      mpiuopen.c   rpath.c \
            fpath.c      fwd.c        grpath.c     mprint.c     sysio.c
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = ffpath.o     ftest.o      ghome.o      mpiuopen.o   rpath.o \
            fpath.o      fwd.o        grpath.o     mprint.o     sysio.o
OBJSF	  =
MANSEC	  = 7
LIBBASE	  = libpetscsys
DIRS	  = 
LOCDIR	  = src/sys/src/fileio/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base




