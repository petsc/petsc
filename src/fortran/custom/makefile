# $Id: makefile,v 1.28 2001/02/26 04:53:38 bsmith Exp bsmith $ 

ALL: lib
SPECIALFASTLIB = yes
specialfastlib: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = zoptions.c zksp.c zpc.c zsnes.c zsys.c zmat.c zvec.c zsles.c \
           zdraw.c zda.c zviewer.c zis.c zplog.c zstart.c zstartf.c zts.c \
	   zao.c 
SOURCEF  = somefort.F
SOURCEH  = zpetsc.h
OBJSC    = zoptions.o zksp.o zpc.o zsnes.o zsys.o zmat.o zvec.o zsles.o \
           zdraw.o zda.o zviewer.o zis.o zplog.o zstart.o zstartf.o zts.o \
	   zao.o 
OBJSF    = somefort.o
LIBBASE  = libpetscfortran
MANSEC   = Sys
LOCDIR   = src/fortran/custom/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test


