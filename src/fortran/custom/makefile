# $Id: makefile,v 1.24 1998/06/15 16:22:39 balay Exp bsmith $ 

ALL: lib

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
MANSEC   = 7
LOCDIR   = src/fortran/custom/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base


