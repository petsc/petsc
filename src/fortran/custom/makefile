# $Id: makefile,v 1.30 2001/08/22 18:02:57 balay Exp balay $ 

ALL: lib

CFLAGS   = -DPETSC_FORTRAN_STUBS
FFLAGS   =
SOURCEC  = zoptions.c zksp.c zpc.c zsnes.c zsys.c zmat.c zvec.c zsles.c \
           zdraw.c zda.c zviewer.c zis.c zplog.c zstart.c zstartf.c zts.c \
	   zao.c zsda.c
SOURCEF  = 
SOURCEH  = zpetsc.h
OBJSC    = zoptions.o zksp.o zpc.o zsnes.o zsys.o zmat.o zvec.o zsles.o \
           zdraw.o zda.o zviewer.o zis.o zplog.o zstart.o zstartf.o zts.o \
	   zao.o zsda.o
OBJSF    = 
LIBBASE  = libpetscfortran
MANSEC   = Sys
LOCDIR   = src/fortran/custom/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test


