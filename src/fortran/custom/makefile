PETSC_DIR = ../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) -I../.. $(CONF) -D__DIR__='"$(LOCDIR)"'
FFLAGS   = 
SOURCEC  = zoptions.c zksp.c zpc.c zsnes.c zsys.c zmat.c zvec.c zsles.c \
           zdraw.c zda.c zviewer.c zis.c zplog.c zstart.c
SOURCEF  = somefort.F
SOURCEH  = zpetsc.h
OBJSC    = zoptions.o zksp.o zpc.o zsnes.o zsys.o zmat.o zvec.o zsles.o \
           zdraw.o zda.o zviewer.o zis.o zplog.o zstart.o
OBJSF    = somefort.o
LIBBASE  = libpetscfortran
MANSEC   = 7
LOCDIR   = src/fortran/custom/
include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

