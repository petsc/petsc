PETSC_DIR = ../../..

ALL: lib

CFLAGS   = $(OPT) $(PETSC_INCLUDE) -I../..  \
           $(CONF) -D__DIR__='"fortran/custom"'
FFLAGS   = 
SOURCEC  = zoptions.c zksp.c zpc.c zsnes.c zsys.c zmat.c zvec.c zsles.c \
           zdraw.c zda.c zviewer.c
SOURCEF  = somefort.F
SOURCEH  = zpetsc.h
OBJSC    = zoptions.o zksp.o zpc.o zsnes.o zsys.o zmat.o zvec.o zsles.o \
           zdraw.o zda.o zviewer.o
OBJSF    = somefort.o
LIBBASE  = libpetscfortran
MANSEC   = 1

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

