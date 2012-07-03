#requiresdefine   'PETSC_HAVE_FORTRAN'

ALL: lib

CFLAGS   = 
FFLAGS   =
SOURCEC  = zplog.c zstart.c zstartf.c zsys.c  zutils.c
SOURCEF  = 
SOURCEH  = ../../../include/petsc-private/fortranimpl.h
OBJSC    =  zplog.o zstart.o zstartf.o zsys.o zutils.o
OBJSF    = 
LIBBASE  = libpetscsys
MANSEC   = Sys
LOCDIR   = src/sys/ftn-custom/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test


