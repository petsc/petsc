#requirespackage   'PETSC_HAVE_FORTRAN'

ALL: lib

CFLAGS   = 
FFLAGS   =
SOURCEC  = zplog.c zstart.c zstartf.c zsys.c  zutils.c
SOURCEF  = 
SOURCEH  = 
OBJSC    =  zplog.o zstart.o zstartf.o zsys.o zutils.o
OBJSF    = 
LIBBASE  = libpetsc
MANSEC   = Sys
LOCDIR   = src/sys/ftn-custom/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test


