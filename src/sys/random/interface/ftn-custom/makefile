#requiresdefine   'PETSC_HAVE_FORTRAN'

ALL: lib

CFLAGS    =
FFLAGS    =
SOURCEC	  = zrandomf.c
SOURCEF	  =
SOURCEH	  =
OBJSC	  = ${SOURCEC:.c=.o}
OBJSF	  =
MANSEC	  = Sys
LIBBASE	  = libpetscsys
DIRS	  =
LOCDIR	  = src/sys/random/interface/ftn-custom/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

