#requirespackage 'PETSC_HAVE_BG'

ALL: lib

SOURCEH	 = ../../../include/private/bgimpl.h ../../../include/petscbg.h
SOURCEC  = dlregisbg.c bg.c
OBJSC	 = $(SOURCEC:.c=.o)
OBJSF	 =
LIBBASE	 = libpetscvec
DIRS	 =
LOCDIR   = src/vec/bg/
MANSEC   = PetscBG

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

