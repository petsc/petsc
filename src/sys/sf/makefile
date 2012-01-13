ALL: lib

SOURCEH	 = ../../../include/private/sfimpl.h ../../../include/petscsf.h
SOURCEC  = dlregissf.c sf.c
OBJSC	 = $(SOURCEC:.c=.o)
OBJSF	 =
LIBBASE	 = libpetscvec
DIRS	 = examples
LOCDIR   = src/sys/sf/
MANSEC   = PetscSF

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

