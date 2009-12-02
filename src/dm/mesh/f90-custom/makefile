
#requirespackage 'PETSC_USING_F90'
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = zmeshf90.c zsectionf90.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = ${SOURCEC:.c=.o}
OBJSF    = 
LIBBASE  = libpetscdm
LOCDIR   = src/dm/mesh/f90-custom/


include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
