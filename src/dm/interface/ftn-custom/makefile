#requiresdefine   'PETSC_HAVE_FORTRAN'

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = zdmf.c zdmgetf.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = ${SOURCEC:.c=.o}
OBJSF    = 
LIBBASE  = libpetscdm
MANSEC   = DM
LOCDIR   = src/dm/interface/ftn-custom/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test


