ALL: lib

#requirespackage 'PETSC_HAVE_PTSCOTCH'

CFLAGS   =
FFLAGS   =
CPPFLAGS = ${SCOTCH_INCLUDE}
SOURCEC  = scotch.c
SOURCEH  = 
OBJSC    = scotch.o
LIBBASE  = libpetscmat
LOCDIR   = src/mat/partition/impls/scotch/
MANSEC   = MatOrderings

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
