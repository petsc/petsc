ALL: lib

#requirespackage 'PETSC_HAVE_CHACO'

CFLAGS   =
FFLAGS   =
CPPFLAGS = ${CHACO_INCLUDE}
SOURCEC  = chaco.c
SOURCEH  = 
OBJSC    = chaco.o
LIBBASE  = libpetscmat
LOCDIR   = src/mat/partition/impls/chaco/
MANSEC   = MatOrderings

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
