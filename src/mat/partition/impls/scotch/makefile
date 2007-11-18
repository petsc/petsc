ALL: lib

#requirespackage 'PETSC_HAVE_SCOTCH'

CFLAGS   =
FFLAGS   =
CPPFLAGS = ${SCOTCH_INCLUDE}
SOURCEC  = scotch.c
SOURCEH  = 
OBJSC    = scotch.o
LIBBASE  = libpetscmat
LOCDIR   = src/mat/partition/impls/scotch/
MANSEC   = MatOrderings



include ${PETSC_DIR}/conf/base
include ${PETSC_DIR}/conf/test
