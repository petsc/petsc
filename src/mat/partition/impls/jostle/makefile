ALL: lib

#requirespackage 'PETSC_HAVE_JOSTLE'

CFLAGS   =
FFLAGS   =
CPPFLAGS = ${JOSTLE_INCLUDE}
SOURCEC  = jostle.c
SOURCEH  = 
OBJSC    = jostle.o
LIBBASE  = libpetscmat
LOCDIR   = src/mat/partition/impls/jostle/
MANSEC   = MatOrderings



include ${PETSC_DIR}/conf/base
include ${PETSC_DIR}/conf/test
