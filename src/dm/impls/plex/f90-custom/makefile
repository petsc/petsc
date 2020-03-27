-include ../petscdir.mk
#requiresdefine   'PETSC_HAVE_FORTRAN'
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = zplexf90.c zplexgeometryf90.c zplexsectionf90.c zplexfemf90.c
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscdm
LOCDIR   = src/dm/impls/plex/f90-custom/


include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
