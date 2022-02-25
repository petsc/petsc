-include ../../../../../petscdir.mk
#requiresdefine   'PETSC_HAVE_FORTRAN'
ALL: lib
CFLAGS   =
FFLAGS   =
SOURCEC  = zplexcreate.c zplexdistribute.c zplexsubmesh.c zplexexodusii.c zplexgmsh.c zplexfluent.c zplexextrude.c
SOURCEF  =
SOURCEH  =
DIRS     =
LIBBASE  = libpetscdm
LOCDIR   = src/dm/impls/plex/ftn-custom/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
