
#requirespackage   'PETSC_HAVE_FORTRAN'
ALL: lib
CFLAGS   =
FFLAGS   =
SOURCEC  = zda1f.c zda2f.c zda3f.c zdacornf.c zdagetscatterf.c zdaindexf.c zdainterpf.c zdaloadf.c zdaviewf.c
OBJSC    = ${SOURCEC:.c=.o}
SOURCEF  =
SOURCEH  =
DIRS     =
LIBBASE  = libpetscdm
LOCDIR   = src/dm/da/src/ftn/custom/
include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
