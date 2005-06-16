
#requirespackage   'PETSC_HAVE_FORTRAN'
ALL: lib
CFLAGS   =
FFLAGS   =
SOURCEC  = zshellpcf.c
OBJSC    = ${SOURCEC:.c=.o}
SOURCEF  =
SOURCEH  =
DIRS     =
LIBBASE  = libpetscksp
LOCDIR   = src/ksp/pc/impls/shell/ftn-custom/
include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
