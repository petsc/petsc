
#requirespackage   'PETSC_HAVE_FORTRAN'
ALL: lib
CFLAGS   =
FFLAGS   =
SOURCEC  = zerrf.c
OBJSC    = ${SOURCEC:.c=.o}
SOURCEF  =
SOURCEH  =
DIRS     =
LIBBASE  = libpetsc
LOCDIR   = src/sys/error/ftn-custom/
include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
