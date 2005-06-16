
#requirespackage   'PETSC_HAVE_FORTRAN'
ALL: lib
CFLAGS   =
FFLAGS   =
SOURCEC  = zgcommf.c  zoptionsf.c  zpgnamef.c  zpnamef.c
OBJSC    = ${SOURCEC:.c=.o}
SOURCEF  =
SOURCEH  =
DIRS     =
LIBBASE  = libpetsc
LOCDIR   = src/sys/objects/ftn-custom/
include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
