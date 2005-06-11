
#requirespackage   'PETSC_HAVE_FORTRAN'
ALL: lib
CFLAGS   =
FFLAGS   =
SOURCEC  = zaobasicf.c
OBJSC    = ${SOURCEC:.c=.o}
SOURCEF  =
SOURCEH  =
DIRS     =
LIBBASE  = libpetsdm
LOCDIR   = src/dm/ao/impls/basic/ftn-custom/
include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
