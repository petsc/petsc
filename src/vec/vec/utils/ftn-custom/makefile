
#requirespackage   'PETSC_HAVE_FORTRAN'
ALL: lib
CFLAGS   =
FFLAGS   =
SOURCEC  = zveciof.c  zvscatf.c zvecmpitoseqf.c
OBJSC    = zveciof.o  zvscatf.o zvecmpitoseqf.o
SOURCEF  =
SOURCEH  =
DIRS     =
LIBBASE  = libpetscvec
LOCDIR   = src/vec/vec/utils/ftn-custom/
include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
