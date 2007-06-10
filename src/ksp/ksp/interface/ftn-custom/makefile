
#requirespackage   'PETSC_HAVE_FORTRAN'
ALL: lib
CFLAGS   =
FFLAGS   =
SOURCEC  = zitclf.c  zitcreatef.c  zitfuncf.c  zxonf.c
OBJSC    = ${SOURCEC:.c=.o}
SOURCEF  =
SOURCEH  =
DIRS     =
LIBBASE  = libpetscksp
LOCDIR   = src/ksp/ksp/interface/ftn-custom/
include ${PETSC_DIR}/conf/base
include ${PETSC_DIR}/conf/test
