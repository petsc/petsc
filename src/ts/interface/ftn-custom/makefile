
#requirespackage   'PETSC_HAVE_FORTRAN'
ALL: lib
CFLAGS   =
FFLAGS   =
SOURCEC  = ztscreatef.c  ztsf.c  ztsregf.c ztsfdf.c
OBJSC    = ${SOURCEC:.c=.o}
SOURCEF  =
SOURCEH  =
DIRS     =
LIBBASE  = libpetscts
LOCDIR   = src/ts/interface/ftn-custom/
include ${PETSC_DIR}/conf/base
include ${PETSC_DIR}/conf/test
