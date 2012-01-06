
#requirespackage   'PETSC_HAVE_UMFPACK'

ALL: lib

CFLAGS   = ${UMFPACK_INCLUDE}
FFLAGS   =
SOURCEC  = amd.c
SOURCEF  =
SOURCEH  =
OBJSC    = amd.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     =
MANSEC   = Mat
LOCDIR   = src/mat/order/amd/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
