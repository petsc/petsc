
ALL: lib

#requirespackage   'PETSC_HAVE_SUNDIALS'
#requiresprecision double
#requiresscalar    real

CFLAGS   =
FFLAGS   =
CPPFLAGS = ${SUNDIALS_INCLUDE} ${SUNDIALS_INCLUDE}/cvode ${SUNDIALS_INCLUDE}/sundials
SOURCEC  = sundials.c
SOURCEF  =
SOURCEH  = sundials.h
OBJSC    = sundials.o
OBJSF    =
LIBBASE  = libpetscts
MANSEC   = TS
LOCDIR   = src/ts/impls/implicit/sundials/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

