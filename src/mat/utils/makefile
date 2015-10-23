
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = convert.c matstash.c axpy.c zerodiag.c \
           getcolv.c gcreate.c freespace.c compressedrow.c multequal.c \
           matstashspace.c pheap.c bandwidth.c overlapsplit.c
SOURCEF  =
SOURCEH  = freespace.h petscheap.h
LIBBASE  = libpetscmat
DIRS     =
MANSEC   = Mat
LOCDIR   = src/mat/utils/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

