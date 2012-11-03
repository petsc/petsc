
ALL: lib

CFLAGS   = ${PNETCDF_INCLUDE}
FFLAGS   =
SOURCEC  = vinv.c vscat.c vpscat.c cmesh.c vecio.c \
           comb.c vecstash.c vecmpitoseq.c
SOURCEF  =
SOURCEH  = vpscat.h
DIRS     = matlab veccusp
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/vec/utils/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

