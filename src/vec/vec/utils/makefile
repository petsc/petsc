
ALL: lib

CFLAGS   = ${PNETCDF_INCLUDE}
FFLAGS   =
SOURCEC  = vinv.c vscat.c vpscat.c cmesh.c vecio.c \
           comb.c vecstash.c vecmpitoseq.c
SOURCEF  =
SOURCEH  = vpscat.h
OBJSC    = vinv.o vscat.o vpscat.o cmesh.o vecio.o \
           comb.o vecstash.o vecmpitoseq.o
OBJSF    =
DIRS     = matlab
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/vec/utils/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

