
ALL: lib

CFLAGS   = ${PNETCDF_INCLUDE}
FFLAGS   =
SOURCEC  = vinv.c vscat.c vpscat.c cmesh.c vecio.c \
           comb.c vecstash.c vecmpitoseq.c
SOURCEF  =
SOURCEH  = 
OBJSC    = vinv.o vscat.o vpscat.o cmesh.o vecio.o \
           comb.o vecstash.o vecmpitoseq.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/utils/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

