-include ../../../../../petscdir.mk

CFLAGS    = ${GAMG_INCLUDE}
SOURCEC   = gamg.c agg.c geo.c util.c classical.c
LIBBASE   = libpetscksp
MANSEC    = KSP
SUBMANSEC = PC

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
