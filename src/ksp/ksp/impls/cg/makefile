-include ../../../../../petscdir.mk
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = cg.c cgeig.c cgtype.c cgls.c
SOURCEF  =
SOURCEH  = cgimpl.h
LIBBASE  = libpetscksp
DIRS     = cgne gltr nash stcg pipecg pipecgrr groppcg pipelcg pipeprcg
MANSEC   = KSP
LOCDIR   = src/ksp/ksp/impls/cg/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
