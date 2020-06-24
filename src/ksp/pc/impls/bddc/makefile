-include ../../../../../petscdir.mk
ALL: lib

CFLAGS    =
FFLAGS    =
SOURCEC   = bddc.c bddcprivate.c bddcgraph.c bddcscalingbasic.c bddcnullspace.c bddcfetidp.c bddcschurs.c
SOURCEF   =
SOURCEH   = bddc.h bddcstructs.h bddcprivate.h
LIBBASE   = libpetscksp
DIRS      =
MANSEC    = KSP
SUBMANSEC = PC
LOCDIR    = src/ksp/pc/impls/bddc/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
