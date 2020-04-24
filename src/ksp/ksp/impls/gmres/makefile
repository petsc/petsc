-include ../petscdir.mk
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = gmres.c borthog.c borthog2.c gmres2.c gmreig.c gmpre.c
SOURCEH  = gmresimpl.h
SOURCEF  =
LIBBASE  = libpetscksp
DIRS     = lgmres fgmres dgmres pgmres pipefgmres agmres
MANSEC   = KSP
LOCDIR   = src/ksp/ksp/impls/gmres/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
