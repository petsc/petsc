
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = gmres.c borthog.c borthog2.c gmres2.c gmreig.c gmpre.c
SOURCEH  = gmresimpl.h
SOURCEF  = 
OBJSC    = gmres.o borthog.o borthog2.o gmres2.o gmreig.o gmpre.o
OBJSF    = 
LIBBASE  = libpetscksp
DIRS     = lgmres fgmres dgmres pgmres
MANSEC   = KSP
LOCDIR   = src/ksp/ksp/impls/gmres/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
