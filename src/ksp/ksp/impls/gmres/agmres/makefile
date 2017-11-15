#requiresscalar real
#requirespackage AGMRES_CURRENTLY_DISABLED

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = agmres.c agmresorthog.c agmresleja.c agmresdeflation.c
SOURCEF  =
SOURCEH  = agmresimpl.h
OBJSF    =
LIBBASE  = libpetscksp
MANSEC   = KSP
LOCDIR   = src/ksp/ksp/impls/gmres/agmres/
DIRS     =

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
