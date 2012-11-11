#requiresscalar real

ALL: lib

CFLAGS   = 
FFLAGS   =
SOURCEC  = agmres.c agmresorthog.c agmresLeja.c
SOURCEF  =
SOURCEH  = agmresimpl.h
OBJSC    = agmres.o agmresorthog.o agmresLeja.o agmresDeflation.o
OBJSF    =
LIBBASE  = libagmres.a
MANSEC   = KSP
LOCDIR   = src/ksp/ksp/impls/gmres/agmres
DIRS     =

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
