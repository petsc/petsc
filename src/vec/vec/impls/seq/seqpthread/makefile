#requirespackage 'PETSC_HAVE_PTHREADCLASSES'
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = vecpthread.c tmap.c
SOURCEF  =
SOURCEH  = vecpthreadimpl.h
OBJSC    = vecpthread.o tmap.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/vec/impls/seq/seqpthread/
DIRS     = 

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

