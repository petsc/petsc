#requirespackage 'PETSC_HAVE_PTHREADCLASSES'
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = pthread.c lockfree.c
SOURCEF  =
SOURCEH  = pthreadimpl.h
OBJSC    = pthread.o lockfree.o
OBJSF    =
LIBBASE  = libpetscsys
MANSEC   = PetscThreadComm
LOCDIR   = src/sys/threadcomm/impls/pthread/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

