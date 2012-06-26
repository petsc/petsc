#requirespackage 'PETSC_HAVE_PTHREADCLASSES'
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = tcpthread.c tclockfree.c
SOURCEF  =
SOURCEH  = tcpthreadimpl.h
OBJSC    = $(SOURCEC:.c=.o)
OBJSF    =
LIBBASE  = libpetscsys
MANSEC   = PetscThreadComm
LOCDIR   = src/sys/threadcomm/impls/pthread/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

