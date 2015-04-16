#requirespackage 'PETSC_HAVE_PTHREADCLASSES'
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = tcpthread.c tclockfree.c
SOURCEF  =
SOURCEH  = tcpthreadimpl.h
LIBBASE  = libpetscsys
MANSEC   = PetscThreadComm
LOCDIR   = src/sys/threadcomm/impls/pthread/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

