#requirespackage 'PETSC_HAVE_PTHREADCLASSES'
ALL: lib

CFLAGS   = ${VALGRIND_INCLUDE}
FFLAGS   =
SOURCEC  = pthread.c lockfree.c # pthreadpool_main.c pthreadpool_chain.c pthreadpool_tree.c pthreadpool_true.c nopool.c
SOURCEF  =
SOURCEH  = pthreadimpl.h
OBJSC    = pthread.o lockfree.o # pthreadpool_main.o pthreadpool_chain.o pthreadpool_tree.o pthreadpool_true.o nopool.o
OBJSF    =
LIBBASE  = libpetscsys
MANSEC   = Sys
LOCDIR   = src/sys/objects/pthread/
DIRS     = 

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

