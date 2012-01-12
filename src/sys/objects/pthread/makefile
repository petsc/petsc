#requirespackage 'PETSC_HAVE_PTHREADCLASSES'
ALL: lib

CFLAGS   = ${VALGRIND_INCLUDE}
FFLAGS   =
SOURCEC  = pthread.c pthreadpool_main.c pthreadpool_chain.c pthreadpool_tree.c pthreadpool_true.c lockfree.c nopool.c
SOURCEF  =
SOURCEH  = ../../../include/petscoptions.h
OBJSC    = pthread.o pthreadpool_main.o pthreadpool_chain.o pthreadpool_tree.o pthreadpool_true.o lockfree.o nopool.o
OBJSF    =
LIBBASE  = libpetscsys
MANSEC   = Sys
LOCDIR   = src/sys/objects/pthread/
DIRS     = 

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

