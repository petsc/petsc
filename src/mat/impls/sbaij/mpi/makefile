# $Id: makefile,v 1.16 1999/04/01 17:10:02 balay Exp $ 

ALL: lib

CFLAGS 	        = 
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = /disks/sandbox/hzhang/petsc.old/src/mat/impls/sbaij/mpi
SOURCEC         = mpisbaij.c mmsbaij.c sbaijov.c
SOURCEF         = 
SOURCEH         = mpisbaij.h
OBJSC           = mpisbaij.o mmsbaij.o sbaijov.o
OBJSF           = 
LIBBASE         = libpetscmat
EXAMPLESC_1     = main.c 
#EXAMPLESF      =

main: main.o mpisbaij.o mmsbaij.o sbaijov.o chkopts
	-${CLINKER} -o main main.o mpisbaij.o mmsbaij.o sbaijov.o ${PETSC_LIB} 
	${RM} main.o mpisbaij.o mmsbaij.o sbaijov.o

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base





