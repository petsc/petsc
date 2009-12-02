
ALL: lib

CFLAGS   =
FFLAGS   =
CPPFLAGS =
SOURCEC  = baij.c baij2.c baijfact.c baijfact2.c dgefa.c dgedi.c dgefa3.c \
	   dgefa4.c dgefa5.c dgefa2.c dgefa6.c dgefa7.c aijbaij.c baijfact3.c baijfact4.c \
           baijfact5.c baijfact7.c baijfact9.c \
           baijfact11.c baijfact13.c
SOURCEF  =
SOURCEH  = baij.h
OBJSC    = baij.o baij2.o baijfact.o baijfact2.o dgefa.o dgedi.o dgefa3.o \
	   dgefa4.o dgefa5.o dgefa2.o dgefa6.o dgefa7.o aijbaij.o baijfact3.o baijfact4.o \
           baijfact5.o baijfact7.o baijfact9.o \
           baijfact11.o baijfact13.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = ftn-kernels
MANSEC   = Mat
LOCDIR   = src/mat/impls/baij/seq/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
