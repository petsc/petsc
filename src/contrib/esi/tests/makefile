# $Id: makefile,v 1.4 2001/08/24 21:03:37 bsmith Exp bsmith $

ALL: lib

CFLAGS   = -I../include ${ESI_INCLUDE}
#CFLAGS   = -I/home/bsmith/esi -features=castop,rtti
FFLAGS   =
SOURCEC  = ESI_Object_test.c ESI_Map_test.c ESI_MapPartition_test.c \
           ESI_Vector_test.c ESI_Matrix_test.c
OBJSC    = ESI_Object_test.o ESI_Map_test.o ESI_MapPartition_test.o \
           ESI_Vector_test.o ESI_Matrix_test.o
LIBBASE  = libpetscesi
MANSEC   = 1
LOCDIR   = src/contrib/esi/

include ${PETSC_DIR}/bmake/common/base

ex1: ex1.o  chkopts 
	-${CLINKER} -o ex1 ex1.o -L${PETSC_LIB_DIR} -lpetscesi ${PETSC_VEC_LIB}
	${RM} -f ex1.o
ex2: ex2.o  chkopts 
	-${CLINKER} -o ex2 ex2.o -L${PETSC_LIB_DIR} -lpetscesi ${PETSC_VEC_LIB}
	${RM} -f ex2.o
ex3: ex3.o  chkopts 
	-${CLINKER} -o ex3 ex3.o -L${PETSC_LIB_DIR} -lpetscesi ${PETSC_MAT_LIB}
	${RM} -f ex3.o
