# $Id: makefile,v 1.1 2000/03/05 03:42:22 bsmith Exp balay $

ALL: lib

CFLAGS   = -I../include -I/home/alice/esi/cxx/include
#CFLAGS   = -I/home/bsmith/esi -features=castop,rtti
FFLAGS   =
SOURCEC  = PETSc_Map.c ESIf77_Map.c  PETSc_Vector.c ESIf77_Vector.c \
           ESI_Object_test.c ESI_Map_test.c ESI_MapAlgebraic_test.c \
           ESI_Vector_test.c ESI_VectorPointerAccess_test.c PETSc_Matrix.c \
           ESI_Matrix_test.c
SOURCEF  = ESIf77_Object_test.F ESIf77_Map_test.F ESIf77_MapAlgebraic_test.F \
           ESIf77_Vector_test.F ESIf77_VectorPointerAccess_test.F
SOURCEH  = PETSc_Map.h PETSc_Vector.h PETSc_Matrix.h
OBJSC    = PETSc_Map.o ESIf77_Map.o  PETSc_Vector.o ESIf77_Vector.o \
           ESI_Object_test.o ESI_Map_test.o ESI_MapAlgebraic_test.o \
           ESI_Vector_test.o ESI_VectorPointerAccess_test.o PETSc_Matrix.o \
           ESI_Matrix_test.o
OBJSF    = ESIf77_Object_test.o ESIf77_Map_test.o ESIf77_MapAlgebraic_test.o \
           ESIf77_Vector_test.o ESIf77_VectorPointerAccess_test.o
LIBBASE  = libpetscesi
MANSEC   = 1
LOCDIR   = src/contrib/esi/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

ex1: ex1.o  chkopts 
	-${CLINKER} -o ex1 ex1.o -L${LDIR} -lpetscesi ${PETSC_VEC_LIB}
	${RM} -f ex1.o
ex2: ex2.o  chkopts 
	-${CLINKER} -o ex2 ex2.o -L${LDIR} -lpetscesi ${PETSC_VEC_LIB}
	${RM} -f ex2.o
ex3: ex3.o  chkopts 
	-${CLINKER} -o ex3 ex3.o -L${LDIR} -lpetscesi ${PETSC_MAT_LIB}
	${RM} -f ex3.o
ex1f: ex1f.o  chkopts 
	-${CLINKER} -o ex1f ex1f.o -L${LDIR} -lpetscesi \
               ${PETSC_FORTRAN_LIB} ${PETSC_VEC_LIB}
	${RM} -f ex1f.o
ex2f: ex2f.o  chkopts 
	-${CLINKER} -o ex2f ex2f.o -L${LDIR} -lpetscesi \
               ${PETSC_FORTRAN_LIB} ${PETSC_VEC_LIB}
	${RM} -f ex2f.o


