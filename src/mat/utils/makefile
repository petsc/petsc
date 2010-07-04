
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = convert.c matstash.c axpy.c zerodiag.c \
           getcolv.c gcreate.c freespace.c compressedrow.c multequal.c \
           ptap.c matstashspace.c
SOURCEF  =
SOURCEH  = freespace.h
OBJSC    = convert.o matstash.o axpy.o zerodiag.o \
           getcolv.o gcreate.o freespace.o compressedrow.o multequal.o \
           ptap.o matstashspace.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = Mat
LOCDIR   = src/mat/utils/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

