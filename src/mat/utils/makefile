
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = convert.c matstash.c axpy.c zerodiag.c matio.c matioall.c \
           getcolv.c gcreate.c freespace.c compressedrow.c multequal.c \
           ptap.c factorshift.c
SOURCEF  =
SOURCEH  = freespace.h
OBJSC    = convert.o matstash.o axpy.o zerodiag.o matio.o matioall.o \
           getcolv.o gcreate.o freespace.o compressedrow.o multequal.o \
           ptap.o factorshift.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = Mat
LOCDIR   = src/mat/utils/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

