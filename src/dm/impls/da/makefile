
ALL: lib

CFLAGS   = ${MATLAB_INCLUDE} ${PNETCDF_INCLUDE} 
FFLAGS   =
SOURCEC  = da2.c da1.c da3.c daghost.c dacorn.c dagtol.c daltol.c \
           daindex.c dascatter.c dacreate.c dadestroy.c dalocal.c \
           dadist.c daview.c dasub.c gr1.c gr2.c daload.c dagtona.c \
	   dainterp.c dapf.c dagetarray.c dagetelem.c da.c dareg.c fdda.c mhyp.c
OBJSC    = da2.o da1.o da3.o daghost.o dacorn.o dagtol.o daltol.o \
           daindex.o dascatter.o dacreate.o dadestroy.o dalocal.o \
           dadist.o daview.o dasub.o gr1.o gr2.o daload.o dagtona.o \
	   dainterp.o dapf.o dagetarray.o dagetelem.o da.o dareg.o fdda.o mhyp.o
SOURCEH  = ../../../../include/private/daimpl.h ../../../../include/petscdm.h mhyp.h
DIRS     = usfft utils
LOCDIR   = src/dm/impls/da/
MANSEC   = DM

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

