#requirespackage    'PETSC_HAVE_MPIUNI'

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC	 = mpi.c
SOURCEF	 = 
SOURCEH	 = ../../../include/mpiuni/mpi.h ../../../include/mpiuni/mpif.h
OBJSC	 = mpi.o
OBJSF	 =
SCRIPTS  = ../../../../bin/mpirun.uni
LIBBASE	 = libmpiuni
LOCDIR   = src/sys/mpiuni/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
