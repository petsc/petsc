#requirespackage    'PETSC_HAVE_MPIUNI'

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC	 = mpi.c
SOURCEF	 = 
SOURCEH	 = ../../../include/mpiuni/mpi.h ../../../include/mpiuni/mpif.h
SCRIPTS  = ../../../../bin/mpiexec.uni
LIBBASE	 = libpetscsys
DIRS     = f90-mod
LOCDIR   = src/sys/mpiuni/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
