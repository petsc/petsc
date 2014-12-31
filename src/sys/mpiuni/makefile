#requirespackage    'PETSC_HAVE_MPIUNI'

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC	 = mpi.c mpitime.c
SOURCEF	 =
SOURCEH	 = ../../../include/petsc-mpiuni/mpi.h ../../../include/petsc-mpiuni/mpif.h
SCRIPTS  = ../../../../bin/petsc-mpiexec.uni
LIBBASE	 = libpetscsys
DIRS     = f90-mod
LOCDIR   = src/sys/mpiuni/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
