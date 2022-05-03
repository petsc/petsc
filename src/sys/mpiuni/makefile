-include ../../../petsckdir.mk
#requirespackage    'PETSC_HAVE_MPIUNI'

SOURCEC	 = mpi.c mpitime.c
SOURCEF	 =
SOURCEH	 = ../../../include/petsc/mpiuni/mpi.h ../../../include/petsc/mpiuni/mpif.h
SCRIPTS  = ../../../../lib/petsc/bin/petsc-mpiexec.uni
LIBBASE	 = libpetscsys
DIRS     = fsrc f90-mod
LOCDIR   = src/sys/mpiuni/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
