program main
#include <petsc/finclude/petscvec.h>
use petscvec
implicit none

  PetscErrorCode ::ierr
  PetscMPIInt ::   rank,mySize
  PetscInt :: i
  PetscInt, parameter :: one = 1
  PetscInt :: m = 10
  PetscInt :: low,high,ldim,iglobal
  PetscScalar :: v
  Vec ::         u
  PetscViewer :: viewer
#if defined(PETSC_USE_LOG)
  PetscClassId classid
#endif

  PetscBool :: flg
#if defined(PETSC_USE_LOG)
  PetscLogEvent  VECTOR_GENERATE,VECTOR_READ
#endif

  PetscCallA(PetscInitialize(ierr))

  PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))

  PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD,mySize,ierr))  !gives number of processes in the group of comm (integer)
  PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-m",m,flg,ierr)) !gives the integer value for a particular option in the database.

  ! PART 1:  Generate vector, then write it in binary format */

#if defined(PETSC_USE_LOG)
  PetscCallA(PetscLogEventRegister("Generate Vector",classid,VECTOR_GENERATE,ierr))
  PetscCallA(PetscLogEventBegin(VECTOR_GENERATE,ierr))
#endif
  ! Create vector
  PetscCallA(VecCreate(PETSC_COMM_WORLD,u,ierr))
  PetscCallA(VecSetSizes(u,PETSC_DECIDE,m,ierr))
  PetscCallA(VecSetFromOptions(u,ierr))
  PetscCallA(VecGetOwnershipRange(u,low,high,ierr))
  PetscCallA(VecGetLocalSize(u,ldim,ierr))
  do i=0,ldim-1
   iglobal = i + low
   v       = real(i + 100*rank)
   PetscCallA(VecSetValues(u,one,iglobal,v,INSERT_VALUES,ierr))
  end do
  PetscCallA(VecAssemblyBegin(u,ierr))
  PetscCallA(VecAssemblyEnd(u,ierr))
  PetscCallA(VecView(u,PETSC_VIEWER_STDOUT_WORLD,ierr))

  PetscCallA(PetscPrintf(PETSC_COMM_WORLD,"writing vector in binary to vector.dat ...\n",ierr))
  PetscCallA(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_WRITE,viewer,ierr))
  PetscCallA(VecView(u,viewer,ierr))
  PetscCallA(PetscViewerDestroy(viewer,ierr))
  PetscCallA(VecDestroy(u,ierr))
  PetscCallA(PetscOptionsSetValue(PETSC_NULL_OPTIONS,"-viewer_binary_mpiio",PETSC_NULL_CHARACTER,ierr))

#if defined(PETSC_USE_LOG)
  PetscCallA(PetscLogEventEnd(VECTOR_GENERATE,ierr))
#endif

  ! PART 2:  Read in vector in binary format

  ! Read new vector in binary format
#if defined(PETSC_USE_LOG)
  PetscCallA(PetscLogEventRegister("Read Vector",classid,VECTOR_READ,ierr))
  PetscCallA(PetscLogEventBegin(VECTOR_READ,ierr))
#endif
  PetscCallA(PetscPrintf(PETSC_COMM_WORLD,"reading vector in binary from vector.dat ...\n",ierr))
  PetscCallA(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_READ,viewer,ierr))
  PetscCallA(VecCreate(PETSC_COMM_WORLD,u,ierr))
  PetscCallA(VecLoad(u,viewer,ierr))
  PetscCallA(PetscViewerDestroy(viewer,ierr))

#if defined(PETSC_USE_LOG)
  PetscCallA(PetscLogEventEnd(VECTOR_READ,ierr))
#endif
  PetscCallA(VecView(u,PETSC_VIEWER_STDOUT_WORLD,ierr))

  ! Free data structures
  PetscCallA(VecDestroy(u,ierr))
  PetscCallA(PetscFinalize(ierr))

end program

!/*TEST
!
!     test:
!       nsize: 1
!       requires: mpiio
!       output_file: output/ex5_1.out
!
!     test:
!       suffix: 2
!       nsize: 2
!       requires: mpiio
!       output_file: output/ex5_2.out
!
!TEST*/
