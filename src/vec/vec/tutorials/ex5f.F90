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
  PetscClassId classid

  PetscBool :: flg
#if defined(PETSC_USE_LOG)
  PetscLogEvent  VECTOR_GENERATE,VECTOR_READ
#endif

  call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
  if (ierr /= 0) then
   print*,'PetscInitialize failed'
   stop
  endif

  call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)

  call MPI_Comm_size(PETSC_COMM_WORLD,mySize,ierr);CHKERRA(ierr)  !gives number of processes in the group of comm (integer)
  call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-m",m,flg,ierr);CHKERRA(ierr) !gives the integer value for a particular option in the database.

  ! PART 1:  Generate vector, then write it in binary format */

  call PetscLogEventRegister("Generate Vector",classid,VECTOR_GENERATE,ierr);CHKERRA(ierr)
  call PetscLogEventBegin(VECTOR_GENERATE,ierr);CHKERRA(ierr)
  ! Generate vector
  call VecCreate(PETSC_COMM_WORLD,u,ierr);CHKERRA(ierr)
  call VecSetSizes(u,PETSC_DECIDE,m,ierr);CHKERRA(ierr)
  call VecSetFromOptions(u,ierr);CHKERRA(ierr)
  call VecGetOwnershipRange(u,low,high,ierr);CHKERRA(ierr)
  call VecGetLocalSize(u,ldim,ierr);CHKERRA(ierr)
  do i=0,ldim-1
   iglobal = i + low
   v       = real(i + 100*rank)
   call VecSetValues(u,one,iglobal,v,INSERT_VALUES,ierr);CHKERRA(ierr)
  end do
  call VecAssemblyBegin(u,ierr);CHKERRA(ierr)
  call VecAssemblyEnd(u,ierr);CHKERRA(ierr)
  call VecView(u,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)

  call PetscPrintf(PETSC_COMM_WORLD,"writing vector in binary to vector.dat ...\n",ierr);CHKERRA(ierr)
  call PetscViewerBinaryOpen(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_WRITE,viewer,ierr);CHKERRA(ierr)
  call VecView(u,viewer,ierr);CHKERRA(ierr)
  call PetscViewerDestroy(viewer,ierr);CHKERRA(ierr)
  call VecDestroy(u,ierr);CHKERRA(ierr)
  call PetscOptionsSetValue(PETSC_NULL_OPTIONS,"-viewer_binary_mpiio",PETSC_NULL_CHARACTER,ierr);CHKERRA(ierr)

  call PetscLogEventEnd(VECTOR_GENERATE,ierr);CHKERRA(ierr)

  ! PART 2:  Read in vector in binary format

  ! Read new vector in binary format
  call PetscLogEventRegister("Read Vector",classid,VECTOR_READ,ierr);CHKERRA(ierr)
  call PetscLogEventBegin(VECTOR_READ,ierr);CHKERRA(ierr)
  call PetscPrintf(PETSC_COMM_WORLD,"reading vector in binary from vector.dat ...\n",ierr);CHKERRA(ierr)
  call PetscViewerBinaryOpen(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_READ,viewer,ierr);CHKERRA(ierr)
  call VecCreate(PETSC_COMM_WORLD,u,ierr);CHKERRA(ierr)
  call VecLoad(u,viewer,ierr);CHKERRA(ierr)
  call PetscViewerDestroy(viewer,ierr);CHKERRA(ierr)

  call PetscLogEventEnd(VECTOR_READ,ierr);CHKERRA(ierr)
  call VecView(u,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)

  ! Free data structures
  call VecDestroy(u,ierr);CHKERRA(ierr)
  call PetscFinalize(ierr);CHKERRA(ierr)

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
