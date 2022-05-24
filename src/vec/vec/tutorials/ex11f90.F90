  program main
#include <petsc/finclude/petscvec.h>
  use petscvec
  implicit none

  Vec        ::   x
  PetscReal  :: norm
  PetscMPIInt :: rank
  PetscInt,parameter :: n = 20
  PetscErrorCode :: ierr
  PetscScalar,parameter :: sone = 1.0
  PetscBool :: flg
  character(len=PETSC_MAX_PATH_LEN) :: outputString
  PetscInt,parameter :: zero = 0, one = 1, two = 2

  PetscCallA(PetscInitialize(ierr))

  PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))

  PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-n",n,flg,ierr))

     !Create a vector, specifying only its global dimension.
     !When using VecCreate(), VecSetSizes() and VecSetFromOptions(),
     !the vector format (currently parallel,
     !shared, or sequential) is determined at runtime.  Also, the parallel
     !partitioning of the vector is determined by PETSc at runtime.

     !Routines for creating particular vector types directly are:
     !VecCreateSeq() - uniprocessor vector
     !VecCreateMPI() - distributed vector, where the user can
                         !determine the parallel partitioning
     !VecCreateShared() - parallel vector that uses shared memory
                            !(available only on the SGI) otherwise,
                            !is the same as VecCreateMPI()

     !With VecCreate(), VecSetSizes() and VecSetFromOptions() the option
     !-vec_type mpi or -vec_type shared causes the
     !particular type of vector to be formed.

  PetscCallA(VecCreate(PETSC_COMM_WORLD,x,ierr))

  PetscCallA(VecSetSizes(x,PETSC_DECIDE,n,ierr))
  !
  PetscCallA(VecSetBlockSize(x,two,ierr))
  PetscCallA(VecSetFromOptions(x,ierr))

     !Set the vectors to entries to a constant value.

  PetscCallA(VecSet(x,sone,ierr))

  PetscCallA(VecNorm(x,NORM_2,norm,ierr))
  write(outputString,*) norm
  PetscCallA(PetscPrintf(PETSC_COMM_WORLD,"L_2 Norm of entire vector: "//trim(outputString)//"\n",ierr))

  PetscCallA(VecNorm(x,NORM_1,norm,ierr))
  write(outputString,*) norm
  PetscCallA(PetscPrintf(PETSC_COMM_WORLD,"L_1 Norm of entire vector: "//trim(outputString)//"\n",ierr))

  PetscCallA(VecNorm(x,NORM_INFINITY,norm,ierr))
  write(outputString,*) norm
  PetscCallA(PetscPrintf(PETSC_COMM_WORLD,"L_inf Norm of entire vector: "//trim(outputString)//"\n",ierr))

  PetscCallA(VecStrideNorm(x,zero,NORM_2,norm,ierr))
  write(outputString,*) norm
  PetscCallA(PetscPrintf(PETSC_COMM_WORLD,"L_2 Norm of sub-vector 0: "//trim(outputString)//"\n",ierr))

  PetscCallA(VecStrideNorm(x,zero,NORM_1,norm,ierr))
  write(outputString,*) norm
  PetscCallA(PetscPrintf(PETSC_COMM_WORLD,"L_1 Norm of sub-vector 0: "//trim(outputString)//"\n",ierr))

  PetscCallA(VecStrideNorm(x,zero,NORM_INFINITY,norm,ierr))
  write(outputString,*) norm
  PetscCallA(PetscPrintf(PETSC_COMM_WORLD,"L_inf Norm of sub-vector 0: "//trim(outputString)//"\n",ierr))

  PetscCallA(VecStrideNorm(x,one,NORM_2,norm,ierr))
  write(outputString,*) norm
  PetscCallA(PetscPrintf(PETSC_COMM_WORLD,"L_2 Norm of sub-vector 1: "//trim(outputString)//"\n",ierr))

  PetscCallA(VecStrideNorm(x,one,NORM_1,norm,ierr))
  write(outputString,*) norm
  PetscCallA(PetscPrintf(PETSC_COMM_WORLD,"L_1 Norm of sub-vector 1: "//trim(outputString)//"\n",ierr))

  PetscCallA(VecStrideNorm(x,one,NORM_INFINITY,norm,ierr))
  write(outputString,*) norm
  PetscCallA(PetscPrintf(PETSC_COMM_WORLD,"L_inf Norm of sub-vector 1: "//trim(outputString)//"\n",ierr))

  !Free work space.  All PETSc objects should be destroyed when they
  !are no longer needed.
  PetscCallA(VecDestroy(x,ierr))
  PetscCallA(PetscFinalize(ierr))

end program

!/*TEST
!
!     test:
!       nsize: 2
!
!TEST*/
