   !Concepts: vectors^norms of sub-vectors
   !Processors: n
   
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

  call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
  if (ierr /= 0) then
   print*,'PetscInitialize failed'
   stop
  endif
      
  call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
      
  call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-n",n,flg,ierr);CHKERRA(ierr)


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

  call VecCreate(PETSC_COMM_WORLD,x,ierr);CHKERRA(ierr)
  
  call VecSetSizes(x,PETSC_DECIDE,n,ierr);CHKERRA(ierr)
  !
  call VecSetBlockSize(x,two,ierr);CHKERRA(ierr)
  call VecSetFromOptions(x,ierr);CHKERRA(ierr)


     !Set the vectors to entries to a constant value.
  
  call VecSet(x,sone,ierr);CHKERRA(ierr)

  call VecNorm(x,NORM_2,norm,ierr);CHKERRA(ierr)
  write(outputString,*) norm
  call PetscPrintf(PETSC_COMM_WORLD,"L_2 Norm of entire vector: "//trim(outputString)//"\n",ierr);CHKERRA(ierr)
  
  call VecNorm(x,NORM_1,norm,ierr);CHKERRA(ierr)
  write(outputString,*) norm
  call PetscPrintf(PETSC_COMM_WORLD,"L_1 Norm of entire vector: "//trim(outputString)//"\n",ierr);CHKERRA(ierr)
  
  call VecNorm(x,NORM_INFINITY,norm,ierr);CHKERRA(ierr)
  write(outputString,*) norm
  call PetscPrintf(PETSC_COMM_WORLD,"L_inf Norm of entire vector: "//trim(outputString)//"\n",ierr);CHKERRA(ierr)
  
  call VecStrideNorm(x,zero,NORM_2,norm,ierr);CHKERRA(ierr)
  write(outputString,*) norm
  call PetscPrintf(PETSC_COMM_WORLD,"L_2 Norm of sub-vector 0: "//trim(outputString)//"\n",ierr);CHKERRA(ierr)

  call VecStrideNorm(x,zero,NORM_1,norm,ierr);CHKERRA(ierr)
  write(outputString,*) norm
  call PetscPrintf(PETSC_COMM_WORLD,"L_1 Norm of sub-vector 0: "//trim(outputString)//"\n",ierr);CHKERRA(ierr)

  call VecStrideNorm(x,zero,NORM_INFINITY,norm,ierr);CHKERRA(ierr)
  write(outputString,*) norm
  call PetscPrintf(PETSC_COMM_WORLD,"L_inf Norm of sub-vector 0: "//trim(outputString)//"\n",ierr);CHKERRA(ierr)

  call VecStrideNorm(x,one,NORM_2,norm,ierr);CHKERRA(ierr)
  write(outputString,*) norm
  call PetscPrintf(PETSC_COMM_WORLD,"L_2 Norm of sub-vector 1: "//trim(outputString)//"\n",ierr);CHKERRA(ierr)

  call VecStrideNorm(x,one,NORM_1,norm,ierr);CHKERRA(ierr)
  write(outputString,*) norm
  call PetscPrintf(PETSC_COMM_WORLD,"L_1 Norm of sub-vector 1: "//trim(outputString)//"\n",ierr);CHKERRA(ierr)

  call VecStrideNorm(x,one,NORM_INFINITY,norm,ierr);CHKERRA(ierr)
  write(outputString,*) norm
  call PetscPrintf(PETSC_COMM_WORLD,"L_inf Norm of sub-vector 1: "//trim(outputString)//"\n",ierr);CHKERRA(ierr)
   
   
     !Free work space.  All PETSc objects should be destroyed when they
     !are no longer needed.
  call VecDestroy(x,ierr);CHKERRA(ierr)
  call PetscFinalize(ierr);CHKERRA(ierr)
  
end program


!/*TEST
!
!     test:
!       nsize: 2
!
!TEST*/
