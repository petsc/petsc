program main

#include <petsc/finclude/petscvec.h>

use petscvec
implicit none

  PetscErrorCode ierr
  Vec   v,s
  PetscInt,parameter ::      n   = 20
  PetscScalar,parameter ::   sone = 1.0
  PetscBool :: flg
  PetscInt,parameter :: zero = 0, one = 1, two = 2

 call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
  if (ierr /= 0) then
   print*,'PetscInitialize failed'
   stop
  endif

  call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-n",n,flg,ierr);CHKERRA(ierr)


  !Create multi-component vector with 2 components
  call VecCreate(PETSC_COMM_WORLD,v,ierr);CHKERRA(ierr)
  call VecSetSizes(v,PETSC_DECIDE,n,ierr);CHKERRA(ierr)
  call VecSetBlockSize(v,two,ierr);CHKERRA(ierr)
  call VecSetFromOptions(v,ierr);CHKERRA(ierr)


  !Create single-component vector
  call VecCreate(PETSC_COMM_WORLD,s,ierr);CHKERRA(ierr)
  call VecSetSizes(s,PETSC_DECIDE,n/2,ierr);CHKERRA(ierr)
  call VecSetFromOptions(s,ierr);CHKERRA(ierr)

  !Set the vectors to entries to a constant value.
  call VecSet(v,sone,ierr);CHKERRA(ierr)

  !Get the first component from the multi-component vector to the single vector
  call VecStrideGather(v,zero,s,INSERT_VALUES,ierr);CHKERRA(ierr)

  call VecView(s,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)


  !Put the values back into the second component
  call VecStrideScatter(s,one,v,ADD_VALUES,ierr);CHKERRA(ierr)

  call VecView(v,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)


  !Free work space.All PETSc objects should be destroyed when they are no longer needed.

  call VecDestroy(v,ierr);CHKERRA(ierr)
  call VecDestroy(s,ierr);CHKERRA(ierr)
  call PetscFinalize(ierr);CHKERRA(ierr)

  end program

!/*TEST
!
!     test:
!       nsize: 2
!       output_file: output/ex12_1.out
!
!TEST*/
