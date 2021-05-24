program main
#include <petsc/finclude/petscvec.h>
use petscvec
implicit none

  Vec ::           v,s,r
  Vec,pointer,dimension(:) ::  vecs
  PetscInt :: i,start
  PetscInt :: endd
  PetscInt,parameter :: n = 20, four = 4, two = 2, one = 1
  PetscErrorCode ierr
  PetscScalar  ::  myValue
  PetscBool :: flg

  call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

  if (ierr /= 0) then
    print*,'PetscInitialize failed'
    stop
  endif

  call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-n",n,flg,ierr);CHKERRA(ierr)

  !Create multi-component vector with 2 components
  call VecCreate(PETSC_COMM_WORLD,v,ierr);CHKERRA(ierr)
  call VecSetSizes(v,PETSC_DECIDE,n,ierr);CHKERRA(ierr)
  call VecSetBlockSize(v,four,ierr);CHKERRA(ierr)
  call VecSetFromOptions(v,ierr);CHKERRA(ierr)

  ! Create double-component vectors

  call VecCreate(PETSC_COMM_WORLD,s,ierr);CHKERRA(ierr)
  call VecSetSizes(s,PETSC_DECIDE,n/two,ierr);CHKERRA(ierr)
  call VecSetBlockSize(s,two,ierr);CHKERRA(ierr)
  call VecSetFromOptions(s,ierr);CHKERRA(ierr)
  call VecDuplicate(s,r,ierr);CHKERRA(ierr)
  allocate(vecs(0:2))

  vecs(0) = s
  vecs(1) = r

  !Set the vector values

  call VecGetOwnershipRange(v,start,endd,ierr);CHKERRA(ierr)
  do i=start,endd-1
     myValue = real(i)
     call VecSetValues(v,one,i,myValue,INSERT_VALUES,ierr);CHKERRA(ierr)
  end do

  ! Get the components from the multi-component vector to the other vectors

  call VecStrideGatherAll(v,vecs,INSERT_VALUES,ierr);CHKERRA(ierr)

  call VecView(s,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)
  call VecView(r,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)

  call VecStrideScatterAll(vecs,v,ADD_VALUES,ierr);CHKERRA(ierr)

  call VecView(v,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)

  !Free work space.All PETSc objects should be destroyed when they are no longer needed.

  deallocate(vecs)
  call VecDestroy(v,ierr);CHKERRA(ierr)
  call VecDestroy(s,ierr);CHKERRA(ierr)
  call VecDestroy(r,ierr);CHKERRA(ierr)
  call PetscFinalize(ierr);CHKERRA(ierr)

end program

!/*TEST
!
!     test:
!       nsize: 2
!       output_file: output/ex16_1.out
!
!TEST*/
