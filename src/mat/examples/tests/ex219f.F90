program newnonzero
#include <petsc/finclude/petscmat.h>
 use petscmat
 implicit none

 Mat :: A
 PetscInt :: n,m,idxm(1),idxn(1),nl1,nl2
 PetscScalar :: v(1)
 PetscErrorCode :: ierr

 integer :: nproc,iproc,i

 call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

 call MPI_COMM_SIZE(PETSC_COMM_WORLD, nproc,ierr)

 call MPI_Comm_rank( PETSC_COMM_WORLD, iproc, ierr )

 n=3
 m=n
 call MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,m,1,PETSC_NULL_INTEGER,0,PETSC_NULL_INTEGER,A,ierr)


 call MatGetOwnershipRange(A,nl1,nl2,ierr)
 do i=nl1,nl2-1
   idxn(1)=i
   idxm(1)=i
   v(1)=1d0
   call MatSetValues(A,1,idxn,1,idxm, v,INSERT_VALUES,ierr)
 end do
 call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr)
 call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr)

! Ignore any values set into new nonzero locations
 call MatSetOption(A,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE,ierr)

 idxn(1)=0
 idxm(1)=n-1
 if ((idxn(1).ge.nl1).and.(idxn(1).le.nl2-1)) then
   v(1)=2.0
   call MatSetValues(A,1,idxn,1,idxm, v,INSERT_VALUES,ierr);CHKERRQ(ierr)
 end if
 call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr)
 call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr)

 if ((idxn(1).ge.nl1).and.(idxn(1).le.nl2-1)) then
   call MatGetValues(A,1,idxn,1,idxm, v,ierr)
   write(6,*) v
 end if

 call MatDestroy(A,ierr)
 call PetscFinalize(ierr)

 end program newnonzero

!/*TEST
!
!     test:
!       nsize: 2
!       filter: Error:
!
!     test:
!       suffix: 2
!       nsize: 2
!       args: -info
!       filter: grep "Skipping"
!
!TEST*/
