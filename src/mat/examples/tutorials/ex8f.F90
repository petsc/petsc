!
program main
#include <petsc/finclude/petscmat.h>

use petscmat
use iso_c_binding 
use ex8fmod
  implicit none
  
  Mat            mat
  PetscInt   :: i,j,n,a,b
  PetscInt   :: Ii,Jf
  PetscInt,parameter   :: m = 2
  PetscErrorCode ierr
  PetscScalar ::   v
  PetscScalar,parameter :: nonef= -1.0
  PetscMPIInt  ::  rank,sizef


 call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
  
 if (ierr /= 0) then
   print*,'PetscInitialize failed'
   stop
  endif

  call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr);CHKERRA(ierr)
  call MPI_Comm_size(PETSC_COMM_WORLD,sizef,ierr);CHKERRA(ierr)
  
  n = 2*sizef
  print*, n ,m

  !/* create the matrix */
  call MatCreate(PETSC_COMM_WORLD,mat,ierr);CHKERRA(ierr)
  call MatSetSizes(mat,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,ierr);CHKERRA(ierr)
  call MatSetType(mat,MATAIJ,ierr);CHKERRA(ierr)
  call MatSetUp(mat,ierr);CHKERRA(ierr)
  !/* register user defined MatScaleUser() operation for both SeqAIJ
     !and MPIAIJ types */
  call RegisterMatScaleUserImpl(mat,ierr);CHKERRA(ierr)

  !/* assemble the matrix */
  do i=0,m-1
    do j=2*rank,2*rank+1
      v = -1.0  
      Ii = j + n*i
      if (i>0) then  
       Jf = Ii - n; call MatSetValues(mat,1,Ii,1,Jf,v,INSERT_VALUES,ierr);CHKERRA(ierr)
      end if
      if (i<m-1) then
       Jf = Ii + n; call MatSetValues(mat,1,Ii,1,Jf,v,INSERT_VALUES,ierr);CHKERRA(ierr)
      end if
      if (j>0) then 
       Jf = Ii - 1; call MatSetValues(mat,1,Ii,1,Jf,v,INSERT_VALUES,ierr);CHKERRA(ierr)
      end if
      if (j<n-1) then
       Jf = Ii + 1; call MatSetValues(mat,1,Ii,1,Jf,v,INSERT_VALUES,ierr);CHKERRA(ierr)
      end if
      v = 4.0;      call MatSetValues(mat,1,Ii,1,Ii,v,INSERT_VALUES,ierr);CHKERRA(ierr)
    end do
  end do
  call MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY,ierr);CHKERRA(ierr)
  call MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY,ierr);CHKERRA(ierr)

  call MatGetSize(mat,a,b,ierr)
  print*, a,b
  !/* check the matrix before and after scaling by -1.0 */
  call PetscPrintf(PETSC_COMM_WORLD,"Matrix _before_ MatScaleUserImpl() operation\n",ierr);CHKERRA(ierr)
  call MatView(mat,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)
  !call MatScaleUserImpl(mat,nonef,ierr);CHKERRA(ierr)
  call PetscPrintf(PETSC_COMM_WORLD,"Matrix _after_ MatScaleUserImpl() operation\n",ierr);CHKERRA(ierr)
  call MatView(mat,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)

  call MatDestroy(mat,ierr);CHKERRA(ierr)
  call PetscFinalize(ierr);CHKERRA(ierr)

!static char help[] = "Shows how to add a new MatOperation to AIJ MatType\n\n"

!#include <petscmat.h>
!#include <petscblaslapack.h>
contains

!/* This routine registers MatScaleUserImpl_SeqAIJ() and
   !MatScaleUserImpl_MPIAIJ() as methods providing MatScaleUserImpl()
   !functionality for SeqAIJ and MPIAIJ matrix-types */
   
SUBROUTINE RegisterMatScaleUserImpl(mat1,ierr)

 implicit none
 PetscErrorCode :: ierr
 PetscMPIInt    sizef
 Mat :: mat1
 Mat AA,AB

 call  MPI_Comm_size(PETSC_COMM_WORLD,sizef,ierr);CHKERRA(ierr)

    if (sizef == 1)  then 
      call PetscObjectComposeFunction(mat1,"MatScaleUserImpl_C",c_funloc(matscaleuserimpl_seqaij),ierr);CHKERRA(ierr)
    else 
      call MatMPIAIJGetSeqAIJ(mat1,AA,AB,PETSC_NULL_CHARACTER,ierr);CHKERRA(ierr)
      call PetscObjectComposeFunction(mat1,"MatScaleUserImpl_C",c_funloc(MatScaleUserImpl_MPIAIJ),ierr);CHKERRA(ierr)
      call PetscObjectComposeFunction(AA,"MatScaleUserImpl_C",c_funloc(matscaleuserimpl_seqaij),ierr);CHKERRA(ierr)
      call PetscObjectComposeFunction(AB,"MatScaleUserImpl_C",c_funloc(matscaleuserimpl_seqaij),ierr);CHKERRA(ierr)
    end if  



end SUBROUTINE RegisterMatScaleUserImpl

end program

