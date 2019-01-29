
module ex8fmod
#include <petsc/finclude/petscmat.h>

use petscmat
use iso_c_binding 

  implicit none
 
  public :: &
    MatScaleUserImpl_SeqAIJ, &
    MatScaleUserImpl_MPIAIJ

  contains

PetscErrorCode Function MatScaleUserImpl_SeqAIJ(inA,alpha)

  implicit none
  PetscErrorCode :: ierr
  
  Mat :: inA
  PetscScalar :: alpha

  call MatScale(inA,alpha,ierr);CHKERRA(ierr)
  MatScaleUserImpl_SeqAIJ = ierr

end function MatScaleUserImpl_SeqAIJ

PetscErrorCode Function MatScaleUserImpl_MPIAIJ(Af,aa1,ierr)

 implicit none
 PetscErrorCode :: ierr
 Mat                  ::     Af,AA,AB
 PetscScalar :: aa1

  call MatMPIAIJGetSeqAIJ(Af,AA,AB,PETSC_NULL_CHARACTER,ierr);CHKERRA(ierr)
  ierr = MatScaleUserImpl(AA,aa1)
  ierr = MatScaleUserImpl(AB,aa1)
  
end Function MatScaleUserImpl_MPIAIJ

!/* this routines queries the already registered MatScaleUserImp_XXX
   !implementations for the given matrix, and calls the correct
   !routine. i.e if MatType is SeqAIJ, MatScaleUserImpl_SeqAIJ() gets
   !called, and if MatType is MPIAIJ, MatScaleUserImpl_MPIAIJ() gets
   !called */
   
   !Interface

function  MatScaleUserImpl(mat2,a)

   implicit none
   PetscErrorCode :: MatScaleUserImpl
  PetscErrorCode :: ierr
   PetscScalar :: a
   Mat:: mat2
   type(c_funptr) :: fc         !The c pointer definition
   procedure(), pointer :: ff    !The fortran pointer definition

   call PetscObjectQueryFunction(mat2,"MatScaleUserImpl_C",fc,ierr);CHKERRQ(ierr)
   
   call c_f_procpointer(fc,ff)    !Assign the target of the C function pointer CPTR to the Fortran procedure pointer FPTR. 
   call ff(mat2,a,ierr);CHKERRQ(ierr)
   
end function MatScaleUserImpl


end module

