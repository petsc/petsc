!
!  "$Id: ex16f90.F,v 1.11 1998/07/06 16:00:04 balay Exp bsmith $";
!
!
!  Tests MatGetArray() on a dense matrix
!

      program ex16f90
      implicit none


! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                    Include files
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  The following include statements are required for Fortran programs
!  that use PETSc vectors:
!     petsc.h  - base PETSc routines
!     vec.h - defines (INSERT_VALUES)
!     mat.h    - matrices
!     mat.h90  - to allow access to Fortran 90 features of matrices

#include "include/finclude/petsc.h"
#include "include/finclude/vec.h"
#include "include/finclude/mat.h"
#include "include/finclude/mat.h90"

      Mat A
      integer i,j,m,n,ierr
      Scalar  v
      Scalar, pointer :: array(:,:)


      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

      m = 3
      n = 2
!
!      Create a parallel dense matrix shared by all processors 
!
      call MatCreateMPIDense(PETSC_COMM_WORLD,PETSC_DECIDE,                &
     &                       PETSC_DECIDE,m,n,PETSC_NULL_SCALAR,A,ierr)

!
!     Set values into the matrix. All processors set all values.
!
      do 10, i=0,m-1
        do 20, j=0,n-1
          v = 9.d0/(i+j+1)
          call MatSetValues(A,1,i,1,j,v,INSERT_VALUES,ierr)
 20     continue
 10   continue

      call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr) 
      call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr) 

!
!       Print the matrix to the screen 
!
      call MatView(A,VIEWER_STDOUT_WORLD,ierr) 


!
!      Print the local portion of the matrix to the screen
!
      call MatGetArrayF90(A,array,ierr) 
      call PetscSequentialPhaseBegin(PETSC_COMM_WORLD,1,ierr)
      write(6,100) PetscRealPart(array)
 100  format(2F6.2)
      call PetscSequentialPhaseEnd(PETSC_COMM_WORLD,1,ierr)

      call MatRestoreArrayF90(A,array,ierr)   
!
!      Free the space used by the matrix
!
      call MatDestroy(A,ierr) 
      call PetscFinalize(ierr)
      end

