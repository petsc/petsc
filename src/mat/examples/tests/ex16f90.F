
C  "$Id: ex16f90.F,v 1.3 1997/04/04 16:11:40 curfman Exp balay $";

C
C  Tests MatGetArray() on a dense matrix
C

      program ex16f90
      implicit none


C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C                    Include files
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C
C  The following include statements are required for Fortran programs
C  that use PETSc vectors:
C     petsc.h  - base PETSc routines
C     vec.h - defines (INSERT_VALUES)
C     mat.h    - matrices
C     mat.h90  - to allow access to Fortran 90 features of matrices

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/mat.h90"

      Mat A
      integer i,j,m,n,ierr
      Scalar  v
      Scalar, pointer :: array(:,:)


      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

      m = 3
      n = 2
C
C      Create a parallel dense matrix shared by all processors 
C
      call MatCreateMPIDense(PETSC_COMM_WORLD,PETSC_DECIDE,
     &                        PETSC_DECIDE,m,n,PETSC_NULL,A,ierr)

C
C     Set values into the matrix. All processors set all values.
C
      do 10, i=0,m-1
        do 20, j=0,n-1
          v = 1.d0/(i+j+1)
          call MatSetValues(A,1,i,1,j,v,INSERT_VALUES,ierr)
 20     continue
 10   continue

      call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr) 
      call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr) 

C
C       Print the matrix to the screen 
C
      call MatView(A,VIEWER_STDOUT_WORLD,ierr) 


C
C      Print the local portion of the matrix to the screen
C
      call MatGetArrayF90(A,array,ierr) 
      call PetscSequentialPhaseBegin(PETSC_COMM_WORLD,1,ierr)
      print*, array
      call PetscSequentialPhaseEnd(PETSC_COMM_WORLD,1,ierr)

      call MatRestoreArrayF90(A,array,ierr)   
C
C      Free the space used by the matrix
C
      call MatDestroy(A,ierr) 
      call PetscFinalize(ierr)
      end

