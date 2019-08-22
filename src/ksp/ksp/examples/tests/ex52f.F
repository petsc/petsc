!
      program main
#include <petsc/finclude/petscksp.h>
      use petscksp
      implicit none
!
!  Demonstrates using MatFactorGetError() and MatFactorGetErrorZeroPivot()
!

      PetscErrorCode  ierr
      PetscInt m,n,one,row,col
      Vec              x,b
      Mat              A,F
      KSP              ksp
      PetscScalar two,zero
      KSPConvergedReason reason
      PCFailedReason pcreason
      PC pc
      MatFactorError ferr
      PetscReal pivot

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif

      m = 2
      n = 2
      call MatCreate(PETSC_COMM_WORLD,A,ierr)
      call MatSetSizes(A,m,n,m,n,ierr)
      call MatSetType(A, MATSEQAIJ,ierr)
      call MatSetUp(A,ierr)
      row = 0
      col = 0
      two = 2.0
      one = 1
      call MatSetValues(A,one,row,one,col,two,INSERT_VALUES,ierr)
      row = 1
      col = 1
      zero = 0.0
      call MatSetValues(A,one,row,one,col,zero,INSERT_VALUES,ierr)
      call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr)

      call VecCreate(PETSC_COMM_WORLD,b,ierr)
      call VecSetSizes(b,m,m,ierr)
      call VecSetType(b,VECSEQ,ierr)

! Set up solution
      call VecDuplicate(b,x,ierr)

! Solve system
      call KSPCreate(PETSC_COMM_WORLD,ksp,ierr)
      call KSPSetOperators(ksp,A,A,ierr)
      call KSPSetFromOptions(ksp,ierr)
      call KSPSolve(ksp,b,x,ierr)
      call KSPGetConvergedReason(ksp,reason,ierr)
      call KSPGetPC(ksp,pc,ierr)
      call PCGetFailedReason(pc,pcreason,ierr)
      call PCFactorGetMatrix(pc,F,ierr)
      call MatFactorGetError(F,ferr,ierr)
      call MatFactorGetErrorZeroPivot(F,pivot,row,ierr)
      write(6,101) ferr,pivot,row
 101  format('MatFactorError ',i4,' Pivot value ',e12.4,' row ',i4)

! Cleanup
      call KSPDestroy(ksp,ierr)
      call VecDestroy(b,ierr)
      call VecDestroy(x,ierr)
      call MatDestroy(A,ierr)

      call PetscFinalize(ierr)
      end

!  Nag compiler automatically turns on catching of floating point exceptions and prints message at 
!  end of run about the exceptions seen
!
!/*TEST
!
!   test:
!     args: -fp_trap 0
!     filter: grep -v "Warning: Floating"
!
!TEST*/
