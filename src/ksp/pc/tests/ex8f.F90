!
!   Tests PCMGSetResidual
!
! -----------------------------------------------------------------------

      program main
#include <petsc/finclude/petscksp.h>
      use petscksp
      implicit none

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Variables:
!     ksp     - linear solver context
!     x, b, u  - approx solution, right-hand-side, exact solution vectors
!     A        - matrix that defines linear system
!     its      - iterations for convergence
!     norm     - norm of error in solution
!     rctx     - random number context
!

      Mat              A
      Vec              x,b,u
      PC               pc
      PetscInt  n,dim,istart,iend
      PetscInt  i,j,jj,ii,one,zero
      PetscErrorCode ierr
      PetscScalar v
      external         MyResidual
      PetscScalar      pfive
      KSP              ksp

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      PetscCallA(PetscInitialize(ierr))
      pfive = .5
      n      = 6
      dim    = n*n
      one    = 1
      zero   = 0

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!      Compute the matrix and right-hand-side vector that define
!      the linear system, Ax = b.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Create parallel matrix, specifying only its global dimensions.
!  When using MatCreate(), the matrix format can be specified at
!  runtime. Also, the parallel partitioning of the matrix is
!  determined by PETSc at runtime.

      PetscCallA(MatCreate(PETSC_COMM_WORLD,A,ierr))
      PetscCallA(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,dim,dim,ierr))
      PetscCallA(MatSetFromOptions(A,ierr))
      PetscCallA(MatSetUp(A,ierr))

!  Currently, all PETSc parallel matrix formats are partitioned by
!  contiguous chunks of rows across the processors.  Determine which
!  rows of the matrix are locally owned.

      PetscCallA(MatGetOwnershipRange(A,Istart,Iend,ierr))

!  Set matrix elements in parallel.
!   - Each processor needs to insert only elements that it owns
!     locally (but any non-local elements will be sent to the
!     appropriate processor during matrix assembly).
!   - Always specify global rows and columns of matrix entries.

      do 10, II=Istart,Iend-1
        v = -1.0
        i = II/n
        j = II - i*n
        if (i.gt.0) then
          JJ = II - n
          PetscCallA(MatSetValues(A,one,II,one,JJ,v,ADD_VALUES,ierr))
        endif
        if (i.lt.n-1) then
          JJ = II + n
          PetscCallA(MatSetValues(A,one,II,one,JJ,v,ADD_VALUES,ierr))
        endif
        if (j.gt.0) then
          JJ = II - 1
          PetscCallA(MatSetValues(A,one,II,one,JJ,v,ADD_VALUES,ierr))
        endif
        if (j.lt.n-1) then
          JJ = II + 1
          PetscCallA(MatSetValues(A,one,II,one,JJ,v,ADD_VALUES,ierr))
        endif
        v = 4.0
        PetscCallA( MatSetValues(A,one,II,one,II,v,ADD_VALUES,ierr))
 10   continue

!  Assemble matrix, using the 2-step process:
!       MatAssemblyBegin(), MatAssemblyEnd()
!  Computations can be done while messages are in transition
!  by placing code between these two statements.

      PetscCallA(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr))
      PetscCallA(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr))

!  Create parallel vectors.
!   - Here, the parallel partitioning of the vector is determined by
!     PETSc at runtime.  We could also specify the local dimensions
!     if desired.
!   - Note: We form 1 vector from scratch and then duplicate as needed.

      PetscCallA(VecCreate(PETSC_COMM_WORLD,u,ierr))
      PetscCallA(VecSetSizes(u,PETSC_DECIDE,dim,ierr))
      PetscCallA(VecSetFromOptions(u,ierr))
      PetscCallA(VecDuplicate(u,b,ierr))
      PetscCallA(VecDuplicate(b,x,ierr))

!  Set exact solution; then compute right-hand-side vector.

      PetscCallA(VecSet(u,pfive,ierr))
      PetscCallA(MatMult(A,u,b,ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!         Create the linear solver and set various options
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Create linear solver context

      PetscCallA(KSPCreate(PETSC_COMM_WORLD,ksp,ierr))
      PetscCallA(KSPGetPC(ksp,pc,ierr))
      PetscCallA(PCSetType(pc,PCMG,ierr))
      PetscCallA(PCMGSetLevels(pc,one,PETSC_NULL_MPI_COMM,ierr))
      PetscCallA(PCMGSetResidual(pc,zero,MyResidual,A,ierr))

!  Set operators. Here the matrix that defines the linear system
!  also serves as the preconditioning matrix.

      PetscCallA(KSPSetOperators(ksp,A,A,ierr))

      PetscCallA(KSPDestroy(ksp,ierr))
      PetscCallA(VecDestroy(u,ierr))
      PetscCallA(VecDestroy(x,ierr))
      PetscCallA(VecDestroy(b,ierr))
      PetscCallA(MatDestroy(A,ierr))

      PetscCallA(PetscFinalize(ierr))
      end

      subroutine MyResidual(A,b,x,r,ierr)
      use petscksp
      implicit none
      Mat A
      Vec b,x,r
      integer ierr
      return
      end

!/*TEST
!
!   test:
!      nsize: 1
!TEST*/
