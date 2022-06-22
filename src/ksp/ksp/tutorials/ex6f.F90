!
!  Description: This example demonstrates repeated linear solves as
!  well as the use of different preconditioner and linear system
!  matrices.  This example also illustrates how to save PETSc objects
!  in common blocks.
!
!

      program main
#include <petsc/finclude/petscksp.h>
      use petscksp
      implicit none

!  Variables:
!
!  A       - matrix that defines linear system
!  ksp    - KSP context
!  ksp     - KSP context
!  x, b, u - approx solution, RHS, exact solution vectors
!
      Vec     x,u,b
      Mat     A,A2
      KSP    ksp
      PetscInt i,j,II,JJ,m,n
      PetscInt Istart,Iend
      PetscInt nsteps,one
      PetscErrorCode ierr
      PetscBool  flg
      PetscScalar  v

      PetscCallA(PetscInitialize(ierr))
      m      = 3
      n      = 3
      nsteps = 2
      one    = 1
      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-m',m,flg,ierr))
      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-n',n,flg,ierr))
      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-nsteps',nsteps,flg,ierr))

!  Create parallel matrix, specifying only its global dimensions.
!  When using MatCreate(), the matrix format can be specified at
!  runtime. Also, the parallel partitioning of the matrix is
!  determined by PETSc at runtime.

      PetscCallA(MatCreate(PETSC_COMM_WORLD,A,ierr))
      PetscCallA(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,ierr))
      PetscCallA(MatSetFromOptions(A,ierr))
      PetscCallA(MatSetUp(A,ierr))

!  The matrix is partitioned by contiguous chunks of rows across the
!  processors.  Determine which rows of the matrix are locally owned.

      PetscCallA(MatGetOwnershipRange(A,Istart,Iend,ierr))

!  Set matrix elements.
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
        if (i.lt.m-1) then
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
!   - When using VecCreate(), the parallel partitioning of the vector
!     is determined by PETSc at runtime.
!   - Note: We form 1 vector from scratch and then duplicate as needed.

      PetscCallA(VecCreate(PETSC_COMM_WORLD,u,ierr))
      PetscCallA(VecSetSizes(u,PETSC_DECIDE,m*n,ierr))
      PetscCallA(VecSetFromOptions(u,ierr))
      PetscCallA(VecDuplicate(u,b,ierr))
      PetscCallA(VecDuplicate(b,x,ierr))

!  Create linear solver context

      PetscCallA(KSPCreate(PETSC_COMM_WORLD,ksp,ierr))

!  Set runtime options (e.g., -ksp_type <type> -pc_type <type>)

      PetscCallA(KSPSetFromOptions(ksp,ierr))

!  Solve several linear systems in succession

      do 100 i=1,nsteps
         PetscCallA(solve1(ksp,A,x,b,u,i,nsteps,A2,ierr))
 100  continue

!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.

      PetscCallA(VecDestroy(u,ierr))
      PetscCallA(VecDestroy(x,ierr))
      PetscCallA(VecDestroy(b,ierr))
      PetscCallA(MatDestroy(A,ierr))
      PetscCallA(KSPDestroy(ksp,ierr))

      PetscCallA(PetscFinalize(ierr))
      end

! -----------------------------------------------------------------------
!
      subroutine solve1(ksp,A,x,b,u,count,nsteps,A2,ierr)
      use petscksp
      implicit none

!
!   solve1 - This routine is used for repeated linear system solves.
!   We update the linear system matrix each time, but retain the same
!   preconditioning matrix for all linear solves.
!
!      A - linear system matrix
!      A2 - preconditioning matrix
!
      PetscScalar  v,val
      PetscInt II,Istart,Iend
      PetscInt count,nsteps,one
      PetscErrorCode ierr
      Mat     A
      KSP     ksp
      Vec     x,b,u

! Use common block to retain matrix between successive subroutine calls
      Mat              A2
      PetscMPIInt      rank
      PetscBool        pflag
      common /my_data/ pflag,rank

      one = 1
! First time thorough: Create new matrix to define the linear system
      if (count .eq. 1) then
        PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))
        pflag = .false.
        PetscCallA(PetscOptionsHasName(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-mat_view',pflag,ierr))
        if (pflag) then
          if (rank .eq. 0) write(6,100)
          call PetscFlush(6)
        endif
        PetscCallA(MatConvert(A,MATSAME,MAT_INITIAL_MATRIX,A2,ierr))
! All other times: Set previous solution as initial guess for next solve.
      else
        PetscCallA(KSPSetInitialGuessNonzero(ksp,PETSC_TRUE,ierr))
      endif

! Alter the matrix A a bit
      PetscCallA(MatGetOwnershipRange(A,Istart,Iend,ierr))
      do 20, II=Istart,Iend-1
        v = 2.0
        PetscCallA(MatSetValues(A,one,II,one,II,v,ADD_VALUES,ierr))
 20   continue
      PetscCallA(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr))
      if (pflag) then
        if (rank .eq. 0) write(6,110)
        call PetscFlush(6)
      endif
      PetscCallA(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr))

! Set the exact solution; compute the right-hand-side vector
      val = 1.0*real(count)
      PetscCallA(VecSet(u,val,ierr))
      PetscCallA(MatMult(A,u,b,ierr))

! Set operators, keeping the identical preconditioner matrix for
! all linear solves.  This approach is often effective when the
! linear systems do not change very much between successive steps.
      PetscCallA(KSPSetReusePreconditioner(ksp,PETSC_TRUE,ierr))
      PetscCallA(KSPSetOperators(ksp,A,A2,ierr))

! Solve linear system
      PetscCallA(KSPSolve(ksp,b,x,ierr))

! Destroy the preconditioner matrix on the last time through
      if (count .eq. nsteps) PetscCallA(MatDestroy(A2,ierr))

 100  format('previous matrix: preconditioning')
 110  format('next matrix: defines linear system')

      end

!/*TEST
!
!   test:
!      args: -pc_type jacobi -mat_view -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always
!
!TEST*/
