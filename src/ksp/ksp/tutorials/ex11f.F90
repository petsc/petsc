!
!  Description: Solves a complex linear system in parallel with KSP (Fortran code).
!

!
!  The model problem:
!     Solve Helmholtz equation on the unit square: (0,1) x (0,1)
!          -delta u - sigma1*u + i*sigma2*u = f,
!           where delta = Laplace operator
!     Dirichlet b.c.'s on all sides
!     Use the 2-D, five-point finite difference stencil.
!
!     Compiling the code:
!      This code uses the complex numbers version of PETSc, so configure
!      must be run to enable this
!
!
! -----------------------------------------------------------------------

      program main
#include <petsc/finclude/petscksp.h>
      use petscksp
      implicit none

!
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

      KSP             ksp
      Mat              A
      Vec              x,b,u
      PetscRandom      rctx
      PetscReal norm,h2,sigma1
      PetscScalar  none,sigma2,v,pfive,czero
      PetscScalar  cone
      PetscInt dim,its,n,Istart
      PetscInt Iend,i,j,II,JJ,one
      PetscErrorCode ierr
      PetscMPIInt rank
      PetscBool  flg
      logical          use_random

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      PetscCallA(PetscInitialize(ierr))
      none   = -1.0
      n      = 6
      sigma1 = 100.0
      czero  = 0.0
      cone   = PETSC_i
      PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))
      PetscCallA(PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-sigma1',sigma1,flg,ierr))
      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-n',n,flg,ierr))
      dim    = n*n

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

      PetscCallA(PetscOptionsHasName(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-norandom',flg,ierr))
      if (flg) then
         use_random = .false.
         sigma2 = 10.0*PETSC_i
      else
         use_random = .true.
         PetscCallA(PetscRandomCreate(PETSC_COMM_WORLD,rctx,ierr))
         PetscCallA(PetscRandomSetFromOptions(rctx,ierr))
         PetscCallA(PetscRandomSetInterval(rctx,czero,cone,ierr))
      endif
      h2 = 1.0/real((n+1)*(n+1))

      one = 1
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
        if (use_random) PetscCallA(PetscRandomGetValue(rctx,sigma2,ierr))
        v = 4.0 - sigma1*h2 + sigma2*h2
        PetscCallA( MatSetValues(A,one,II,one,II,v,ADD_VALUES,ierr))
 10   continue
      if (use_random) PetscCallA(PetscRandomDestroy(rctx,ierr))

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

      if (use_random) then
         PetscCallA(PetscRandomCreate(PETSC_COMM_WORLD,rctx,ierr))
         PetscCallA(PetscRandomSetFromOptions(rctx,ierr))
         PetscCallA(VecSetRandom(u,rctx,ierr))
      else
         pfive = 0.5
         PetscCallA(VecSet(u,pfive,ierr))
      endif
      PetscCallA(MatMult(A,u,b,ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!         Create the linear solver and set various options
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Create linear solver context

      PetscCallA(KSPCreate(PETSC_COMM_WORLD,ksp,ierr))

!  Set operators. Here the matrix that defines the linear system
!  also serves as the preconditioning matrix.

      PetscCallA(KSPSetOperators(ksp,A,A,ierr))

!  Set runtime options, e.g.,
!      -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>

      PetscCallA(KSPSetFromOptions(ksp,ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                      Solve the linear system
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      PetscCallA(KSPSolve(ksp,b,x,ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                     Check solution and clean up
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Check the error

      PetscCallA(VecAXPY(x,none,u,ierr))
      PetscCallA(VecNorm(x,NORM_2,norm,ierr))
      PetscCallA(KSPGetIterationNumber(ksp,its,ierr))
      if (rank .eq. 0) then
        if (norm .gt. 1.e-12) then
           write(6,100) norm,its
        else
           write(6,110) its
        endif
      endif
  100 format('Norm of error ',e11.4,',iterations ',i5)
  110 format('Norm of error < 1.e-12,iterations ',i5)

!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.

      if (use_random) PetscCallA(PetscRandomDestroy(rctx,ierr))
      PetscCallA(KSPDestroy(ksp,ierr))
      PetscCallA(VecDestroy(u,ierr))
      PetscCallA(VecDestroy(x,ierr))
      PetscCallA(VecDestroy(b,ierr))
      PetscCallA(MatDestroy(A,ierr))

      PetscCallA(PetscFinalize(ierr))
      end

!
!/*TEST
!
!   build:
!      requires: complex
!
!   test:
!      args: -n 6 -norandom -pc_type none -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always
!      output_file: output/ex11f_1.out
!
!TEST*/
