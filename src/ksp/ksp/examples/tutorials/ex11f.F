!
!  Description: Solves a complex linear system in parallel with KSP (Fortran code).
!
!/*T
!  Concepts: KSP^solving a Helmholtz equation
!  Concepts: complex numbers
!  Processors: n
!T*/
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
      implicit none

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                    Include files
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  The following include statements are required for KSP Fortran programs:
!     petscsys.h       - base PETSc routines
!     petscvec.h    - vectors
!     petscmat.h    - matrices
!     petscpc.h     - preconditioners
!     petscksp.h    - Krylov subspace methods
!  Additional include statements may be needed if using other PETSc
!  routines in a Fortran program, e.g.,
!     petscviewer.h - viewers
!     petscis.h     - index sets
!
#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscmat.h"
#include "finclude/petscpc.h"
#include "finclude/petscksp.h"
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
      double precision norm,h2,sigma1
      PetscScalar  none,sigma2,v,pfive,czero
      PetscScalar  cone
      PetscInt dim,its,n,Istart
      PetscInt Iend,i,j,II,JJ,one
      PetscErrorCode ierr
      PetscMPIInt rank
      PetscTruth flg
      logical          use_random

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
#if !defined(PETSC_USE_COMPLEX)
      write(6,*) "This example requires complex numbers."
      goto 200
#endif

      none   = -1.0
      n      = 6
      sigma1 = 100.0
      czero  = 0.0
      cone   = PETSC_i
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
      call PetscOptionsGetReal(PETSC_NULL_CHARACTER,'-sigma1',sigma1,      &
     &                       flg,ierr)
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-n',n,flg,ierr)
      dim    = n*n

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!      Compute the matrix and right-hand-side vector that define
!      the linear system, Ax = b.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

!  Create parallel matrix, specifying only its global dimensions.
!  When using MatCreate(), the matrix format can be specified at
!  runtime. Also, the parallel partitioning of the matrix is
!  determined by PETSc at runtime.

      call MatCreate(PETSC_COMM_WORLD,A,ierr)
      call MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,dim,dim,ierr)
      call MatSetFromOptions(A,ierr)

!  Currently, all PETSc parallel matrix formats are partitioned by
!  contiguous chunks of rows across the processors.  Determine which
!  rows of the matrix are locally owned. 

      call MatGetOwnershipRange(A,Istart,Iend,ierr)

!  Set matrix elements in parallel.
!   - Each processor needs to insert only elements that it owns
!     locally (but any non-local elements will be sent to the
!     appropriate processor during matrix assembly). 
!   - Always specify global rows and columns of matrix entries.

      call PetscOptionsHasName(PETSC_NULL_CHARACTER,'-norandom',        &
     &     flg,ierr)
      if (flg) then
         use_random = .false.
         sigma2 = 10.0*PETSC_i
      else
         use_random = .true.
         call PetscRandomCreate(PETSC_COMM_WORLD,                       &
     &        rctx,ierr)
         call PetscRandomSetFromOptions(rctx,ierr)
         call PetscRandomSetInterval(rctx,czero,cone,ierr)
      endif
      h2 = 1.0/((n+1)*(n+1))

      one = 1
      do 10, II=Istart,Iend-1
        v = -1.0
        i = II/n
        j = II - i*n  
        if (i.gt.0) then
          JJ = II - n
          call MatSetValues(A,one,II,one,JJ,v,ADD_VALUES,ierr)
        endif
        if (i.lt.n-1) then
          JJ = II + n
          call MatSetValues(A,one,II,one,JJ,v,ADD_VALUES,ierr)
        endif
        if (j.gt.0) then
          JJ = II - 1
          call MatSetValues(A,one,II,one,JJ,v,ADD_VALUES,ierr)
        endif
        if (j.lt.n-1) then
          JJ = II + 1
          call MatSetValues(A,one,II,one,JJ,v,ADD_VALUES,ierr)
        endif
        if (use_random) call PetscRandomGetValue(rctx,                          &
     &                        sigma2,ierr)
        v = 4.0 - sigma1*h2 + sigma2*h2
        call  MatSetValues(A,one,II,one,II,v,ADD_VALUES,ierr)
 10   continue
      if (use_random) call PetscRandomDestroy(rctx,ierr)

!  Assemble matrix, using the 2-step process:
!       MatAssemblyBegin(), MatAssemblyEnd()
!  Computations can be done while messages are in transition
!  by placing code between these two statements.

      call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr)

!  Create parallel vectors.
!   - Here, the parallel partitioning of the vector is determined by
!     PETSc at runtime.  We could also specify the local dimensions
!     if desired.
!   - Note: We form 1 vector from scratch and then duplicate as needed.

      call VecCreate(PETSC_COMM_WORLD,u,ierr)
      call VecSetSizes(u,PETSC_DECIDE,dim,ierr)
      call VecSetFromOptions(u,ierr)
      call VecDuplicate(u,b,ierr)
      call VecDuplicate(b,x,ierr)

!  Set exact solution; then compute right-hand-side vector.

      if (use_random) then
         call PetscRandomCreate(PETSC_COMM_WORLD,rctx,ierr)
         call PetscRandomSetFromOptions(rctx,ierr)
         call VecSetRandom(u,rctx,ierr)
      else
         pfive = 0.5
         call VecSet(u,pfive,ierr)
      endif
      call MatMult(A,u,b,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!         Create the linear solver and set various options
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

!  Create linear solver context

      call KSPCreate(PETSC_COMM_WORLD,ksp,ierr)

!  Set operators. Here the matrix that defines the linear system
!  also serves as the preconditioning matrix.

      call KSPSetOperators(ksp,A,A,DIFFERENT_NONZERO_PATTERN,ierr)

!  Set runtime options, e.g.,
!      -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>

      call KSPSetFromOptions(ksp,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!                      Solve the linear system
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call KSPSolve(ksp,b,x,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!                     Check solution and clean up
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

!  Check the error

      call VecAXPY(x,none,u,ierr)
      call VecNorm(x,NORM_2,norm,ierr)
      call KSPGetIterationNumber(ksp,its,ierr)
      if (rank .eq. 0) then
        if (norm .gt. 1.e-12) then
           write(6,100) norm,its
        else
           write(6,110) its
        endif
      endif
  100 format('Norm of error ',e10.4,',iterations ',i5)
  110 format('Norm of error < 1.e-12,iterations ',i5)

!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.

      if (use_random) call PetscRandomDestroy(rctx,ierr)
      call KSPDestroy(ksp,ierr)
      call VecDestroy(u,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(b,ierr)
      call MatDestroy(A,ierr)

#if !defined(PETSC_USE_COMPLEX)
 200  continue
#endif
      call PetscFinalize(ierr)
      end
