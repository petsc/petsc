!  Program usage: mpiexec -n 1 chwirut1f [-help] [all TAO options]
!
!  Description:  This example demonstrates use of the TAO package to solve a
!  nonlinear least-squares problem on a single processor.  We minimize the
!  Chwirut function:
!       sum_{i=0}^{n/2-1} ( alpha*(x_{2i+1}-x_{2i}^2)^2 + (1-x_{2i})^2)
!
!  The C version of this code is test_chwirut1.c
!
!!/*T
!  Concepts: TAO^Solving an unconstrained minimization problem
!  Routines: TaoCreate();
!  Routines: TaoSetType();
!  Routines: TaoSetSolution();
!  Routines: TaoSetResidualRoutine();
!  Routines: TaoSetFromOptions();
!  Routines: TaoSolve();
!  Routines: TaoDestroy();
!  Processors: 1
!T*/

!
! ----------------------------------------------------------------------
!
#include "chwirut1f.h"

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  See additional variable declarations in the file chwirut1f.h

      PetscErrorCode   ierr    ! used to check for functions returning nonzeros
      Vec              x       ! solution vector
      Vec              f       ! vector of functions
      Tao        tao     ! Tao context
      PetscInt         nhist
      PetscMPIInt  size,rank    ! number of processes running
      PetscReal      hist(100) ! objective value history
      PetscReal      resid(100)! residual history
      PetscReal      cnorm(100)! cnorm history
      PetscInt      lits(100)   ! #ksp history
      PetscInt oh
      TaoConvergedReason reason

!  Note: Any user-defined Fortran routines (such as FormGradient)
!  MUST be declared as external.

      external FormFunction

!  Initialize TAO and PETSc
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
         print*,'Unable to initialize PETSc'
         stop
      endif

      call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
      if (size .ne. 1) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_WRONG_MPI_SIZE,'This is a uniprocessor example only '); endif

!  Initialize problem parameters
      m = 214
      n = 3

!  Allocate vectors for the solution and gradient
      call VecCreateSeq(PETSC_COMM_SELF,n,x,ierr)
      call VecCreateSeq(PETSC_COMM_SELF,m,f,ierr)

!  The TAO code begins here

!  Create TAO solver
      call TaoCreate(PETSC_COMM_SELF,tao,ierr);CHKERRA(ierr)
      call TaoSetType(tao,TAOPOUNDERS,ierr);CHKERRA(ierr)
!  Set routines for function, gradient, and hessian evaluation

      call TaoSetResidualRoutine(tao,f,                       &
     &      FormFunction,0,ierr)
      CHKERRA(ierr)

!  Optional: Set initial guess
      call InitializeData()
      call FormStartingPoint(x)
      call TaoSetSolution(tao, x, ierr)
      CHKERRA(ierr)

!  Check for TAO command line options
      call TaoSetFromOptions(tao,ierr)
      CHKERRA(ierr)
      oh = 100
      call TaoSetConvergenceHistory(tao,hist,resid,cnorm,lits,          &
     &     oh,PETSC_TRUE,ierr)
      CHKERRA(ierr)
!  SOLVE THE APPLICATION
      call TaoSolve(tao,ierr)
      CHKERRA(ierr)
      call TaoGetConvergenceHistory(tao,nhist,ierr)
      CHKERRA(ierr)
      call TaoGetConvergedReason(tao, reason, ierr)
      if (reason .le. 0) then
         print *,'Tao failed.'
         print *,'Try a different TAO method, adjust some parameters,'
         print *,'or check the function evaluation routines.'
      endif

!  Free TAO data structures
      call TaoDestroy(tao,ierr)

!  Free PETSc data structures
      call VecDestroy(x,ierr)
      call VecDestroy(f,ierr)

      call PetscFinalize(ierr)

      end

! --------------------------------------------------------------------
!  FormFunction - Evaluates the function f(X) and gradient G(X)
!
!  Input Parameters:
!  tao - the Tao context
!  X   - input vector
!  dummy - not used
!
!  Output Parameters:
!  f - function vector

      subroutine FormFunction(tao, x, f, dummy, ierr)
#include "chwirut1f.h"

      Tao        tao
      Vec              x,f
      PetscErrorCode   ierr
      PetscInt         dummy

      PetscInt         i
      PetscScalar, pointer, dimension(:)  :: x_v,f_v

      ierr = 0

!     Get pointers to vector data
      call VecGetArrayF90(x,x_v,ierr);CHKERRQ(ierr)
      call VecGetArrayF90(f,f_v,ierr);CHKERRQ(ierr)

!     Compute F(X)
      do i=0,m-1
         f_v(i+1) = y(i) - exp(-x_v(1)*t(i))/(x_v(2) + x_v(3)*t(i))
      enddo

!     Restore vectors
      call VecRestoreArrayF90(X,x_v,ierr);CHKERRQ(ierr)
      call VecRestoreArrayF90(F,f_v,ierr);CHKERRQ(ierr)

      return
      end

      subroutine FormStartingPoint(x)
#include "chwirut1f.h"

      Vec             x
      PetscScalar, pointer, dimension(:)  :: x_v
      PetscErrorCode  ierr

      call VecGetArrayF90(x,x_v,ierr)
      x_v(1) = 0.15
      x_v(2) = 0.008
      x_v(3) = 0.01
      call VecRestoreArrayF90(x,x_v,ierr)
      return
      end

      subroutine InitializeData()
#include "chwirut1f.h"

      integer i
      i=0
      y(i) =    92.9000;  t(i) =  0.5000; i=i+1
      y(i) =    78.7000;  t(i) =   0.6250; i=i+1
      y(i) =    64.2000;  t(i) =   0.7500; i=i+1
      y(i) =    64.9000;  t(i) =   0.8750; i=i+1
      y(i) =    57.1000;  t(i) =   1.0000; i=i+1
      y(i) =    43.3000;  t(i) =   1.2500; i=i+1
      y(i) =    31.1000;  t(i) =  1.7500; i=i+1
      y(i) =    23.6000;  t(i) =  2.2500; i=i+1
      y(i) =    31.0500;  t(i) =  1.7500; i=i+1
      y(i) =    23.7750;  t(i) =  2.2500; i=i+1
      y(i) =    17.7375;  t(i) =  2.7500; i=i+1
      y(i) =    13.8000;  t(i) =  3.2500; i=i+1
      y(i) =    11.5875;  t(i) =  3.7500; i=i+1
      y(i) =     9.4125;  t(i) =  4.2500; i=i+1
      y(i) =     7.7250;  t(i) =  4.7500; i=i+1
      y(i) =     7.3500;  t(i) =  5.2500; i=i+1
      y(i) =     8.0250;  t(i) =  5.7500; i=i+1
      y(i) =    90.6000;  t(i) =  0.5000; i=i+1
      y(i) =    76.9000;  t(i) =  0.6250; i=i+1
      y(i) =    71.6000;  t(i) = 0.7500; i=i+1
      y(i) =    63.6000;  t(i) =  0.8750; i=i+1
      y(i) =    54.0000;  t(i) =  1.0000; i=i+1
      y(i) =    39.2000;  t(i) =  1.2500; i=i+1
      y(i) =    29.3000;  t(i) = 1.7500; i=i+1
      y(i) =    21.4000;  t(i) =  2.2500; i=i+1
      y(i) =    29.1750;  t(i) =  1.7500; i=i+1
      y(i) =    22.1250;  t(i) =  2.2500; i=i+1
      y(i) =    17.5125;  t(i) =  2.7500; i=i+1
      y(i) =    14.2500;  t(i) =  3.2500; i=i+1
      y(i) =     9.4500;  t(i) =  3.7500; i=i+1
      y(i) =     9.1500;  t(i) =  4.2500; i=i+1
      y(i) =     7.9125;  t(i) =  4.7500; i=i+1
      y(i) =     8.4750;  t(i) =  5.2500; i=i+1
      y(i) =     6.1125;  t(i) =  5.7500; i=i+1
      y(i) =    80.0000;  t(i) =  0.5000; i=i+1
      y(i) =    79.0000;  t(i) =  0.6250; i=i+1
      y(i) =    63.8000;  t(i) =  0.7500; i=i+1
      y(i) =    57.2000;  t(i) =  0.8750; i=i+1
      y(i) =    53.2000;  t(i) =  1.0000; i=i+1
      y(i) =    42.5000;  t(i) =  1.2500; i=i+1
      y(i) =    26.8000;  t(i) =  1.7500; i=i+1
      y(i) =    20.4000;  t(i) =  2.2500; i=i+1
      y(i) =    26.8500;  t(i) =   1.7500; i=i+1
      y(i) =    21.0000;  t(i) =   2.2500; i=i+1
      y(i) =    16.4625;  t(i) =   2.7500; i=i+1
      y(i) =    12.5250;  t(i) =   3.2500; i=i+1
      y(i) =    10.5375;  t(i) =   3.7500; i=i+1
      y(i) =     8.5875;  t(i) =   4.2500; i=i+1
      y(i) =     7.1250;  t(i) =   4.7500; i=i+1
      y(i) =     6.1125;  t(i) =   5.2500; i=i+1
      y(i) =     5.9625;  t(i) =   5.7500; i=i+1
      y(i) =    74.1000;  t(i) =   0.5000; i=i+1
      y(i) =    67.3000;  t(i) =   0.6250; i=i+1
      y(i) =    60.8000;  t(i) =   0.7500; i=i+1
      y(i) =    55.5000;  t(i) =   0.8750; i=i+1
      y(i) =    50.3000;  t(i) =   1.0000; i=i+1
      y(i) =    41.0000;  t(i) =   1.2500; i=i+1
      y(i) =    29.4000;  t(i) =   1.7500; i=i+1
      y(i) =    20.4000;  t(i) =   2.2500; i=i+1
      y(i) =    29.3625;  t(i) =   1.7500; i=i+1
      y(i) =    21.1500;  t(i) =   2.2500; i=i+1
      y(i) =    16.7625;  t(i) =   2.7500; i=i+1
      y(i) =    13.2000;  t(i) =   3.2500; i=i+1
      y(i) =    10.8750;  t(i) =   3.7500; i=i+1
      y(i) =     8.1750;  t(i) =   4.2500; i=i+1
      y(i) =     7.3500;  t(i) =   4.7500; i=i+1
      y(i) =     5.9625;  t(i) =  5.2500; i=i+1
      y(i) =     5.6250;  t(i) =   5.7500; i=i+1
      y(i) =    81.5000;  t(i) =    .5000; i=i+1
      y(i) =    62.4000;  t(i) =    .7500; i=i+1
      y(i) =    32.5000;  t(i) =   1.5000; i=i+1
      y(i) =    12.4100;  t(i) =   3.0000; i=i+1
      y(i) =    13.1200;  t(i) =   3.0000; i=i+1
      y(i) =    15.5600;  t(i) =   3.0000; i=i+1
      y(i) =     5.6300;  t(i) =   6.0000; i=i+1
      y(i) =    78.0000;  t(i) =   .5000; i=i+1
      y(i) =    59.9000;  t(i) =    .7500; i=i+1
      y(i) =    33.2000;  t(i) =   1.5000; i=i+1
      y(i) =    13.8400;  t(i) =   3.0000; i=i+1
      y(i) =    12.7500;  t(i) =   3.0000; i=i+1
      y(i) =    14.6200;  t(i) =   3.0000; i=i+1
      y(i) =     3.9400;  t(i) =   6.0000; i=i+1
      y(i) =    76.8000;  t(i) =    .5000; i=i+1
      y(i) =    61.0000;  t(i) =    .7500; i=i+1
      y(i) =    32.9000;  t(i) =   1.5000; i=i+1
      y(i) =    13.8700;  t(i) = 3.0000; i=i+1
      y(i) =    11.8100;  t(i) =   3.0000; i=i+1
      y(i) =    13.3100;  t(i) =   3.0000; i=i+1
      y(i) =     5.4400;  t(i) =   6.0000; i=i+1
      y(i) =    78.0000;  t(i) =    .5000; i=i+1
      y(i) =    63.5000;  t(i) =    .7500; i=i+1
      y(i) =    33.8000;  t(i) =   1.5000; i=i+1
      y(i) =    12.5600;  t(i) =   3.0000; i=i+1
      y(i) =     5.6300;  t(i) =   6.0000; i=i+1
      y(i) =    12.7500;  t(i) =   3.0000; i=i+1
      y(i) =    13.1200;  t(i) =   3.0000; i=i+1
      y(i) =     5.4400;  t(i) =   6.0000; i=i+1
      y(i) =    76.8000;  t(i) =    .5000; i=i+1
      y(i) =    60.0000;  t(i) =    .7500; i=i+1
      y(i) =    47.8000;  t(i) =   1.0000; i=i+1
      y(i) =    32.0000;  t(i) =   1.5000; i=i+1
      y(i) =    22.2000;  t(i) =   2.0000; i=i+1
      y(i) =    22.5700;  t(i) =   2.0000; i=i+1
      y(i) =    18.8200;  t(i) =   2.5000; i=i+1
      y(i) =    13.9500;  t(i) =   3.0000; i=i+1
      y(i) =    11.2500;  t(i) =   4.0000; i=i+1
      y(i) =     9.0000;  t(i) =   5.0000; i=i+1
      y(i) =     6.6700;  t(i) =   6.0000; i=i+1
      y(i) =    75.8000;  t(i) =    .5000; i=i+1
      y(i) =    62.0000;  t(i) =    .7500; i=i+1
      y(i) =    48.8000;  t(i) =   1.0000; i=i+1
      y(i) =    35.2000;  t(i) =   1.5000; i=i+1
      y(i) =    20.0000;  t(i) =   2.0000; i=i+1
      y(i) =    20.3200;  t(i) =   2.0000; i=i+1
      y(i) =    19.3100;  t(i) =   2.5000; i=i+1
      y(i) =    12.7500;  t(i) =   3.0000; i=i+1
      y(i) =    10.4200;  t(i) =   4.0000; i=i+1
      y(i) =     7.3100;  t(i) =   5.0000; i=i+1
      y(i) =     7.4200;  t(i) =   6.0000; i=i+1
      y(i) =    70.5000;  t(i) =    .5000; i=i+1
      y(i) =    59.5000;  t(i) =    .7500; i=i+1
      y(i) =    48.5000;  t(i) =   1.0000; i=i+1
      y(i) =    35.8000;  t(i) =   1.5000; i=i+1
      y(i) =    21.0000;  t(i) =   2.0000; i=i+1
      y(i) =    21.6700;  t(i) =   2.0000; i=i+1
      y(i) =    21.0000;  t(i) =   2.5000; i=i+1
      y(i) =    15.6400;  t(i) =   3.0000; i=i+1
      y(i) =     8.1700;  t(i) =   4.0000; i=i+1
      y(i) =     8.5500;  t(i) =   5.0000; i=i+1
      y(i) =    10.1200;  t(i) =   6.0000; i=i+1
      y(i) =    78.0000;  t(i) =    .5000; i=i+1
      y(i) =    66.0000;  t(i) =    .6250; i=i+1
      y(i) =    62.0000;  t(i) =    .7500; i=i+1
      y(i) =    58.0000;  t(i) =    .8750; i=i+1
      y(i) =    47.7000;  t(i) =   1.0000; i=i+1
      y(i) =    37.8000;  t(i) =   1.2500; i=i+1
      y(i) =    20.2000;  t(i) =   2.2500; i=i+1
      y(i) =    21.0700;  t(i) =   2.2500; i=i+1
      y(i) =    13.8700;  t(i) =   2.7500; i=i+1
      y(i) =     9.6700;  t(i) =   3.2500; i=i+1
      y(i) =     7.7600;  t(i) =   3.7500; i=i+1
      y(i) =     5.4400;  t(i) =  4.2500; i=i+1
      y(i) =     4.8700;  t(i) =  4.7500; i=i+1
      y(i) =     4.0100;  t(i) =   5.2500; i=i+1
      y(i) =     3.7500;  t(i) =   5.7500; i=i+1
      y(i) =    24.1900;  t(i) =   3.0000; i=i+1
      y(i) =    25.7600;  t(i) =   3.0000; i=i+1
      y(i) =    18.0700;  t(i) =   3.0000; i=i+1
      y(i) =    11.8100;  t(i) =   3.0000; i=i+1
      y(i) =    12.0700;  t(i) =   3.0000; i=i+1
      y(i) =    16.1200;  t(i) =   3.0000; i=i+1
      y(i) =    70.8000;  t(i) =    .5000; i=i+1
      y(i) =    54.7000;  t(i) =    .7500; i=i+1
      y(i) =    48.0000;  t(i) =   1.0000; i=i+1
      y(i) =    39.8000;  t(i) =   1.5000; i=i+1
      y(i) =    29.8000;  t(i) =   2.0000; i=i+1
      y(i) =    23.7000;  t(i) =   2.5000; i=i+1
      y(i) =    29.6200;  t(i) =   2.0000; i=i+1
      y(i) =    23.8100;  t(i) =   2.5000; i=i+1
      y(i) =    17.7000;  t(i) =   3.0000; i=i+1
      y(i) =    11.5500;  t(i) =   4.0000; i=i+1
      y(i) =    12.0700;  t(i) =   5.0000; i=i+1
      y(i) =     8.7400;  t(i) =   6.0000; i=i+1
      y(i) =    80.7000;  t(i) =    .5000; i=i+1
      y(i) =    61.3000;  t(i) =    .7500; i=i+1
      y(i) =    47.5000;  t(i) =   1.0000; i=i+1
      y(i) =    29.0000;  t(i) =   1.5000; i=i+1
      y(i) =    24.0000;  t(i) =   2.0000; i=i+1
      y(i) =    17.7000;  t(i) =   2.5000; i=i+1
      y(i) =    24.5600;  t(i) =   2.0000; i=i+1
      y(i) =    18.6700;  t(i) =   2.5000; i=i+1
      y(i) =    16.2400;  t(i) =   3.0000; i=i+1
      y(i) =     8.7400;  t(i) =   4.0000; i=i+1
      y(i) =     7.8700;  t(i) =   5.0000; i=i+1
      y(i) =     8.5100;  t(i) =   6.0000; i=i+1
      y(i) =    66.7000;  t(i) =    .5000; i=i+1
      y(i) =    59.2000;  t(i) =    .7500; i=i+1
      y(i) =    40.8000;  t(i) =   1.0000; i=i+1
      y(i) =    30.7000;  t(i) =   1.5000; i=i+1
      y(i) =    25.7000;  t(i) =   2.0000; i=i+1
      y(i) =    16.3000;  t(i) =   2.5000; i=i+1
      y(i) =    25.9900;  t(i) =   2.0000; i=i+1
      y(i) =    16.9500;  t(i) =   2.5000; i=i+1
      y(i) =    13.3500;  t(i) =   3.0000; i=i+1
      y(i) =     8.6200;  t(i) =   4.0000; i=i+1
      y(i) =     7.2000;  t(i) =   5.0000; i=i+1
      y(i) =     6.6400;  t(i) =   6.0000; i=i+1
      y(i) =    13.6900;  t(i) =   3.0000; i=i+1
      y(i) =    81.0000;  t(i) =    .5000; i=i+1
      y(i) =    64.5000;  t(i) =    .7500; i=i+1
      y(i) =    35.5000;  t(i) =   1.5000; i=i+1
      y(i) =    13.3100;  t(i) =   3.0000; i=i+1
      y(i) =     4.8700;  t(i) =   6.0000; i=i+1
      y(i) =    12.9400;  t(i) =   3.0000; i=i+1
      y(i) =     5.0600;  t(i) =   6.0000; i=i+1
      y(i) =    15.1900;  t(i) =   3.0000; i=i+1
      y(i) =    14.6200;  t(i) =   3.0000; i=i+1
      y(i) =    15.6400;  t(i) =   3.0000; i=i+1
      y(i) =    25.5000;  t(i) =   1.7500; i=i+1
      y(i) =    25.9500;  t(i) =   1.7500; i=i+1
      y(i) =    81.7000;  t(i) =    .5000; i=i+1
      y(i) =    61.6000;  t(i) =    .7500; i=i+1
      y(i) =    29.8000;  t(i) =   1.7500; i=i+1
      y(i) =    29.8100;  t(i) =   1.7500; i=i+1
      y(i) =    17.1700;  t(i) =   2.7500; i=i+1
      y(i) =    10.3900;  t(i) =   3.7500; i=i+1
      y(i) =    28.4000;  t(i) =   1.7500; i=i+1
      y(i) =    28.6900;  t(i) =   1.7500; i=i+1
      y(i) =    81.3000;  t(i) =    .5000; i=i+1
      y(i) =    60.9000;  t(i) =    .7500; i=i+1
      y(i) =    16.6500;  t(i) =   2.7500; i=i+1
      y(i) =    10.0500;  t(i) =   3.7500; i=i+1
      y(i) =    28.9000;  t(i) =   1.7500; i=i+1
      y(i) =    28.9500;  t(i) =   1.7500; i=i+1

      return
      end

!/*TEST
!
!   build:
!      requires: !complex
!
!   test:
!      args: -tao_smonitor -tao_max_it 100 -tao_type pounders -tao_gatol 1.e-5
!      requires: !single
!
!TEST*/
