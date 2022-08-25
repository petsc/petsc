!  Program usage: mpiexec -n 1 chwirut1f [-help] [all TAO options]
!
!  Description:  This example demonstrates use of the TAO package to solve a
!  nonlinear least-squares problem on a single processor.  We minimize the
!  Chwirut function:
!       sum_{i=0}^{n/2-1} ( alpha*(x_{2i+1}-x_{2i}^2)^2 + (1-x_{2i})^2)
!
!  The C version of this code is chwirut1.c
!

!
! ----------------------------------------------------------------------
!
      module chwirut2fmodule
      use petscmpi              ! or mpi or mpi_f08
      use petsctao
#include <petsc/finclude/petsctao.h>
      PetscReal t(0:213)
      PetscReal y(0:213)
      PetscInt  m,n
      PetscMPIInt  nn
      PetscMPIInt  rank
      PetscMPIInt  size
      PetscMPIInt  idle_tag, die_tag
      PetscMPIInt  zero,one
      parameter (m=214)
      parameter (n=3)
      parameter (nn=n)
      parameter (idle_tag=2000)
      parameter (die_tag=3000)
      parameter (zero=0,one=1)
      end module chwirut2fmodule

      program main
      use chwirut2fmodule

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  See additional variable declarations in the file chwirut2f.h

      PetscErrorCode   ierr    ! used to check for functions returning nonzeros
      Vec              x       ! solution vector
      Vec              f       ! vector of functions
      Tao        tao     ! Tao context

!  Note: Any user-defined Fortran routines (such as FormGradient)
!  MUST be declared as external.

      external FormFunction

!  Initialize TAO and PETSc
      PetscCallA(PetscInitialize(ierr))
      PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD,size,ierr))
      PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))

!  Initialize problem parameters
      call InitializeData()

      if (rank .eq. 0) then
!  Allocate vectors for the solution and gradient
         PetscCallA(VecCreateSeq(PETSC_COMM_SELF,n,x,ierr))
         PetscCallA(VecCreateSeq(PETSC_COMM_SELF,m,f,ierr))

!     The TAO code begins here

!     Create TAO solver
         PetscCallA(TaoCreate(PETSC_COMM_SELF,tao,ierr))
         PetscCallA(TaoSetType(tao,TAOPOUNDERS,ierr))

!     Set routines for function, gradient, and hessian evaluation
         PetscCallA(TaoSetResidualRoutine(tao,f,FormFunction,0,ierr))

!     Optional: Set initial guess
         call FormStartingPoint(x)
         PetscCallA(TaoSetSolution(tao, x, ierr))

!     Check for TAO command line options
         PetscCallA(TaoSetFromOptions(tao,ierr))
!     SOLVE THE APPLICATION
         PetscCallA(TaoSolve(tao,ierr))

!     Free TAO data structures
         PetscCallA(TaoDestroy(tao,ierr))

!     Free PETSc data structures
         PetscCallA(VecDestroy(x,ierr))
         PetscCallA(VecDestroy(f,ierr))
         PetscCallA(StopWorkers(ierr))

      else
         PetscCallA(TaskWorker(ierr))
      endif

      PetscCallA(PetscFinalize(ierr))
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
      use chwirut2fmodule

      Tao        tao
      Vec              x,f
      PetscErrorCode   ierr

      PetscInt         i,checkedin
      PetscInt         finished_tasks
      PetscMPIInt      next_task
      PetscMPIInt      status(MPI_STATUS_SIZE),tag,source
      PetscInt         dummy

! PETSc's VecGetArray acts differently in Fortran than it does in C.
! Calling VecGetArray((Vec) X, (PetscReal) x_array(0:1), (PetscOffset) x_index, ierr))
! will return an array of doubles referenced by x_array offset by x_index.
!  i.e.,  to reference the kth element of X, use x_array(k + x_index).
! Notice that by declaring the arrays with range (0:1), we are using the C 0-indexing practice.
      PetscReal        f_v(0:1),x_v(0:1),fval(1)
      PetscOffset      f_i,x_i

      ierr = 0

!     Get pointers to vector data
      PetscCall(VecGetArray(x,x_v,x_i,ierr))
      PetscCall(VecGetArray(f,f_v,f_i,ierr))

!     Compute F(X)
      if (size .eq. 1) then
         ! Single processor
         do i=0,m-1
            PetscCall(RunSimulation(x_v(x_i),i,f_v(i+f_i),ierr))
         enddo
      else
         ! Multiprocessor main
         next_task = zero
         finished_tasks = 0
         checkedin = 0

         do while (finished_tasks .lt. m .or. checkedin .lt. size-1)
            PetscCallMPI(MPI_Recv(fval,one,MPIU_SCALAR,MPI_ANY_SOURCE,MPI_ANY_TAG,PETSC_COMM_WORLD,status,ierr))
            tag = status(MPI_TAG)
            source = status(MPI_SOURCE)
            if (tag .eq. IDLE_TAG) then
               checkedin = checkedin + 1
            else
               f_v(f_i+tag) = fval(1)
               finished_tasks = finished_tasks + 1
            endif
            if (next_task .lt. m) then
               ! Send task to worker
               PetscCallMPI(MPI_Send(x_v(x_i),nn,MPIU_SCALAR,source,next_task,PETSC_COMM_WORLD,ierr))
               next_task = next_task + one
            else
               ! Send idle message to worker
               PetscCallMPI(MPI_Send(x_v(x_i),nn,MPIU_SCALAR,source,IDLE_TAG,PETSC_COMM_WORLD,ierr))
            end if
         enddo
      endif

!     Restore vectors
      PetscCall(VecRestoreArray(x,x_v,x_i,ierr))
      PetscCall(VecRestoreArray(F,f_v,f_i,ierr))
      return
      end

      subroutine FormStartingPoint(x)
      use chwirut2fmodule

      Vec             x
      PetscReal       x_v(0:1)
      PetscOffset     x_i
      PetscErrorCode  ierr

      PetscCall(VecGetArray(x,x_v,x_i,ierr))
      x_v(x_i) = 0.15
      x_v(x_i+1) = 0.008
      x_v(x_i+2) = 0.01
      PetscCall(VecRestoreArray(x,x_v,x_i,ierr))
      return
      end

      subroutine InitializeData()
      use chwirut2fmodule

      PetscInt i
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

      subroutine TaskWorker(ierr)
      use chwirut2fmodule

      PetscErrorCode ierr
      PetscReal x(n),f(1)
      PetscMPIInt tag
      PetscInt index
      PetscMPIInt status(MPI_STATUS_SIZE)

      tag = IDLE_TAG
      f   = 0.0
      ! Send check-in message to rank-0
      PetscCallMPI(MPI_Send(f,one,MPIU_SCALAR,zero,IDLE_TAG,PETSC_COMM_WORLD,ierr))
      do while (tag .ne. DIE_TAG)
         PetscCallMPI(MPI_Recv(x,nn,MPIU_SCALAR,zero,MPI_ANY_TAG,PETSC_COMM_WORLD,status,ierr))
         tag = status(MPI_TAG)
         if (tag .eq. IDLE_TAG) then
            PetscCallMPI(MPI_Send(f,one,MPIU_SCALAR,zero,IDLE_TAG,PETSC_COMM_WORLD,ierr))
         else if (tag .ne. DIE_TAG) then
            index = tag
            ! Compute local part of residual
            PetscCall(RunSimulation(x,index,f(1),ierr))

            ! Return residual to rank-0
            PetscCallMPI(MPI_Send(f,one,MPIU_SCALAR,zero,tag,PETSC_COMM_WORLD,ierr))
         end if
      enddo
      ierr = 0
      return
      end

      subroutine RunSimulation(x,i,f,ierr)
      use chwirut2fmodule

      PetscReal x(n),f
      PetscInt i
      PetscErrorCode ierr
      f = y(i) - exp(-x(1)*t(i))/(x(2)+x(3)*t(i))
      ierr = 0
      return
      end

      subroutine StopWorkers(ierr)
      use chwirut2fmodule

      integer checkedin
      PetscMPIInt status(MPI_STATUS_SIZE)
      PetscMPIInt source
      PetscReal f(1),x(n)
      PetscErrorCode ierr
      PetscInt i

      checkedin=0
      do while (checkedin .lt. size-1)
         PetscCallMPI(MPI_Recv(f,one,MPIU_SCALAR,MPI_ANY_SOURCE,MPI_ANY_TAG,PETSC_COMM_WORLD,status,ierr))
         checkedin=checkedin+1
         source = status(MPI_SOURCE)
         do i=1,n
           x(i) = 0.0
         enddo
         PetscCallMPI(MPI_Send(x,nn,MPIU_SCALAR,source,DIE_TAG,PETSC_COMM_WORLD,ierr))
      enddo
      ierr = 0
      return
      end

!/*TEST
!
!   build:
!      requires: !complex
!
!   test:
!      nsize: 3
!      args: -tao_smonitor -tao_max_it 100 -tao_type pounders -tao_gatol 1.e-5
!      requires: !single
!
!
!TEST*/
