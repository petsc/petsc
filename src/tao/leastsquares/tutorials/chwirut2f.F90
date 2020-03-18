!  Program usage: mpiexec -n 1 chwirut1f [-help] [all TAO options]
!
!  Description:  This example demonstrates use of the TAO package to solve a
!  nonlinear least-squares problem on a single processor.  We minimize the
!  Chwirut function:
!       sum_{i=0}^{n/2-1} ( alpha*(x_{2i+1}-x_{2i}^2)^2 + (1-x_{2i})^2 )
!
!  The C version of this code is chwirut1.c
!
!!/*T
!  Concepts: TAO^Solving an unconstrained minimization problem
!  Routines: TaoCreate();
!  Routines: TaoSetType();
!  Routines: TaoSetResidualRoutine();
!  Routines: TaoSetInitialVector();
!  Routines: TaoSetFromOptions();
!  Routines: TaoSolve();
!  Routines: TaoDestroy();
!  Processors: n
!T*/


!
! ----------------------------------------------------------------------
!
#include "chwirut2f.h"

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
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
         print*,'Unable to initialize PETSc'
         stop
      endif

      call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr)
      CHKERRA(ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
      CHKERRA(ierr)

!  Initialize problem parameters
      call InitializeData()

      if (rank .eq. 0) then
!  Allocate vectors for the solution and gradient
         call VecCreateSeq(PETSC_COMM_SELF,n,x,ierr)
         CHKERRA(ierr)
         call VecCreateSeq(PETSC_COMM_SELF,m,f,ierr)
         CHKERRA(ierr)


!     The TAO code begins here

!     Create TAO solver
         call TaoCreate(PETSC_COMM_SELF,tao,ierr)
         CHKERRA(ierr)
         call TaoSetType(tao,TAOPOUNDERS,ierr)
         CHKERRA(ierr)

!     Set routines for function, gradient, and hessian evaluation
         call TaoSetResidualRoutine(tao,f,                    &
     &        FormFunction,0,ierr)
         CHKERRA(ierr)

!     Optional: Set initial guess
         call FormStartingPoint(x)
         call TaoSetInitialVector(tao, x, ierr)
         CHKERRA(ierr)


!     Check for TAO command line options
         call TaoSetFromOptions(tao,ierr)
         CHKERRA(ierr)
!     SOLVE THE APPLICATION
         call TaoSolve(tao,ierr)
         CHKERRA(ierr)

!     Free TAO data structures
         call TaoDestroy(tao,ierr)
         CHKERRA(ierr)

!     Free PETSc data structures
         call VecDestroy(x,ierr)
         CHKERRA(ierr)
         call VecDestroy(f,ierr)
         CHKERRA(ierr)
         call StopWorkers(ierr)
         CHKERRA(ierr)

      else
         call TaskWorker(ierr)
         CHKERRA(ierr)
      endif

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
#include "chwirut2f.h"

      Tao        tao
      Vec              x,f
      PetscErrorCode   ierr

      PetscInt         i,checkedin
      PetscInt         finished_tasks
      integer          next_task
      PetscMPIInt      status(MPI_STATUS_SIZE),tag,source
      PetscInt         dummy

! PETSc's VecGetArray acts differently in Fortran than it does in C.
! Calling VecGetArray((Vec) X, (PetscReal) x_array(0:1), (PetscOffset) x_index, ierr)
! will return an array of doubles referenced by x_array offset by x_index.
!  i.e.,  to reference the kth element of X, use x_array(k + x_index).
! Notice that by declaring the arrays with range (0:1), we are using the C 0-indexing practice.
      PetscReal        f_v(0:1),x_v(0:1),fval
      PetscOffset      f_i,x_i

      ierr = 0

!     Get pointers to vector data
      call VecGetArray(x,x_v,x_i,ierr)
      CHKERRQ(ierr)
      call VecGetArray(f,f_v,f_i,ierr)
      CHKERRQ(ierr)


!     Compute F(X)
      if (size .eq. 1) then
         ! Single processor
         do i=0,m-1
            call RunSimulation(x_v(x_i),i,f_v(i+f_i),ierr)
         enddo
      else
         ! Multiprocessor master
         next_task = 0
         finished_tasks = 0
         checkedin = 0

         do while (finished_tasks .lt. m .or. checkedin .lt. size-1)
            call MPI_Recv(fval,1,MPIU_SCALAR,MPI_ANY_SOURCE,               &
     &           MPI_ANY_TAG,PETSC_COMM_WORLD,status,ierr)
            tag = status(MPI_TAG)
            source = status(MPI_SOURCE)
            if (tag .eq. IDLE_TAG) then
               checkedin = checkedin + 1
            else
               f_v(f_i+tag) = fval
               finished_tasks = finished_tasks + 1
            endif
            if (next_task .lt. m) then
               ! Send task to worker
               call MPI_Send(x_v(x_i),n,MPIU_SCALAR,source,next_task,             &
     &              PETSC_COMM_WORLD,ierr)
               next_task = next_task + 1
            else
               ! Send idle message to worker
               call MPI_Send(x_v(x_i),n,MPIU_SCALAR,source,IDLE_TAG,              &
     &              PETSC_COMM_WORLD,ierr)
            end if
         enddo
      endif

!     Restore vectors
      call VecRestoreArray(x,x_v,x_i,ierr)
      CHKERRQ(ierr)
      call VecRestoreArray(F,f_v,f_i,ierr)
      CHKERRQ(ierr)
      return
      end

      subroutine FormStartingPoint(x)
#include "chwirut2f.h"

      Vec             x
      PetscReal       x_v(0:1)
      PetscOffset     x_i
      PetscErrorCode  ierr

      call VecGetArray(x,x_v,x_i,ierr)
      CHKERRQ(ierr)
      x_v(x_i) = 0.15
      x_v(x_i+1) = 0.008
      x_v(x_i+2) = 0.01
      call VecRestoreArray(x,x_v,x_i,ierr)
      CHKERRQ(ierr)
      return
      end


      subroutine InitializeData()
#include "chwirut2f.h"

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
#include "chwirut2f.h"

      PetscErrorCode ierr
      PetscReal x(n),f
      PetscMPIInt tag
      PetscInt index
      PetscMPIInt status(MPI_STATUS_SIZE)

      tag = IDLE_TAG
      f   = 0.0
      ! Send check-in message to master
      call MPI_Send(f,1,MPIU_SCALAR,0,IDLE_TAG,PETSC_COMM_WORLD,ierr)
      CHKERRQ(ierr)
      do while (tag .ne. DIE_TAG)
         call MPI_Recv(x,n,MPIU_SCALAR,0,MPI_ANY_TAG,PETSC_COMM_WORLD,     &
     &        status,ierr)
         CHKERRQ(ierr)
         tag = status(MPI_TAG)
         if (tag .eq. IDLE_TAG) then
            call MPI_Send(f,1,MPIU_SCALAR,0,IDLE_TAG,PETSC_COMM_WORLD,     &
     &           ierr)
            CHKERRQ(ierr)
         else if (tag .ne. DIE_TAG) then
            index = tag
            ! Compute local part of residual
            call RunSimulation(x,index,f,ierr)
            CHKERRQ(ierr)

            ! Return residual to master
            call MPI_Send(f,1,MPIU_SCALAR,0,tag,PETSC_COMM_WORLD,ierr)
            CHKERRQ(ierr)
         end if
      enddo
      ierr = 0
      return
      end



      subroutine RunSimulation(x,i,f,ierr)
#include "chwirut2f.h"

      PetscReal x(n),f
      PetscInt i
      PetscErrorCode ierr
      f = y(i) - exp(-x(1)*t(i))/(x(2)+x(3)*t(i))
      ierr = 0
      return
      end

      subroutine StopWorkers(ierr)
#include "chwirut2f.h"

      integer checkedin
      PetscMPIInt status(MPI_STATUS_SIZE)
      PetscMPIInt source
      PetscReal f,x(n)
      PetscErrorCode ierr
      PetscInt i

      checkedin=0
      do while (checkedin .lt. size-1)
         call MPI_Recv(f,1,MPIU_SCALAR,MPI_ANY_SOURCE,MPI_ANY_TAG,         &
     &        PETSC_COMM_WORLD,status,ierr)
         CHKERRQ(ierr)
         checkedin=checkedin+1
         source = status(MPI_SOURCE)
         do i=1,n
           x(i) = 0.0
         enddo
         call MPI_Send(x,n,MPIU_SCALAR,source,DIE_TAG,PETSC_COMM_WORLD,    &
     &        ierr)
         CHKERRQ(ierr)
      enddo
      ierr=0
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
