!  Program usage: mpiexec -n 1 rosenbrock1f [-help] [all TAO options]
!
!  Description:  This example demonstrates use of the TAO package to solve an
!  unconstrained minimization problem on a single processor.  We minimize the
!  extended Rosenbrock function:
!       sum_{i=0}^{n/2-1} (alpha*(x_{2i+1}-x_{2i}^2)^2 + (1-x_{2i})^2)
!
!  The C version of this code is rosenbrock1.c
!

!

! ----------------------------------------------------------------------
!
#include "rosenbrock1f.h"

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  See additional variable declarations in the file rosenbrock1f.h

      PetscErrorCode   ierr    ! used to check for functions returning nonzeros
      Vec              x       ! solution vector
      Mat              H       ! hessian matrix
      Tao        tao     ! TAO_SOVER context
      PetscBool       flg
      PetscInt         i2,i1
      PetscMPIInt     size
      PetscReal      zero

!  Note: Any user-defined Fortran routines (such as FormGradient)
!  MUST be declared as external.

      external FormFunctionGradient,FormHessian

      zero = 0.0d0
      i2 = 2
      i1 = 1

!  Initialize TAO and PETSc
      PetscCallA(PetscInitialize(ierr))

      PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD,size,ierr))
      if (size .ne. 1) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_WRONG_MPI_SIZE,'This is a uniprocessor example only'); endif

!  Initialize problem parameters
      n     = 2
      alpha = 99.0d0

! Check for command line arguments to override defaults
      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-n',n,flg,ierr))
      PetscCallA(PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-alpha',alpha,flg,ierr))

!  Allocate vectors for the solution and gradient
      PetscCallA(VecCreateSeq(PETSC_COMM_SELF,n,x,ierr))

!  Allocate storage space for Hessian;
      PetscCallA(MatCreateSeqBAIJ(PETSC_COMM_SELF,i2,n,n,i1,PETSC_NULL_INTEGER, H,ierr))

      PetscCallA(MatSetOption(H,MAT_SYMMETRIC,PETSC_TRUE,ierr))

!  The TAO code begins here

!  Create TAO solver
      PetscCallA(TaoCreate(PETSC_COMM_SELF,tao,ierr))
      PetscCallA(TaoSetType(tao,TAOLMVM,ierr))

!  Set routines for function, gradient, and hessian evaluation
      PetscCallA(TaoSetObjectiveAndGradient(tao,PETSC_NULL_VEC,FormFunctionGradient,0,ierr))
      PetscCallA(TaoSetHessian(tao,H,H,FormHessian,0,ierr))

!  Optional: Set initial guess
      PetscCallA(VecSet(x, zero, ierr))
      PetscCallA(TaoSetSolution(tao, x, ierr))

!  Check for TAO command line options
      PetscCallA(TaoSetFromOptions(tao,ierr))

!  SOLVE THE APPLICATION
      PetscCallA(TaoSolve(tao,ierr))

!  TaoView() prints ierr about the TAO solver; the option
!      -tao_view
!  can alternatively be used to activate this at runtime.
!      PetscCallA(TaoView(tao,PETSC_VIEWER_STDOUT_SELF,ierr))

!  Free TAO data structures
      PetscCallA(TaoDestroy(tao,ierr))

!  Free PETSc data structures
      PetscCallA(VecDestroy(x,ierr))
      PetscCallA(MatDestroy(H,ierr))

      PetscCallA(PetscFinalize(ierr))
      end

! --------------------------------------------------------------------
!  FormFunctionGradient - Evaluates the function f(X) and gradient G(X)
!
!  Input Parameters:
!  tao - the Tao context
!  X   - input vector
!  dummy - not used
!
!  Output Parameters:
!  G - vector containing the newly evaluated gradient
!  f - function value

      subroutine FormFunctionGradient(tao, X, f, G, dummy, ierr)
#include "rosenbrock1f.h"

      Tao        tao
      Vec              X,G
      PetscReal        f
      PetscErrorCode   ierr
      PetscInt         dummy

      PetscReal        ff,t1,t2
      PetscInt         i,nn

! PETSc's VecGetArray acts differently in Fortran than it does in C.
! Calling VecGetArray((Vec) X, (PetscReal) x_array(0:1), (PetscOffset) x_index, ierr)
! will return an array of doubles referenced by x_array offset by x_index.
!  i.e.,  to reference the kth element of X, use x_array(k + x_index).
! Notice that by declaring the arrays with range (0:1), we are using the C 0-indexing practice.
      PetscReal        g_v(0:1),x_v(0:1)
      PetscOffset      g_i,x_i

      ierr = 0
      nn = n/2
      ff = 0

!     Get pointers to vector data
      PetscCall(VecGetArrayRead(X,x_v,x_i,ierr))
      PetscCall(VecGetArray(G,g_v,g_i,ierr))

!     Compute G(X)
      do i=0,nn-1
         t1 = x_v(x_i+2*i+1) - x_v(x_i+2*i)*x_v(x_i+2*i)
         t2 = 1.0 - x_v(x_i + 2*i)
         ff = ff + alpha*t1*t1 + t2*t2
         g_v(g_i + 2*i) = -4*alpha*t1*x_v(x_i + 2*i) - 2.0*t2
         g_v(g_i + 2*i + 1) = 2.0*alpha*t1
      enddo

!     Restore vectors
      PetscCall(VecRestoreArrayRead(X,x_v,x_i,ierr))
      PetscCall(VecRestoreArray(G,g_v,g_i,ierr))

      f = ff
      PetscCall(PetscLogFlops(15.0d0*nn,ierr))

      return
      end

!
! ---------------------------------------------------------------------
!
!  FormHessian - Evaluates Hessian matrix.
!
!  Input Parameters:
!  tao     - the Tao context
!  X       - input vector
!  dummy   - optional user-defined context, as set by SNESSetHessian()
!            (not used here)
!
!  Output Parameters:
!  H      - Hessian matrix
!  PrecH  - optionally different preconditioning matrix (not used here)
!  flag   - flag indicating matrix structure
!  ierr   - error code
!
!  Note: Providing the Hessian may not be necessary.  Only some solvers
!  require this matrix.

      subroutine FormHessian(tao,X,H,PrecH,dummy,ierr)
#include "rosenbrock1f.h"

!  Input/output variables:
      Tao        tao
      Vec              X
      Mat              H, PrecH
      PetscErrorCode   ierr
      PetscInt         dummy

      PetscReal        v(0:1,0:1)
      PetscBool assembled

! PETSc's VecGetArray acts differently in Fortran than it does in C.
! Calling VecGetArray((Vec) X, (PetscReal) x_array(0:1), (PetscOffset) x_index, ierr)
! will return an array of doubles referenced by x_array offset by x_index.
!  i.e.,  to reference the kth element of X, use x_array(k + x_index).
! Notice that by declaring the arrays with range (0:1), we are using the C 0-indexing practice.
      PetscReal        x_v(0:1)
      PetscOffset      x_i
      PetscInt         i,nn,ind(0:1),i2

      ierr = 0
      nn= n/2
      i2 = 2

!  Zero existing matrix entries
      PetscCall(MatAssembled(H,assembled,ierr))
      if (assembled .eqv. PETSC_TRUE) PetscCall(MatZeroEntries(H,ierr))

!  Get a pointer to vector data

      PetscCall(VecGetArrayRead(X,x_v,x_i,ierr))

!  Compute Hessian entries

      do i=0,nn-1
         v(1,1) = 2.0*alpha
         v(0,0) = -4.0*alpha*(x_v(x_i+2*i+1) - 3*x_v(x_i+2*i)*x_v(x_i+2*i))+2
         v(1,0) = -4.0*alpha*x_v(x_i+2*i)
         v(0,1) = v(1,0)
         ind(0) = 2*i
         ind(1) = 2*i + 1
         PetscCall(MatSetValues(H,i2,ind,i2,ind,v,INSERT_VALUES,ierr))
      enddo

!  Restore vector

      PetscCall(VecRestoreArrayRead(X,x_v,x_i,ierr))

!  Assemble matrix

      PetscCall(MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY,ierr))
      PetscCall(MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY,ierr))

      PetscCall(PetscLogFlops(9.0d0*nn,ierr))

      return
      end

!
!/*TEST
!
!   build:
!      requires: !complex
!
!   test:
!      args: -tao_smonitor -tao_type ntr -tao_gatol 1.e-5
!      requires: !single
!
!TEST*/
