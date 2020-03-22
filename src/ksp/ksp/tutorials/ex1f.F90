!
!   Description: Solves a tridiagonal linear system with KSP.
!
!/*T
!   Concepts: KSP^solving a system of linear equations
!   Processors: 1
!T*/
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
!     ksp      - Krylov subspace method context
!     pc       - preconditioner context
!     x, b, u  - approx solution, right-hand-side, exact solution vectors
!     A        - matrix that defines linear system
!     its      - iterations for convergence
!     norm     - norm of error in solution
!
      Vec              x,b,u
      Mat              A
      KSP              ksp
      PC               pc
      PetscReal        norm,tol
      PetscErrorCode   ierr
      PetscInt i,n,col(3),its,i1,i2,i3
      PetscBool  flg
      PetscMPIInt size
      PetscScalar      none,one,value(3)
      PetscLogStage    stages(2);

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif
      call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr)
      if (size .ne. 1) then; SETERRA(PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,'This is a uniprocessor example only'); endif
      none = -1.0
      one  = 1.0
      n    = 10
      i1 = 1
      i2 = 2
      i3 = 3
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-n',n,flg,ierr)

      call PetscLogStageRegister("MatVec Assembly",stages(1),ierr)
      call PetscLogStageRegister("KSP Solve",stages(2),ierr)
      call PetscLogStagePush(stages(1),ierr)
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!         Compute the matrix and right-hand-side vector that define
!         the linear system, Ax = b.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Create matrix.  When using MatCreate(), the matrix format can
!  be specified at runtime.

      call MatCreate(PETSC_COMM_WORLD,A,ierr)
      call MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n,ierr)
      call MatSetFromOptions(A,ierr)
      call MatSetUp(A,ierr)

!  Assemble matrix.
!   - Note that MatSetValues() uses 0-based row and column numbers
!     in Fortran as well as in C (as set here in the array "col").

      value(1) = -1.0
      value(2) = 2.0
      value(3) = -1.0
      do 50 i=1,n-2
         col(1) = i-1
         col(2) = i
         col(3) = i+1
         call MatSetValues(A,i1,i,i3,col,value,INSERT_VALUES,ierr)
  50  continue
      i = n - 1
      col(1) = n - 2
      col(2) = n - 1
      call MatSetValues(A,i1,i,i2,col,value,INSERT_VALUES,ierr)
      i = 0
      col(1) = 0
      col(2) = 1
      value(1) = 2.0
      value(2) = -1.0
      call MatSetValues(A,i1,i,i2,col,value,INSERT_VALUES,ierr)
      call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr)

!  Create vectors.  Note that we form 1 vector from scratch and
!  then duplicate as needed.

      call VecCreate(PETSC_COMM_WORLD,x,ierr)
      call VecSetSizes(x,PETSC_DECIDE,n,ierr)
      call VecSetFromOptions(x,ierr)
      call VecDuplicate(x,b,ierr)
      call VecDuplicate(x,u,ierr)

!  Set exact solution; then compute right-hand-side vector.

      call VecSet(u,one,ierr)
      call MatMult(A,u,b,ierr)
      call PetscLogStagePop(ierr)
      call PetscLogStagePush(stages(2),ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!          Create the linear solver and set various options
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Create linear solver context

      call KSPCreate(PETSC_COMM_WORLD,ksp,ierr)

!  Set operators. Here the matrix that defines the linear system
!  also serves as the preconditioning matrix.

      call KSPSetOperators(ksp,A,A,ierr)

!  Set linear solver defaults for this problem (optional).
!   - By extracting the KSP and PC contexts from the KSP context,
!     we can then directly directly call any KSP and PC routines
!     to set various options.
!   - The following four statements are optional; all of these
!     parameters could alternatively be specified at runtime via
!     KSPSetFromOptions();

      call KSPGetPC(ksp,pc,ierr)
      call PCSetType(pc,PCJACOBI,ierr)
      tol = .0000001
      call KSPSetTolerances(ksp,tol,PETSC_DEFAULT_REAL,                         &
     &     PETSC_DEFAULT_REAL,PETSC_DEFAULT_INTEGER,ierr)

!  Set runtime options, e.g.,
!      -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
!  These options will override those specified above as long as
!  KSPSetFromOptions() is called _after_ any other customization
!  routines.

      call KSPSetFromOptions(ksp,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                      Solve the linear system
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call KSPSolve(ksp,b,x,ierr)
      call PetscLogStagePop(ierr)

!  View solver info; we could instead use the option -ksp_view

      call KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                      Check solution and clean up
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Check the error

      call VecAXPY(x,none,u,ierr)
      call VecNorm(x,NORM_2,norm,ierr)
      call KSPGetIterationNumber(ksp,its,ierr)
      if (norm .gt. 1.e-12) then
        write(6,100) norm,its
      else
        write(6,200) its
      endif
 100  format('Norm of error ',e11.4,',  Iterations = ',i5)
 200  format('Norm of error < 1.e-12, Iterations = ',i5)

!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.

      call VecDestroy(x,ierr)
      call VecDestroy(u,ierr)
      call VecDestroy(b,ierr)
      call MatDestroy(A,ierr)
      call KSPDestroy(ksp,ierr)
      call PetscFinalize(ierr)

      end

!/*TEST
!
!     test:
!       args: -ksp_monitor_short
!
!TEST*/
