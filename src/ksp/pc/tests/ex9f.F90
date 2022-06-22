!
!   Description: Tests PCFieldSplitGetIS and PCFieldSplitSetIS from Fortran.
!
      program main
#include <petsc/finclude/petscksp.h>
      use petscksp
      implicit none

      Vec              x,b,u
      Mat              A
      KSP              ksp
      PC               pc
      PetscReal        norm;
      PetscErrorCode   ierr
      PetscInt i,n,col(3),its,i1,i2,i3
      PetscInt ione,izero
      PetscBool  flg
      PetscMPIInt size
      PetscScalar      none,one,value(3)
      IS isin,isout

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      PetscCallA(PetscInitialize(ierr))
      PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD,size,ierr))
      if (size .ne. 1) then; SETERRA(PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,'This is a uniprocessor example only'); endif
      none = -1.0
      one  = 1.0
      n    = 10
      i1 = 1
      i2 = 2
      i3 = 3
      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-n',n,flg,ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!         Compute the matrix and right-hand-side vector that define
!         the linear system, Ax = b.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Create matrix.  When using MatCreate(), the matrix format can
!  be specified at runtime.

      PetscCallA(MatCreate(PETSC_COMM_WORLD,A,ierr))
      PetscCallA(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n,ierr))
      PetscCallA(MatSetFromOptions(A,ierr))
      PetscCallA(MatSetUp(A,ierr))

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
         PetscCallA(MatSetValues(A,i1,i,i3,col,value,INSERT_VALUES,ierr))
  50  continue
      i = n - 1
      col(1) = n - 2
      col(2) = n - 1
      PetscCallA(MatSetValues(A,i1,i,i2,col,value,INSERT_VALUES,ierr))
      i = 0
      col(1) = 0
      col(2) = 1
      value(1) = 2.0
      value(2) = -1.0
      PetscCallA(MatSetValues(A,i1,i,i2,col,value,INSERT_VALUES,ierr))
      PetscCallA(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr))
      PetscCallA(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr))

!  Create vectors.  Note that we form 1 vector from scratch and
!  then duplicate as needed.

      PetscCallA(VecCreate(PETSC_COMM_WORLD,x,ierr))
      PetscCallA(VecSetSizes(x,PETSC_DECIDE,n,ierr))
      PetscCallA(VecSetFromOptions(x,ierr))
      PetscCallA(VecDuplicate(x,b,ierr))
      PetscCallA(VecDuplicate(x,u,ierr))

!  Set exact solution; then compute right-hand-side vector.

      PetscCallA(VecSet(u,one,ierr))
      PetscCallA(MatMult(A,u,b,ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!          Create the linear solver and set various options
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Create linear solver context

      PetscCallA(KSPCreate(PETSC_COMM_WORLD,ksp,ierr))

!  Set operators. Here the matrix that defines the linear system
!  also serves as the preconditioning matrix.

      PetscCallA(KSPSetOperators(ksp,A,A,ierr))

!  Set linear solver defaults for this problem (optional).
!   - By extracting the KSP and PC contexts from the KSP context,
!     we can then directly call any KSP and PC routines
!     to set various options.
!   - The following four statements are optional; all of these
!     parameters could alternatively be specified at runtime via
!     KSPSetFromOptions();

      PetscCallA(KSPGetPC(ksp,pc,ierr))
      PetscCallA(PCSetType(pc,PCFIELDSPLIT,ierr))
      izero = 0
      ione  = 1
      PetscCallA(ISCreateStride(PETSC_COMM_SELF,n,izero,ione,isin,ierr))
      PetscCallA(PCFieldSplitSetIS(pc,"splitname",isin,ierr))
      PetscCallA(PCFieldSplitGetIS(pc,"splitname",isout,ierr))
      if (isin .ne. isout) then ; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PCFieldSplitGetIS() failed"); endif

!  Set runtime options, e.g.,
!      -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
!  These options will override those specified above as long as
!  KSPSetFromOptions() is called _after_ any other customization
!  routines.

      PetscCallA(KSPSetFromOptions(ksp,ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                      Solve the linear system
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      PetscCallA(KSPSolve(ksp,b,x,ierr))
      PetscCallA(PetscLogStagePop(ierr))

!  View solver info; we could instead use the option -ksp_view

      PetscCallA(KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD,ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                      Check solution and clean up
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Check the error

      PetscCallA(VecAXPY(x,none,u,ierr))
      PetscCallA(VecNorm(x,NORM_2,norm,ierr))
      PetscCallA(KSPGetIterationNumber(ksp,its,ierr))
      if (norm .gt. 1.e-12) then
        write(6,100) norm,its
      else
        write(6,200) its
      endif
 100  format('Norm of error ',e11.4,',  Iterations = ',i5)
 200  format('Norm of error < 1.e-12, Iterations = ',i5)

!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.

      PetscCallA(ISDestroy(isin,ierr))
      PetscCallA(VecDestroy(x,ierr))
      PetscCallA(VecDestroy(u,ierr))
      PetscCallA(VecDestroy(b,ierr))
      PetscCallA(MatDestroy(A,ierr))
      PetscCallA(KSPDestroy(ksp,ierr))
      PetscCallA(PetscFinalize(ierr))

      end

!/*TEST
!
!     test:
!       args: -ksp_monitor
!
!TEST*/
