!
!   Solves a linear system in parallel with KSP.  Also indicates
!   use of a user-provided preconditioner.  Input parameters include:
!      -user_defined_pc : Activate a user-defined preconditioner
!
!
!!/*T
!   Concepts: KSP^basic parallel example
!   Concepts: PC^setting a user-defined shell preconditioner
!   Processors: n
!T*/

!
!     -------------------------------------------------------------------------
!
!     Module contains diag needed by shell preconditioner
!
      module mymoduleex15f
#include <petsc/finclude/petscksp.h>
      use petscksp
      Vec    diag
      end module

      program main
      use mymoduleex15f
      implicit none

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
!     norm     - norm of solution error

      Vec              x,b,u
      Mat              A
      PC               pc
      KSP              ksp
      PetscScalar      v,one,neg_one
      PetscReal norm,tol
      PetscErrorCode ierr
      PetscInt   i,j,II,JJ,Istart
      PetscInt   Iend,m,n,i1,its,five
      PetscMPIInt rank
      PetscBool  user_defined_pc,flg

!  Note: Any user-defined Fortran routines MUST be declared as external.

      external SampleShellPCSetUp, SampleShellPCApply
      external  SampleShellPCDestroy

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif
      one     = 1.0
      neg_one = -1.0
      i1 = 1
      m       = 8
      n       = 7
      five    = 5
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-m',m,flg,ierr)
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-n',n,flg,ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!      Compute the matrix and right-hand-side vector that define
!      the linear system, Ax = b.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Create parallel matrix, specifying only its global dimensions.
!  When using MatCreate(), the matrix format can be specified at
!  runtime. Also, the parallel partitioning of the matrix is
!  determined by PETSc at runtime.

      call MatCreate(PETSC_COMM_WORLD,A,ierr)
      call MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,ierr)
      call MatSetType(A, MATAIJ,ierr)
      call MatSetFromOptions(A,ierr)
      call MatMPIAIJSetPreallocation(A,five,PETSC_NULL_INTEGER,five,PETSC_NULL_INTEGER,ierr)
      call MatSeqAIJSetPreallocation(A,five,PETSC_NULL_INTEGER,ierr)

!  Currently, all PETSc parallel matrix formats are partitioned by
!  contiguous chunks of rows across the processors.  Determine which
!  rows of the matrix are locally owned.

      call MatGetOwnershipRange(A,Istart,Iend,ierr)

!  Set matrix elements for the 2-D, five-point stencil in parallel.
!   - Each processor needs to insert only elements that it owns
!     locally (but any non-local elements will be sent to the
!     appropriate processor during matrix assembly).
!   - Always specify global row and columns of matrix entries.
!   - Note that MatSetValues() uses 0-based row and column numbers
!     in Fortran as well as in C.

      do 10, II=Istart,Iend-1
        v = -1.0
        i = II/n
        j = II - i*n
        if (i.gt.0) then
          JJ = II - n
          call MatSetValues(A,i1,II,i1,JJ,v,ADD_VALUES,ierr)
        endif
        if (i.lt.m-1) then
          JJ = II + n
          call MatSetValues(A,i1,II,i1,JJ,v,ADD_VALUES,ierr)
        endif
        if (j.gt.0) then
          JJ = II - 1
          call MatSetValues(A,i1,II,i1,JJ,v,ADD_VALUES,ierr)
        endif
        if (j.lt.n-1) then
          JJ = II + 1
          call MatSetValues(A,i1,II,i1,JJ,v,ADD_VALUES,ierr)
        endif
        v = 4.0
        call  MatSetValues(A,i1,II,i1,II,v,ADD_VALUES,ierr)
 10   continue

!  Assemble matrix, using the 2-step process:
!       MatAssemblyBegin(), MatAssemblyEnd()
!  Computations can be done while messages are in transition,
!  by placing code between these two statements.

      call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr)

!  Create parallel vectors.
!   - Here, the parallel partitioning of the vector is determined by
!     PETSc at runtime.  We could also specify the local dimensions
!     if desired -- or use the more general routine VecCreate().
!   - When solving a linear system, the vectors and matrices MUST
!     be partitioned accordingly.  PETSc automatically generates
!     appropriately partitioned matrices and vectors when MatCreate()
!     and VecCreate() are used with the same communicator.
!   - Note: We form 1 vector from scratch and then duplicate as needed.

      call VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,m*n,u,ierr)
      call VecDuplicate(u,b,ierr)
      call VecDuplicate(b,x,ierr)

!  Set exact solution; then compute right-hand-side vector.

      call VecSet(u,one,ierr)
      call MatMult(A,u,b,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!         Create the linear solver and set various options
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

      call KSPGetPC(ksp,pc,ierr)
      tol = 1.e-7
      call KSPSetTolerances(ksp,tol,PETSC_DEFAULT_REAL,PETSC_DEFAULT_REAL,PETSC_DEFAULT_INTEGER,ierr)

!
!  Set a user-defined shell preconditioner if desired
!
      call PetscOptionsHasName(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-user_defined_pc',user_defined_pc,ierr)

      if (user_defined_pc) then

!  (Required) Indicate to PETSc that we are using a shell preconditioner
         call PCSetType(pc,PCSHELL,ierr)

!  (Required) Set the user-defined routine for applying the preconditioner
         call PCShellSetApply(pc,SampleShellPCApply,ierr)

!  (Optional) Do any setup required for the preconditioner
         call PCShellSetSetUp(pc,SampleShellPCSetUp,ierr)

!  (Optional) Frees any objects we created for the preconditioner
         call PCShellSetDestroy(pc,SampleShellPCDestroy,ierr)

      else
         call PCSetType(pc,PCJACOBI,ierr)
      endif

!  Set runtime options, e.g.,
!      -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
!  These options will override those specified above as long as
!  KSPSetFromOptions() is called _after_ any other customization
!  routines.

      call KSPSetFromOptions(ksp,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                      Solve the linear system
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call KSPSolve(ksp,b,x,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                     Check solution and clean up
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Check the error

      call VecAXPY(x,neg_one,u,ierr)
      call VecNorm(x,NORM_2,norm,ierr)
      call KSPGetIterationNumber(ksp,its,ierr)

      if (rank .eq. 0) then
        if (norm .gt. 1.e-12) then
           write(6,100) norm,its
        else
           write(6,110) its
        endif
      endif
  100 format('Norm of error ',1pe11.4,' iterations ',i5)
  110 format('Norm of error < 1.e-12,iterations ',i5)

!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.

      call KSPDestroy(ksp,ierr)
      call VecDestroy(u,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(b,ierr)
      call MatDestroy(A,ierr)

!  Always call PetscFinalize() before exiting a program.

      call PetscFinalize(ierr)
      end

!/***********************************************************************/
!/*          Routines for a user-defined shell preconditioner           */
!/***********************************************************************/

!
!   SampleShellPCSetUp - This routine sets up a user-defined
!   preconditioner context.
!
!   Input Parameters:
!   pc - preconditioner object
!
!   Output Parameter:
!   ierr  - error code (nonzero if error has been detected)
!
!   Notes:
!   In this example, we define the shell preconditioner to be Jacobi
!   method.  Thus, here we create a work vector for storing the reciprocal
!   of the diagonal of the preconditioner matrix; this vector is then
!   used within the routine SampleShellPCApply().
!
      subroutine SampleShellPCSetUp(pc,ierr)
      use mymoduleex15f
      use petscksp
      implicit none

      PC      pc
      Mat     pmat
      PetscErrorCode ierr

      call PCGetOperators(pc,PETSC_NULL_MAT,pmat,ierr)
      call MatCreateVecs(pmat,diag,PETSC_NULL_VEC,ierr)
      call MatGetDiagonal(pmat,diag,ierr)
      call VecReciprocal(diag,ierr)

      end

! -------------------------------------------------------------------
!
!   SampleShellPCApply - This routine demonstrates the use of a
!   user-provided preconditioner.
!
!   Input Parameters:
!   pc - preconditioner object
!   x - input vector
!
!   Output Parameters:
!   y - preconditioned vector
!   ierr  - error code (nonzero if error has been detected)
!
!   Notes:
!   This code implements the Jacobi preconditioner, merely as an
!   example of working with a PCSHELL.  Note that the Jacobi method
!   is already provided within PETSc.
!
      subroutine SampleShellPCApply(pc,x,y,ierr)
      use mymoduleex15f
      implicit none

      PC      pc
      Vec     x,y
      PetscErrorCode ierr

      call VecPointwiseMult(y,x,diag,ierr)

      end

!/***********************************************************************/
!/*          Routines for a user-defined shell preconditioner           */
!/***********************************************************************/

!
!   SampleShellPCDestroy - This routine destroys (frees the memory of) any
!      objects we made for the preconditioner
!
!   Input Parameters:
!   pc - for this example we use the actual PC as our shell context
!
!   Output Parameter:
!   ierr  - error code (nonzero if error has been detected)
!

      subroutine SampleShellPCDestroy(pc,ierr)
      use mymoduleex15f
      implicit none

      PC      pc
      PetscErrorCode ierr

!  Normally we would recommend storing all the work data (like diag) in
!  the context set with PCShellSetContext()

      call VecDestroy(diag,ierr)

      end

!
!/*TEST
!
!   test:
!      nsize: 2
!      args: -ksp_view -user_defined_pc -ksp_gmres_cgs_refinement_type refine_always
!
!TEST*/
