! begin WFM
!
! Example of using BLOPEX with PETSc from Fortran.
!
! Author: William F. Mitchell
!         National Institute of Standards and Technology
!         Gaithersburg, MD
!         william.mitchell@nist.gov
!         April 8, 2008
!         Last modified August 29, 2008
!
! The parts of this software designated by WFM were produced as part of work
! done by the U.S. Government, and is not subject to copyright in the United
! States.
!
! This is a modified version of the Fortran example ex2f.F from the PETSc
! source distribution (Version 2.3.3-p4).  For simplicity, as much of the
! original code is left intact as possible, even if it performs no useful
! function for this example; for example the right hand side vector b is not
! used in an eigenvalue problem.
!
! All modifications to the original PETSc example are marked with WFM.
!
! end WFM

!
!  Description: Solves a linear system in parallel with KSP (Fortran code).
!               Also shows how to set a user-defined monitoring routine.
!
!  Program usage: mpiexec -np <procs> ex2f [-help] [all PETSc options]
!
!/*T
!  Concepts: KSP^basic parallel example
!  Concepts: KSP^setting a user-defined monitoring routine
!  Processors: n
!T*/
!
! -----------------------------------------------------------------------

! begin WFM

! place variables that will be needed by callback functions in a module so
! that they will still be accessable when the BLOPEX C library calls the
! callback functions

      module hold_vars
      private
      public :: vec_container, evecs, A, ksp, nevec

! the PETSc include files

#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscmat.h"
#include "finclude/petscpc.h"
#include "finclude/petscksp.h"

! a type containing a PETSc Vec, so we can declare an array of them to solve
! for multiple eigenvectors

      type vec_container
         Vec :: vec
      end type vec_container

! the eigenvectors

      type(vec_container), allocatable :: evecs(:)

! the matrix is needed for operatorA and the ksp is needed for operatorT

      Mat         A 
      KSP         ksp

! the number of eigenvectors is needed by the routine that sets the inital
! guess, and the routine that receives the results

      integer :: nevec

      end module hold_vars

! end WFM

      program main
      use hold_vars ! WFM module see above
      implicit none

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                    Include files
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  This program uses CPP for preprocessing, as indicated by the use of
!  PETSc include files in the directory petsc/include/finclude.  This
!  convention enables use of the CPP preprocessor, which allows the use
!  of the #include statements that define PETSc objects and variables.
!
!  Use of the conventional Fortran include statements is also supported
!  In this case, the PETsc include files are located in the directory
!  petsc/include/foldinclude.
!         
!  Since one must be very careful to include each file no more than once
!  in a Fortran routine, application programmers must exlicitly list
!  each file needed for the various PETSc components within their
!  program (unlike the C/C++ interface).
!
!  See the Fortran section of the PETSc users manual for details.
!
!  The following include statements are required for KSP Fortran programs:
!     petscsys.h       - base PETSc routines
!     petscvec.h    - vectors
!     petscmat.h    - matrices
!     petscpc.h     - preconditioners
!     petscksp.h    - Krylov subspace methods

!  Additional include statements may be needed if using additional
!  PETSc routines in a Fortran program, e.g.,
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
!     ksp      - Krylov subspace method context
!     pc       - preconditioner context
!     x, b, u  - approx solution, right-hand-side, exact solution vectors
!     A        - matrix that defines linear system
!     its      - iterations for convergence
!     norm     - norm of error in solution
!     rctx     - random number generator context
!
!  Note that vectors are declared as PETSc "Vec" objects.  These vectors
!  are mathematical objects that contain more than just an array of
!  double precision numbers. I.e., vectors in PETSc are not just
!        double precision x(*).
!  However, local vector data can be easily accessed via VecGetArray().
!  See the Fortran section of the PETSc users manual for details.
!  
      double precision  norm
      PetscInt  i,j,II,JJ,m,n,its
      PetscInt  Istart,Iend,ione
      PetscErrorCode ierr
      PetscMPIInt     rank,size
      PetscTruth  flg
      PetscScalar v,one,neg_one
      Vec         x,b,u
! WFM A and ksp were declared in module hold_vars, so comment them out
!      Mat         A 
!      KSP         ksp
      PetscRandom rctx

!  These variables are not currently used.
!      PC          pc
!      PCType      ptype 
!      double precision tol

! WFM the eigenvalues
      double precision, allocatable :: eigenvalues(:)

! WFM the subroutines that will be passed as callback functions
      external :: opA, opB, opT, return_evec, evec_init_guess

!  Note: Any user-defined Fortran routines (such as MyKSPMonitor)
!  MUST be declared as external.

      external MyKSPMonitor,MyKSPConverged

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
! WFM solve a bigger problem so the number of eigenvalues computed is a smaller
!     fraction of the total number of eigenvalues
!      m=3
!      n=3
      m = 30
      n = 30
! end WFM
      one  = 1.0
      neg_one = -1.0
      ione    = 1
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-m',m,flg,ierr)
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-n',n,flg,ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
      call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr)

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
      call MatSetFromOptions(A,ierr)

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

!     Note: this uses the less common natural ordering that orders first
!     all the unknowns for x = h then for x = 2h etc; Hence you see JH = II +- n
!     instead of JJ = II +- m as you might expect. The more standard ordering
!     would first do all variables for y = h, then y = 2h etc.

      do 10, II=Istart,Iend-1
        v = -1.0
        i = II/n
        j = II - i*n  
        if (i.gt.0) then
          JJ = II - n
          call MatSetValues(A,ione,II,ione,JJ,v,INSERT_VALUES,ierr)
        endif
        if (i.lt.m-1) then
          JJ = II + n
          call MatSetValues(A,ione,II,ione,JJ,v,INSERT_VALUES,ierr)
        endif
        if (j.gt.0) then
          JJ = II - 1
          call MatSetValues(A,ione,II,ione,JJ,v,INSERT_VALUES,ierr)
        endif
        if (j.lt.n-1) then
          JJ = II + 1
          call MatSetValues(A,ione,II,ione,JJ,v,INSERT_VALUES,ierr)
        endif
        v = 4.0
        call  MatSetValues(A,ione,II,ione,II,v,INSERT_VALUES,ierr)
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
      call VecSetFromOptions(u,ierr)
      call VecDuplicate(u,b,ierr)
      call VecDuplicate(b,x,ierr)

! begin WFM create the eigenvectors

      nevec = 5
      allocate(evecs(nevec), eigenvalues(nevec))
      do i=1,nevec
         call VecDuplicate(b,evecs(i)%vec,ierr)
      end do

! end WFM

!  Set exact solution; then compute right-hand-side vector.
!  By default we use an exact solution of a vector with all
!  elements of 1.0;  Alternatively, using the runtime option
!  -random_sol forms a solution vector with random components.

      call PetscOptionsHasName(PETSC_NULL_CHARACTER,                    &
     &             "-random_exact_sol",flg,ierr)
      if (flg) then
         call PetscRandomCreate(PETSC_COMM_WORLD,rctx,ierr)
         call PetscRandomSetFromOptions(rctx,ierr)
         call VecSetRandom(u,rctx,ierr)
         call PetscRandomDestroy(rctx,ierr)
      else
         call VecSet(u,one,ierr)
      endif
      call MatMult(A,u,b,ierr)

!  View the exact solution vector if desired

      call PetscOptionsHasName(PETSC_NULL_CHARACTER,                    &
     &             "-view_exact_sol",flg,ierr)
      if (flg) then
         call VecView(u,PETSC_VIEWER_STDOUT_WORLD,ierr)
      endif

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!         Create the linear solver and set various options
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

!  Create linear solver context

      call KSPCreate(PETSC_COMM_WORLD,ksp,ierr)

!  Set operators. Here the matrix that defines the linear system
!  also serves as the preconditioning matrix.

      call KSPSetOperators(ksp,A,A,DIFFERENT_NONZERO_PATTERN,ierr)

!  Set linear solver defaults for this problem (optional).
!   - By extracting the KSP and PC contexts from the KSP context,
!     we can then directly directly call any KSP and PC routines
!     to set various options.
!   - The following four statements are optional; all of these
!     parameters could alternatively be specified at runtime via
!     KSPSetFromOptions(). All of these defaults can be
!     overridden at runtime, as indicated below.

!     We comment out this section of code since the Jacobi
!     preconditioner is not a good general default.

!      call KSPGetPC(ksp,pc,ierr)
!      ptype = PCJACOBI
!      call PCSetType(pc,ptype,ierr)
!      tol = 1.e-7
!      call KSPSetTolerances(ksp,tol,PETSC_DEFAULT_DOUBLE_PRECISION,
!     &     PETSC_DEFAULT_DOUBLE_PRECISION,PETSC_DEFAULT_INTEGER,ierr)

!  Set user-defined monitoring routine if desired

      call PetscOptionsHasName(PETSC_NULL_CHARACTER,'-my_ksp_monitor',  &
     &                    flg,ierr)
      if (flg) then
        call KSPMonitorSet(ksp,MyKSPMonitor,PETSC_NULL_OBJECT,          &
     &                     PETSC_NULL_FUNCTION,ierr)
      endif


!  Set runtime options, e.g.,
!      -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
!  These options will override those specified above as long as
!  KSPSetFromOptions() is called _after_ any other customization
!  routines.

      call KSPSetFromOptions(ksp,ierr)

!  Set convergence test routine if desired

      call PetscOptionsHasName(PETSC_NULL_CHARACTER,                    &
     &     '-my_ksp_convergence',flg,ierr)
      if (flg) then
        call KSPSetConvergenceTest(ksp,MyKSPConverged,                  &
     &          PETSC_NULL_OBJECT,ierr)
      endif
!
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!                      Solve the linear system
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

! begin WFM

! instead of solving the linear system ...
!      call KSPSolve(ksp,b,x,ierr)
! ... call the C interface routine to BLOPEX's lobpcg_solve
      call petsc_lobpcg_solve_c(
     &   evecs(1)%vec,      ! prototype of a vector, not used
     &   nevec,             ! number of eigenvalues to compute
     &   100,               ! maximum number of iterations
     &   its,               ! actual number of iterations
     &   1.0d-6,            ! absolute error tolerance
     &   1.0d-50,           ! relative error tolerance
     &   eigenvalues,       ! computed eigenvalues
     &   opA,               ! Fortran routine for operator A
     &   opB,               ! Fortran routine for operator B
     &   opT,               ! Fortran routine for operator T
     &   return_evec,       ! Fortran routine gets eigenvectors
     &   evec_init_guess,   ! Fortran routine for initial guess
     &   ierr)              ! error code

! end WFM

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!                     Check solution and clean up
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

!  Check the error

! begin WFM cannot check the error, so comment out
!      call VecAXPY(x,neg_one,u,ierr)
!      call VecNorm(x,NORM_2,norm,ierr)
!      call KSPGetIterationNumber(ksp,its,ierr)
!      if (rank .eq. 0) then
!        if (norm .gt. 1.e-12) then
!           write(6,100) norm,its
!        else
!           write(6,110) its
!        endif
!      endif
!  100 format('Norm of error ',e10.4,' iterations ',i5)
!  110 format('Norm of error < 1.e-12,iterations ',i5)
! end WFM

! begin WFM instead, print the eigenvalues and residuals
      if (rank .eq. 0) then
         write(6,"('BLOPEX complete in ',i3,' iterations')") its
      endif
      do i=1,nevec
         call MatMult(A,evecs(i)%vec,x,ierr)
         v = -eigenvalues(i)
         call VecAXPY(x,v,evecs(i)%vec,ierr)
         call VecNorm(x,NORM_2,norm,ierr)
         if (rank .eq. 0) then
            write(6,"('eval ',i3,' is ',e15.9,' residual ',e10.4)")
     &            i,eigenvalues(i),norm
         endif
      end do
! end WFM

!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.

      call KSPDestroy(ksp,ierr)
      call VecDestroy(u,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(b,ierr)
      call MatDestroy(A,ierr)

      deallocate(evecs,eigenvalues) ! WFM

!  Always call PetscFinalize() before exiting a program.  This routine
!    - finalizes the PETSc libraries as well as MPI
!    - provides summary and diagnostic information if certain runtime
!      options are chosen (e.g., -log_summary).  See PetscFinalize()
!      manpage for more information.

      call PetscFinalize(ierr)
      end

! --------------------------------------------------------------
!
!  MyKSPMonitor - This is a user-defined routine for monitoring
!  the KSP iterative solvers.
!
!  Input Parameters:
!    ksp   - iterative context
!    n     - iteration number
!    rnorm - 2-norm (preconditioned) residual value (may be estimated)
!    dummy - optional user-defined monitor context (unused here)
!
      subroutine MyKSPMonitor(ksp,n,rnorm,dummy,ierr)

      implicit none

#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscksp.h"

      KSP              ksp
      Vec              x
      PetscErrorCode ierr
      PetscInt n,dummy
      PetscMPIInt rank
      double precision rnorm

!  Build the solution vector

      call KSPBuildSolution(ksp,PETSC_NULL_OBJECT,x,ierr)

!  Write the solution vector and residual norm to stdout
!   - Note that the parallel viewer PETSC_VIEWER_STDOUT_WORLD
!     handles data from multiple processors so that the
!     output is not jumbled.

      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
      if (rank .eq. 0) write(6,100) n
      call VecView(x,PETSC_VIEWER_STDOUT_WORLD,ierr)
      if (rank .eq. 0) write(6,200) n,rnorm

 100  format('iteration ',i5,' solution vector:')
 200  format('iteration ',i5,' residual norm ',e10.4)
      ierr = 0
      end

! --------------------------------------------------------------
!
!  MyKSPConverged - This is a user-defined routine for testing
!  convergence of the KSP iterative solvers.
!
!  Input Parameters:
!    ksp   - iterative context
!    n     - iteration number
!    rnorm - 2-norm (preconditioned) residual value (may be estimated)
!    dummy - optional user-defined monitor context (unused here)
!
      subroutine MyKSPConverged(ksp,n,rnorm,flag,dummy,ierr)

      implicit none

#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscksp.h"

      KSP              ksp
      PetscErrorCode ierr
      PetscInt n,dummy
      KSPConvergedReason flag
      double precision rnorm

      if (rnorm .le. .05) then 
        flag = 1
      else
        flag = 0
      endif
      ierr = 0

      end

! begin WFM

! the remainder of this file contains the callback functions

!                ---------------
      subroutine evec_init_guess(evec)
!                ---------------

!----------------------------------------------------
! This routine sets the initial guess for the eigenvectors.  It is called
! once for each eigenvector, in succession.
!----------------------------------------------------

      use hold_vars
      implicit none
#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscmat.h"
#include "finclude/petscpc.h"
#include "finclude/petscksp.h"

!----------------------------------------------------
! Dummy arguments

      Vec :: evec
!----------------------------------------------------
! Local variables:

      PetscRandom rctx
      integer :: ierr
      integer, save :: count=0
!----------------------------------------------------
! Begin executable code

      if (count == 0) then
         call PetscRandomCreate(PETSC_COMM_WORLD,rctx,ierr)
         call PetscRandomSetFromOptions(rctx,ierr)
      endif
      call VecSetRandom(evec,rctx,ierr)
      count = count + 1
      if (count == nevec) then
         call PetscRandomDestroy(rctx,ierr)
         count = 0
      endif

      end subroutine evec_init_guess

!                -----------
      subroutine return_evec(evec)
!                -----------

!----------------------------------------------------
! This routine receives the eigenvectors from the lobpcg C routine.  It is
! called once for each eigenvector, in succession.
!----------------------------------------------------

      use hold_vars
      implicit none
#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscmat.h"
#include "finclude/petscpc.h"
#include "finclude/petscksp.h"

!----------------------------------------------------
! Dummy arguments

      Vec :: evec
!----------------------------------------------------
! Local variables:

      integer, save :: count=1
      integer :: ierr
!----------------------------------------------------
! Begin executable code

      call VecCopy(evec,evecs(count)%vec,ierr)
      count = count + 1

      end subroutine return_evec

!                ---
      subroutine opA(matrix,avec,bvec)
!                ---

!----------------------------------------------------
! This routine provides the callback for multiplication by the matrix A
!----------------------------------------------------

      use hold_vars
      implicit none
#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscmat.h"
#include "finclude/petscpc.h"
#include "finclude/petscksp.h"

!----------------------------------------------------
! Dummy arguments

      Mat :: matrix
      Vec :: avec, bvec
!----------------------------------------------------
! Local variables:

      integer :: jerr
!----------------------------------------------------
! Begin executable code

      call MatMult(A,avec,bvec,jerr)

      end subroutine opA

!                ---
      subroutine opB(matrix,avec,bvec)
!                ---

!----------------------------------------------------
! This routine provides the callback for multiplication by the secondary
! matrix (operator B) for lobpcg.
!----------------------------------------------------

      use hold_vars
      implicit none
#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscmat.h"
#include "finclude/petscpc.h"
#include "finclude/petscksp.h"

!----------------------------------------------------
! Dummy arguments

      Mat :: matrix
      Vec :: avec, bvec
!----------------------------------------------------
! Local variables:

      integer :: jerr
!----------------------------------------------------
! Begin executable code

! In this example, B is the identity matrix, so just copy the input
! vector to the output vector

      call VecCopy(avec,bvec,jerr)

      end subroutine opB

!                ---
      subroutine opT(matrix,avec,bvec)
!                ---

!----------------------------------------------------
! This routine provides the callback for multiplication by the approximate
! inverse of A, i.e. preconditioning (operator T), for lobpcg
!----------------------------------------------------

      use hold_vars
      implicit none
#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscmat.h"
#include "finclude/petscpc.h"
#include "finclude/petscksp.h"

!----------------------------------------------------
! Dummy arguments

      Mat :: matrix
      Vec :: avec, bvec
!----------------------------------------------------
! Local variables:

      integer :: jerr
!----------------------------------------------------
! Begin executable code

! multiply avec by approximate A^-1 (i.e. solve linear system) to get bvec

      call KSPSolve(ksp,avec,bvec,jerr)

      end subroutine opT

! end WFM
