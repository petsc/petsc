C
C    "$Id: ex15f.F,v 1.2 1997/09/11 17:01:32 balay Exp balay $";
C
C   Solves a linear system in parallel with SLES.  Also indicates
C   use of a user-provided preconditioner.  Input parameters include:
C      -user_defined_pc : Activate a user-defined preconditioner
C
C  Program usage: mpirun ex15f [-help] [all PETSc options]
C
C/*T
C   Concepts: SLES^Solving a system of linear equations (basic parallel example);
C   Concepts: PC^Setting a user-defined shell preconditioner
C   Routines: SLESCreate(); SLESSetOperators(); SLESSetFromOptions();
C   Routines: SLESSolve(); SLESGetKSP(); SLESGetPC(); KSPSetTolerances(); 
C   Routines: PCSetType(); PCShellSetApply(); 
C   Processors: n
CT*/
C
C  ------------------------------------------------------------------------- 

      program main
      implicit none

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C                    Include files
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C
C     petsc.h  - base PETSc routines      vec.h - vectors
C     sys.h    - system routines          mat.h - matrices
C     ksp.h    - Krylov subspace methods  pc.h  - preconditioners
C     sles.h   - SLES interface

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/pc.h"
#include "include/FINCLUDE/ksp.h"
#include "include/FINCLUDE/sles.h"

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C                   Variable declarations
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C
C  Variables:
C     sles     - linear solver context
C     ksp      - Krylov subspace method context
C     pc       - preconditioner context
C     x, b, u  - approx solution, right-hand-side, exact solution vectors
C     A        - matrix that defines linear system
C     its      - iterations for convergence
C     norm     - norm of solution error

      Vec              x, b, u
      Mat              A      
      SLES             sles   
      PC               pc     
      KSP              ksp    
      double precision norm, v, one, neg_one, tol
      integer          i, j, II, JJ, Istart, Iend, ierr, m, n
      integer          user_defined_pc, its, flg, rank

C  Note: Any user-defined Fortran routines MUST be declared as external.

      external SampleShellPCSetUp, SampleShellPCApply

C  Common block to store data for user-provided preconditioner 
      common /myshellpc/ diag
      Vec    diag

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C                 Beginning of program
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      one     = 1.0
      neg_one = -1.0
      m       = 8
      n       = 7
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-m',m,flg,ierr)
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-n',n,flg,ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C      Compute the matrix and right-hand-side vector that define
C      the linear system, Ax = b.
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

C  Create parallel matrix, specifying only its global dimensions.
C  When using MatCreate(), the matrix format can be specified at
C  runtime. Also, the parallel partitioning of the matrix is
C  determined by PETSc at runtime.

      call MatCreate(PETSC_COMM_WORLD,m*n,m*n,A,ierr)

C  Currently, all PETSc parallel matrix formats are partitioned by
C  contiguous chunks of rows across the processors.  Determine which
C  rows of the matrix are locally owned. 

      call MatGetOwnershipRange(A,Istart,Iend,ierr)

C  Set matrix elements for the 2-D, five-point stencil in parallel.
C   - Each processor needs to insert only elements that it owns
C     locally (but any non-local elements will be sent to the
C     appropriate processor during matrix assembly). 
C   - Always specify global row and columns of matrix entries.
C   - Note that MatSetValues() uses 0-based row and column numbers
C     in Fortran as well as in C.

      do 10, II=Istart,Iend-1
        v = -1.0
        i = II/n
        j = II - i*n  
        if ( i.gt.0 ) then
          JJ = II - n
          call MatSetValues(A,1,II,1,JJ,v,ADD_VALUES,ierr)
        endif
        if ( i.lt.m-1 ) then
          JJ = II + n
          call MatSetValues(A,1,II,1,JJ,v,ADD_VALUES,ierr)
        endif
        if ( j.gt.0 ) then
          JJ = II - 1
          call MatSetValues(A,1,II,1,JJ,v,ADD_VALUES,ierr)
        endif
        if ( j.lt.n-1 ) then
          JJ = II + 1
          call MatSetValues(A,1,II,1,JJ,v,ADD_VALUES,ierr)
        endif
        v = 4.0
        call  MatSetValues(A,1,II,1,II,v,ADD_VALUES,ierr)
 10   continue

C  Assemble matrix, using the 2-step process:
C       MatAssemblyBegin(), MatAssemblyEnd()
C  Computations can be done while messages are in transition,
C  by placing code between these two statements.

      call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr)

C  Create parallel vectors.
C   - Here, the parallel partitioning of the vector is determined by
C     PETSc at runtime.  We could also specify the local dimensions
C     if desired -- or use the more general routine VecCreate().
C   - When solving a linear system, the vectors and matrices MUST
C     be partitioned accordingly.  PETSc automatically generates
C     appropriately partitioned matrices and vectors when MatCreate()
C     and VecCreate() are used with the same communicator. 
C   - Note: We form 1 vector from scratch and then duplicate as needed.

      call VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,m*n,u,ierr)
      call VecDuplicate(u,b,ierr)
      call VecDuplicate(b,x,ierr)

C  Set exact solution; then compute right-hand-side vector.

      call VecSet(one,u,ierr)
      call MatMult(A,u,b,ierr)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C         Create the linear solver and set various options
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

C  Create linear solver context

      call SLESCreate(PETSC_COMM_WORLD,sles,ierr)

C  Set operators. Here the matrix that defines the linear system
C  also serves as the preconditioning matrix.

      call SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN,
     &                      ierr)

C  Set linear solver defaults for this problem (optional).
C   - By extracting the KSP and PC contexts from the SLES context,
C     we can then directly directly call any KSP and PC routines
C     to set various options.

      call SLESGetKSP(sles,ksp,ierr)
      call SLESGetPC(sles,pc,ierr)
      tol = 1.e-7
      call KSPSetTolerances(ksp,tol,PETSC_DEFAULT_DOUBLE_PRECISION,
     &     PETSC_DEFAULT_DOUBLE_PRECISION,PETSC_DEFAULT_INTEGER,ierr)

C
C  Set a user-defined shell preconditioner if desired
C
      call OptionsHasName(PETSC_NULL_CHARACTER,'-user_defined_pc',
     &                    user_defined_pc,ierr)

      if (user_defined_pc .eq. 1) then

C  (Required) Indicate to PETSc that we're using a shell preconditioner 
         call PCSetType(pc,PCSHELL,ierr)

C  (Required) Set the user-defined routine for applying the preconditioner
         call PCShellSetApply(pc,SampleShellPCApply,PETSC_NULL,ierr)

C  (Optional) Do any setup required for the preconditioner
         call SampleShellPCSetUp(A,x,ierr)

      else 
         call PCSetType(pc,PCJACOBI,ierr)
      endif

C  Set runtime options, e.g.,
C      -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
C  These options will override those specified above as long as
C  SLESSetFromOptions() is called _after_ any other customization
C  routines.

      call SLESSetFromOptions(sles,ierr)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C                      Solve the linear system
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call SLESSolve(sles,b,x,its,ierr)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C                     Check solution and clean up
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

C  Check the error

      call VecAXPY(neg_one,u,x,ierr)
      call VecNorm(x,NORM_2,norm,ierr)
      if (rank .eq. 0) then
        if (norm .gt. 1.e-12) then
           write(6,100) norm, its
        else
           write(6,110) its
        endif
      endif
  100 format('Norm of error ',e10.4,' iterations ',i5)
  110 format('Norm of error < 1.e-12, iterations ',i5)

C  Free work space.  All PETSc objects should be destroyed when they
C  are no longer needed.

      call SLESDestroy(sles,ierr)
      call VecDestroy(u,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(b,ierr)
      call MatDestroy(A,ierr)
      if (user_defined_pc .eq. 1) then
         call VecDestroy(diag,ierr)
      endif


C  Always call PetscFinalize() before exiting a program. 

      call PetscFinalize(ierr)
      end

C/***********************************************************************/
C/*          Routines for a user-defined shell preconditioner           */
C/***********************************************************************/

C
C   SampleShellPCSetUp - This routine sets up a user-defined
C   preconditioner context.  
C
C   Input Parameters:
C   pmat  - preconditioner matrix
C   x     - vector
C
C   Output Parameter:
C   ierr  - error code (nonzero if error has been detected)
C
C   Notes:
C   In this example, we define the shell preconditioner to be Jacobi's
C   method.  Thus, here we create a work vector for storing the reciprocal
C   of the diagonal of the preconditioner matrix; this vector is then
C   used within the routine SampleShellPCApply().
C
      subroutine SampleShellPCSetUp(pmat,x,ierr)

      implicit none

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/mat.h"

      Vec     x
      Mat     pmat
      integer ierr

C  Common block to store data for user-provided preconditioner 
      common /myshellpc/ diag
      Vec    diag

      call VecDuplicate(x,diag,ierr)
      call MatGetDiagonal(pmat,diag,ierr)
      call VecReciprocal(diag,ierr)

      end

C ------------------------------------------------------------------- 
C
C   SampleShellPCApply - This routine demonstrates the use of a
C   user-provided preconditioner.
C
C   Input Parameters:
C   dummy - optional user-defined context, not used here
C   x - input vector
C
C   Output Parameters:
C   y - preconditioned vector
C   ierr  - error code (nonzero if error has been detected)
C
C   Notes:
C   This code implements the Jacobi preconditioner, merely as an
C   example of working with a PCSHELL.  Note that the Jacobi method
C   is already provided within PETSc.
C
      subroutine SampleShellPCApply(dummy,x,y,ierr)

      implicit none

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"

      Vec     x, y
      integer dummy, ierr

C  Common block to store data for user-provided preconditioner 
      common /myshellpc/ diag
      Vec    diag

      call VecPointwiseMult(x,diag,y,ierr)

      end

