
C    "$Id: ex6f.F 1.13 1996/08/27 21:36:04 curfman Exp curfman $";
C
C  Description: This example demonstrates repeated linear solves as
C  well as the use of different preconditioner and linear system
C  matrices.  This example also illustrates how to save PETSc objects
C  in common blocks.
C
C/*T
C  Concepts: SLES (solving linear systems)
C  Routines: SLESCreate(); SLESSetOperators(); SLESSetFromOptions();
C  Routines: SLESSolve(); SLESView(); SLESGetKSP(); KSPSetMonitor();
C  Routines: KSPBuildSolution();
C  Processors: n
CT*/
C
C  The following include statements are required for SLES Fortran programs:
C     petsc.h  - base PETSc routines
C     vec.h    - vectors
C     mat.h    - matrices
C     pc.h     - preconditioners
C     ksp.h    - Krylov subspace methods
C     sles.h   - SLES interface
C  Other include statements may be needed if using additional PETSc
C  routines in a Fortran program, e.g.,
C     viewer.h - viewers
C     is.h     - index sets
C
#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/sles.h"
#include "include/FINCLUDE/pc.h"
#include "include/FINCLUDE/ksp.h"

C  Variables:
C
C  A       - matrix that defines linear system
C  sles    - SLES context
C  ksp     - KSP context
C  x, b, u - approx solution, RHS, exact solution vectors
C
      Vec     x, u, b
      Mat     A
      SLES    sles
      integer i, j, II, JJ, ierr, m, n
      integer Istart, Iend, flg, nsteps
      Scalar  v

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      m      = 3
      n      = 3
      nsteps = 2
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-m',m,flg,ierr)
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-n',n,flg,ierr)
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-nsteps',nsteps,flg,ierr)

C  Create parallel matrix, specifying only its global dimensions.
C  When using MatCreate(), the matrix format can be specified at
C  runtime. Also, the parallel partioning of the matrix is
C  determined by PETSc at runtime.

      call MatCreate(MPI_COMM_WORLD,m*n,m*n,A,ierr)

C  The matrix is partitioned by contiguous chunks of rows across the
C  processors.  Determine which rows of the matrix are locally owned. 

      call MatGetOwnershipRange(A,Istart,Iend,ierr)

C  Set matrix elements. 
C   - Each processor needs to insert only elements that it owns
C     locally (but any non-local elements will be sent to the
C     appropriate processor during matrix assembly). 
C   - Always specify global row and columns of matrix entries.

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
C   - When using VecCreate(), the parallel partitioning of the vector
C     is determined by PETSc at runtime.
C   - Note: We form 1 vector from scratch and then duplicate as needed.

      call VecCreate(MPI_COMM_WORLD,m*n,u,ierr)
      call VecDuplicate(u,b,ierr)
      call VecDuplicate(b,x,ierr)

C  Create linear solver context

      call SLESCreate(MPI_COMM_WORLD,sles,ierr)

C  Set runtime options (e.g., -ksp_type <type> -pc_type <type>)

      call SLESSetFromOptions(sles,ierr)

C  Solve several linear systems in succession

      do 100 i=1,nsteps
         call solve1(sles,A,x,b,u,i,nsteps,ierr)
 100  continue

C  Free work space.  All PETSc objects should be destroyed when they
C  are no longer needed.

      call VecDestroy(u,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(b,ierr)
      call MatDestroy(A,ierr)
      call SLESDestroy(sles,ierr)

      call PetscFinalize(ierr)
      stop
      end

C -----------------------------------------------------------------------
C
      subroutine solve1(sles,A,x,b,u,count,nsteps,ierr)

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/sles.h"
#include "include/FINCLUDE/pc.h"
#include "include/FINCLUDE/ksp.h"

C
C   solve1 - This routine is used for repeated linear system solves.
C   We update the linear system matrix each time, but retain the same
C   preconditioning matrix for all linear solves.
C
C      A - linear system matrix
C      A2 - preconditioning matrix
C
      Scalar  v, val
      integer II, ierr, Istart, Iend, count, nsteps, its
      Mat     A
      SLES    sles
      KSP     ksp
      Vec     x, b, u

C Use common block to retain matrix between successive subroutine calls
      Mat              A2
      integer          rank, pflag
      common /my_data/ A2, pflag, rank

C First time thorough: Create new matrix to define the linear system
      if (count .eq. 1) then
        call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
        pflag = 0
        call OptionsHasName(PETSC_NULL_CHARACTER,"-mat_view",pflag,ierr)
        if (pflag .ne. 0) then
          if (rank .eq. 0) write(6,100)
        endif
        call MatConvert(A,MATSAME,A2,ierr)
C All other times: Set previous solution as initial guess for next solve.
      else
        call SLESGetKSP(sles,ksp,ierr)
        call KSPSetInitialGuessNonzero(ksp,ierr)
      endif

C Alter the matrix A a bit
      call MatGetOwnershipRange(A,Istart,Iend,ierr)
      do 20, II=Istart,Iend-1
        v = 2.0
        call MatSetValues(A,1,II,1,II,v,ADD_VALUES,ierr)
 20   continue
      call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr)
      if (pflag .ne. 0) then
        if (rank .eq. 0) write(6,110)
      endif
      call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr)

C Set the exact solution; compute the right-hand-side vector
      val = 1.0*count
      call VecSet(val,u,ierr)
      call MatMult(A,u,b,ierr)

C Set operators, keeping the identical preconditioner matrix for
C all linear solves.  This approach is often effective when the
C linear systems do not change very much between successive steps.
      call SLESSetOperators(sles,A,A2,SAME_PRECONDITIONER,ierr)

C Solve linear system
      call SLESSolve(sles,b,x,its,ierr)

C Destroy the preconditioner matrix on the last time through
      if (count .eq. nsteps) call MatDestroy(A2,ierr)

 100  format('previous matrix: preconditioning')
 110  format('next matrix: defines linear system')

      end

