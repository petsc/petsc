
C    "$Id: ex7.F,v 1.23 1996/08/17 17:12:38 curfman Exp curfman $";
C
C  Description: Solves a linear system in parallel with SLES (Fortran code).
C               Also shows how to set a user-defined monitoring routine.
C
C  Concepts: SLES, solving linear systems
C  Routines: SLESCreate(), SLESSetOperators(), SLESSetFromOptions()
C  Routines: SLESSolve(), SLESView(), SLESGetKSP(), KSPSetMonitor()
C  Routines: KSPBuildSolution()
C  Multiprocessor code
C
C  The following include statements are required for SLES Fortran programs:
C     petsc.h  - base PETSc routines
C     vec.h    - vectors
C     mat.h    - matrices
C     pc.h     - preconditioners
C     ksp.h    - Krylov subspace methods
C     sles.h   - SLES interface
C  Additional include statements may be needed if using additional
C  PETSc routines in a Fortran program, e.g.,
C     viewer.h - viewers
C     is.h     - index sets

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/pc.h"
#include "include/FINCLUDE/ksp.h"
#include "include/FINCLUDE/sles.h"
C
C  Variable declarations:
C
C  A - matrix that defines linear system
C  sles - SLES context
C  ksp - KSP context
C  x, b, u - approx solution, RHS, exact solution vectors
C  norm - norm of error in solution
C
C  Note:  "Double" -> "double precision" except for machines such
C          as the Cray T3d, where "Double" -> "real"

      Double   norm
      integer  i, j, II, JJ, ierr, m, n
      integer  rank, size, its, Istart, Iend, flg
      Scalar   v, one, none
      Vec      x, b, u
      Mat      A 
      KSP      ksp
      SLES     sles

C  Note: Any user-defined Fortran routines (such as MyKSPMonitor)
C  MUST be declared as external.

      external MyKSPMonitor

      one  = 1.0
      none = -1.0
      call PetscInitialize(PETSC_NULL_CHAR,ierr)
      m = 3
      n = 3
      call OptionsGetInt(PETSC_NULL_CHAR,'-m',m,flg,ierr)
      call OptionsGetInt(PETSC_NULL_CHAR,'-n',n,flg,ierr)
      call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
      call MPI_Comm_size(MPI_COMM_WORLD,size,ierr)

C  Create parallel matrix.  Here we use the MATMPIAIJ matrix format.
C   - We specify only the global matrix dimensions, and PETSc
C     determines the parallel partioning of the matrix at runtime.
C     We could also specify the local matrix dimensions if desired.
C   - Alternative: MatCreate(), enables the matrix format to be
C     specified at runtime.

      call MatCreateMPIAIJ(MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,
     *                      m*n,m*n,0,PETSC_NULL,0,PETSC_NULL,A,ierr)

C  The matrix is partitioned by contiguous chunks of rows across the
C  processors.  Determine which rows of the matrix are locally owned. 

      call MatGetOwnershipRange(A,Istart,Iend,ierr)

C  Set matrix elements for the 2-D, five-point stencil in parallel.
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
C   - Here, the parallel partitioning of the vector is determined by
C     PETSc at runtime.  We could also specify the local dimensions
C     if desired.
C   - Note: We form 1 vector from scratch and then duplicate as needed.

      call VecCreateMPI(MPI_COMM_WORLD,PETSC_DECIDE,m*n,u,ierr)
      call VecDuplicate(u,b,ierr)
      call VecDuplicate(b,x,ierr)

C  Set exact solution; then compute right-hand-side vector.

      call VecSet(one,u,ierr)
      call MatMult(A,u,b,ierr)

C  Create linear solver context

      call SLESCreate(MPI_COMM_WORLD,sles,ierr)

C  Set operators. Here the matrix that defines the linear system
C  also serves as the preconditioning matrix.

      call SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN,
     *                      ierr)

C  Set user-defined monitoring routine if desired

      flg = 0
      call OptionsHasName(PETSC_NULL_CHAR,"-my_ksp_monitor",flg,ierr)
      if (flg .eq. 1) then
        call SLESGetKSP(sles,ksp,ierr)
        call KSPSetMonitor(ksp,MyKSPMonitor,PETSC_NULL,ierr)
      endif

C  Set runtime options (e.g., -ksp_type <type> -pc_type <type>)

      call SLESSetFromOptions(sles,ierr)

C  Solve linear system

      call SLESSolve(sles,b,x,its,ierr)

C  Check the error

      call VecAXPY(none,u,x,ierr)
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

      call PetscFinalize(ierr)
      stop
      end
C --------------------------------------------------------------
C
C  MyKSPMonitor - This is a user-defined routine for monitoring
C  the SLES iterative solvers.
C
C  Input Parameters:
C    ksp   - iterative context
C    n     - iteration number
C    rnorm - 2-norm (preconditioned) residual value (may be estimated)
C    dummy - optional user-defined monitor context (unused here)
C
      subroutine MyKSPMonitor(ksp,n,rnorm,dummy)

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/ksp.h"

      KSP       ksp
      Vec       x
      integer   ierr, n, dummy, rank
      Double    rnorm

C  Build the solution vector

      call KSPBuildSolution(ksp,PETSC_NULL,x,ierr)

C  Write the solution vector and residual norm to stdout
C   - Note that the parallel viewer VIEWER_STDOUT_WORLD
C     handles data from multiple processors so that the
C     output is not jumbled.

      call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
      if (rank .eq. 0) write(6,100) n
      call VecView(x,VIEWER_STDOUT_WORLD,ierr)
      if (rank .eq. 0) write(6,200) n, rnorm

 100  format('iteration ',i5,' solution vector:')
 200  format('iteration ',i5,' residual norm ',e10.4)
      end
