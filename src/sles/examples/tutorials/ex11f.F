
C    "$Id: ex11f.F,v 1.2 1996/09/25 20:10:09 curfman Exp curfman $";
C
C/*T
C  Concepts: SLES (solving a Helmholtz equation)
C  Concepts: Complex numbers
C  Routines: SLESCreate(); SLESSetOperators(); SLESSetFromOptions();
C  Routines: SLESSolve(); PetscRandomCreate(); PetscRandomGetValue();
C  Routines: PetscRandomDestroy();
C  Processors: n
CT*/
C
C  Description: Solves a complex linear system in parallel with SLES (Fortran code).
C
C  The model problem:
C     Solve Helmholtz equation on the unit square: (0,1) x (0,1)
C          -delta u - sigma1*u + i*sigma2*u = f, 
C           where delta = Laplace operator
C     Dirichlet b.c.'s on all sides
C     Use the 2-D, five-point finite difference stencil.
C
C  Compiling the code:
C     This code uses the complex numbers version of PETSc, so one of the
C     following values of BOPT must be used for compiling the PETSc libraries
C     and this example:
C        BOPT=g_complex   - debugging version
C        BOPT=O_complex   - optimized version
C        BOPT=Opg_complex - profiling version
C
C -----------------------------------------------------------------------

      program main
      implicit none

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C                    Include files
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C
C  The following include statements are required for SLES Fortran programs:
C     petsc.h  - base PETSc routines
C     vec.h    - vectors
C     mat.h    - matrices
C     pc.h     - preconditioners
C     ksp.h    - Krylov subspace methods
C     sles.h   - SLES interface
C  Include the following to use PETSc random numbers:
C     sys.h    - system routines
C  Additional include statements may be needed if using other PETSc
C  routines in a Fortran program, e.g.,
C     viewer.h - viewers
C     is.h     - index sets
C
#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/pc.h"
#include "include/FINCLUDE/ksp.h"
#include "include/FINCLUDE/sles.h"
#include "include/FINCLUDE/sys.h"
C
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C                   Variable declarations
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C
C  Variables:
C     sles     - linear solver context
C     x, b, u  - approx solution, right-hand-side, exact solution vectors
C     A        - matrix that defines linear system
C     its      - iterations for convergence
C     norm     - norm of error in solution
C     rctx     - random number context
C
C  Note:  "Double" -> "double precision" except for machines such
C          as the Cray T3d, where "Double" -> "real"

      SLES        sles
      Mat         A 
      Vec         x, b, u
      PetscRandom rctx
      Double      norm, h2, sigma1
      integer     dim, flg, its, ierr, n, rank
      integer     Istart, Iend, i, j, II, JJ
      Scalar      none, sigma2, v

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C                 Beginning of program
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      none   = -1.0
      n      = 10
      sigma1 = 100.0
      call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
      call OptionsGetDouble(PETSC_NULL_CHARACTER,'-sigma1',sigma1,
     *                       flg,ierr)
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-n',n,flg,ierr)
      dim    = n*n

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C      Compute the matrix and right-hand-side vector that define
C      the linear system, Ax = b.
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

C  Create parallel matrix, specifying only its global dimensions.
C  When using MatCreate(), the matrix format can be specified at
C  runtime. Also, the parallel partioning of the matrix is
C  determined by PETSc at runtime.

      call MatCreate(MPI_COMM_WORLD,dim,dim,A,ierr)

C  Currently, all PETSc parallel matrix formats are partitioned by
C  contiguous chunks of rows across the processors.  Determine which
C  rows of the matrix are locally owned. 

      call MatGetOwnershipRange(A,Istart,Iend,ierr)

C  Set matrix elements in parallel.
C   - Each processor needs to insert only elements that it owns
C     locally (but any non-local elements will be sent to the
C     appropriate processor during matrix assembly). 
C   - Always specify global row and columns of matrix entries.

      call PetscRandomCreate(MPI_COMM_WORLD,RANDOM_DEFAULT_IMAGINARY,
     *                       rctx,ierr)
      h2 = 1.0/((n+1)*(n+1))

      do 10, II=Istart,Iend-1
        v = -1.0
        i = II/n
        j = II - i*n  
        if ( i.gt.0 ) then
          JJ = II - n
          call MatSetValues(A,1,II,1,JJ,v,ADD_VALUES,ierr)
        endif
        if ( i.lt.n-1 ) then
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
        call PetscRandomGetValue(rctx,sigma2,ierr)
        v = 4.0 - sigma1*h2 + sigma2*h2
        call  MatSetValues(A,1,II,1,II,v,ADD_VALUES,ierr)
 10   continue

      call PetscRandomDestroy(rctx,ierr)

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

      call VecCreate(MPI_COMM_WORLD,dim,u,ierr)
      call VecDuplicate(u,b,ierr)
      call VecDuplicate(b,x,ierr)

C  Set exact solution; then compute right-hand-side vector.

      call PetscRandomCreate(MPI_COMM_WORLD,RANDOM_DEFAULT,rctx,ierr)
      call VecSetRandom(rctx,u,ierr)
      call MatMult(A,u,b,ierr)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C         Create the linear solver and set various options
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

C  Create linear solver context

      call SLESCreate(MPI_COMM_WORLD,sles,ierr)

C  Set operators. Here the matrix that defines the linear system
C  also serves as the preconditioning matrix.

      call SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN,
     *                      ierr)

C  Set runtime options, e.g.,
C      -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>

      call SLESSetFromOptions(sles,ierr)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C                      Solve the linear system
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call SLESSolve(sles,b,x,its,ierr)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C                     Check solution and clean up
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

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

      call PetscRandomDestroy(rctx,ierr)
      call SLESDestroy(sles,ierr)
      call VecDestroy(u,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(b,ierr)
      call MatDestroy(A,ierr)

      call PetscFinalize(ierr)
      stop
      end
