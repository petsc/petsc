
C    "$Id: ex1f.F,v 1.8 1996/09/28 14:11:03 curfman Exp balay $";
C
C   Description: Solves a tridiagonal linear system with SLES.
C
C/*T
C   Concepts: SLES^Solving a system of linear equations (basic uniprocessor example);
C   Routines: SLESCreate(); SLESSetOperators(); SLESSetFromOptions();
C   Routines: SLESSolve(); SLESView(); SLESGetKSP(); SLESGetPC();
C   Routines: KSPSetTolerances(); PCSetType();
C   Processors: 1
CT*/
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
C     ksp.h    - Krylov subspace methods
C     pc.h     - preconditioners
C     sles.h   - SLES interface
C  Other include statements may be needed if using additional PETSc
C  routines in a Fortran program, e.g.,
C     viewer.h - viewers
C     is.h     - index sets
C
#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/ksp.h"
#include "include/FINCLUDE/pc.h"
#include "include/FINCLUDE/sles.h"
C
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
C     norm     - norm of error in solution
C
C  Note:  "Double" -> "double precision" except for machines such
C          as the Cray T3d, where "Double" -> "real"

      Vec      x, b, u
      Mat      A 
      KSP      ksp
      PC       pc
      SLES     sles
      Double   norm
      integer  ierr, i, n, col(3), its, flg, size, rank
      Scalar   none, one, value(3)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C                 Beginning of program
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call MPI_Comm_size(MPI_COMM_WORLD,size,ierr)
      if (size .ne. 1) then
         call MPI_Comm_size(MPI_COMM_WORLD,rank,ierr)
         if (rank .eq. 0)
     &      write(6,*) 'This is a uniprocessor example only!'
         SETERRA(1,0)
      endif
      none = -1.0
      one  = 1.0
      n    = 10
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-n',n,flg,ierr)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C         Compute the matrix and right-hand-side vector that define
C         the linear system, Ax = b.
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

C  Create matrix.  When using MatCreate(), the matrix format can
C  be specified at runtime.

      call MatCreate(MPI_COMM_WORLD,n,n,A,ierr)

C  Assemble matrix.
C   - Note that MatSetValues() uses 0-based row and column numbers
C     in Fortran as well as in C (as set here in the array "col").

      value(1) = -1.0
      value(2) = 2.0
      value(3) = -1.0
      do 50 i=1,n-2
         col(1) = i-1
         col(2) = i
         col(3) = i+1
         call MatSetValues(A,1,i,3,col,value,INSERT_VALUES,ierr)
  50  continue
      i = n - 1
      col(1) = n - 2
      col(2) = n - 1
      call MatSetValues(A,1,i,2,col,value,INSERT_VALUES,ierr)
      i = 0
      col(1) = 0
      col(2) = 1
      value(1) = 2.0
      value(2) = -1.0
      call MatSetValues(A,1,i,2,col,value,INSERT_VALUES,ierr)
      call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr)

C  Create vectors.  Note that we form 1 vector from scratch and
C  then duplicate as needed.

      call VecCreate(MPI_COMM_WORLD,n,x,ierr)
      call VecDuplicate(x,b,ierr)
      call VecDuplicate(x,u,ierr)

C  Set exact solution; then compute right-hand-side vector.

      call VecSet(one,u,ierr)
      call MatMult(A,u,b,ierr)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C          Create the linear solver and set various options
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

C  Create linear solver context

      call SLESCreate(MPI_COMM_WORLD,sles,ierr)

C  Set operators. Here the matrix that defines the linear system
C  also serves as the preconditioning matrix.

      call SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN,ierr)

C  Set linear solver defaults for this problem (optional).
C   - By extracting the KSP and PC contexts from the SLES context,
C     we can then directly directly call any KSP and PC routines
C     to set various options.
C   - The following four statements are optional; all of these
C     parameters could alternatively be specified at runtime via
C     SLESSetFromOptions();

      call SLESGetKSP(sles,ksp,ierr)
      call SLESGetPC(sles,pc,ierr)
      call PCSetType(pc,PCJACOBI,ierr)
      call KSPSetTolerances(ksp,1.d-7,PETSC_DEFAULT_DOUBLE_PRECISION,
     &     PETSC_DEFAULT_DOUBLE_PRECISION,PETSC_DEFAULT_INTEGER,ierr)

C  Set runtime options, e.g.,
C      -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
C  These options will override those specified above as long as
C  SLESSetFromOptions() is called _after_ any other customization
C  routines.

      call SLESSetFromOptions(sles,ierr)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C                      Solve the linear system
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call SLESSolve(sles,b,x,its,ierr)

C  View solver info; we could instead use the option -sles_view

      call SLESView(sles,VIEWER_STDOUT_WORLD,ierr)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C                      Check solution and clean up
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

C  Check the error

      call VecAXPY(none,u,x,ierr)
      call VecNorm(x,NORM_2,norm,ierr)
      if (norm .gt. 1.e-12) then
        write(6,100) norm, its
      else
        write(6,200) its
      endif
 100  format('Norm of error = ',e10.4,',  Iterations = ',i5)
 200  format('Norm of error < 1.e-12, Iterations = ',i5)

C  Free work space.  All PETSc objects should be destroyed when they
C  are no longer needed.

      call VecDestroy(x,ierr)
      call VecDestroy(u,ierr)
      call VecDestroy(b,ierr)
      call MatDestroy(A,ierr)
      call SLESDestroy(sles,ierr)
      call PetscFinalize(ierr)

      end
