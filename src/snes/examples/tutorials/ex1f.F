
C "$Id: ex1f.F,v 1.6 1996/09/28 14:02:04 curfman Exp curfman $";
C
C/*T
C  Concepts: SNES^Solving a system of nonlinear equations (basic uniprocessor example)
C  Routines: SNESCreate(); SNESSetFunction(); SNESSetJacobian(); 
C  Routines: SNESSolve(); SNESSetFromOptions(); SNESGetSLES();
C  Routines: SLESGetPC(); SLESGetKSP(); KSPSetTolerances(); PCSetType();
C  Processors: 1
CT*/
C
C  Description: Uses the Newton method to solve a two-variable system.
C
C -----------------------------------------------------------------------

      program main
      implicit none

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C                    Include files
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C
C  The following include statements are generally used in SNES Fortran
C  programs:
C     petsc.h  - base PETSc routines
C     vec.h    - vectors
C     mat.h    - matrices
C     ksp.h    - Krylov subspace methods
C     pc.h     - preconditioners
C     sles.h   - SLES interface
C     snes.h   - SNES interface
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
#include "include/FINCLUDE/snes.h"
C
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C                   Variable declarations
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C
C  Variables:
C     snes        - nonlinear solver
C     sles        - linear solver
C     pc          - preconditioner context
C     ksp         - Krylov subspace method context
C     x, r        - solution, residual vectors
C     J           - Jacobian matrix
C     its         - iterations for convergence
C
      SNES     snes
      SLES     sles
      PC       pc
      KSP      ksp
      Vec      x, r
      Mat      J
      integer  ierr, its, size, rank
      Scalar   pfive

C  Note: Any user-defined Fortran routines (such as FormJacobian)
C  MUST be declared as external.

      external FormFunction, FormJacobian

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C                   Macro definitions
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C
C  Macros to make clearer the process of setting values in vectors and
C  getting values from vectors.  These vectors are used in the routines
C  FormFunction() and FormJacobian().
C   - The element lx_a(ib) is element ib in the vector x
C
#define lx_a(ib) lx_v(lx_i + (ib))
#define lf_a(ib) lf_v(lf_i + (ib))
C
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C                 Beginning of program
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call MPI_Comm_size(MPI_COMM_WORLD,size,ierr)
      if (size .ne. 1) then
         call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
         if (rank .eq. 0)
     &      write(6,*) 'This is a uniprocessor example only!'
         SETERRA(1,0)
      endif

C - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - - 
C  Create nonlinear solver context
C - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - - 

      call SNESCreate(MPI_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,snes,ierr)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C  Create matrix and vector data structures; set corresponding routines
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

C  Create vectors for solution and nonlinear function

      call VecCreateSeq(MPI_COMM_SELF,2,x,ierr)
      call VecDuplicate(x,r,ierr)

C  Create Jacobian matrix data structure

      call MatCreate(MPI_COMM_SELF,2,2,J,ierr)

C  Set function evaluation routine and vector

      call SNESSetFunction(snes,r,FormFunction,PETSC_NULL,ierr)

C  Set Jacobian matrix data structure and Jacobian evaluation routine

      call SNESSetJacobian(snes,J,J,FormJacobian,PETSC_NULL,ierr)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C  Customize nonlinear solver; set runtime options
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

C  Set linear solver defaults for this problem. By extracting the
C  SLES, KSP, and PC contexts from the SNES context, we can then
C  directly call any SLES, KSP, and PC routines to set various options.

      call SNESGetSLES(snes,sles,ierr)
      call SLESGetKSP(sles,ksp,ierr)
      call SLESGetPC(sles,pc,ierr)
      call PCSetType(pc,PCNONE,ierr)
      call KSPSetTolerances(ksp,1.d-4,PETSC_DEFAULT_DOUBLE_PRECISION,
     &                      PETSC_DEFAULT_DOUBLE_PRECISION,20,ierr)

C  Set SNES/SLES/KSP/PC runtime options, e.g.,
C      -snes_view -snes_monitor -ksp_type <ksp> -pc_type <pc>
C  These options will override those specified above as long as
C  SNESSetFromOptions() is called _after_ any other customization
C  routines.

      call SNESSetFromOptions(snes,ierr)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C  Evaluate initial guess; then solve nonlinear system
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

C  Note: The user should initialize the vector, x, with the initial guess
C  for the nonlinear solver prior to calling SNESSolve().  In particular,
C  to employ an initial guess of zero, the user should explicitly set
C  this vector to zero by calling VecSet().

      pfive = 0.5
      call VecSet(pfive,x,ierr)
      call SNESSolve(snes,x,its,ierr)
      if (rank .eq. 0) then
         write(6,100) its
      endif
  100 format('Number of Newton iterations = ',i5)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C  Free work space.  All PETSc objects should be destroyed when they
C  are no longer needed.
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call VecDestroy(x,ierr)
      call VecDestroy(r,ierr)
      call MatDestroy(J,ierr)
      call SNESDestroy(snes,ierr)
      call PetscFinalize(ierr)

      stop
      end
C ---------------------------------------------------------------------
C
C  FormFunction - Evaluates nonlinear function, F(x).
C
C  Input Parameters:
C  snes - the SNES context
C  x - input vector
C  dummy - optional user-defined context (not used here)
C
C  Output Parameter:
C  f - function vector
C
      subroutine FormFunction(snes,x,f,dummy)
      implicit none

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/snes.h"

      SNES     snes
      Vec      x, f
      integer  ierr, dummy(*)

C  Declarations for use with local arrays

      Scalar   lx_v(1), lf_v(1)
      integer  lx_i, lf_i 

C  Get pointers to vector data.
C    - For default PETSc vectors, VecGetArray() returns a pointer to
C      the data array.  Otherwise, the routine is implementation dependent.
C    - You MUST call VecRestoreArray() when you no longer need access to
C      the array.
C    - Note that the Fortran interface to VecGetArray() differs from the
C      C version.  See the Fortran chapter of the users manual for details.

      call VecGetArray(x,lx_v,lx_i,ierr)
      call VecGetArray(f,lf_v,lf_i,ierr)

C  Compute function

      lf_a(1) = lx_a(1)*lx_a(1) 
     &          + lx_a(1)*lx_a(2) - 3.0
      lf_a(2) = lx_a(1)*lx_a(2)
     &          + lx_a(2)*lx_a(2) - 6.0

C  Restore vectors

      call VecRestoreArray(x,lx_v,lx_i,ierr)
      call VecRestoreArray(f,lf_v,lf_i,ierr)

      return
      end

C ---------------------------------------------------------------------
C
C  FormJacobian - Evaluates Jacobian matrix.
C
C  Input Parameters:
C  snes - the SNES context
C  x - input vector
C  dummy - optional user-defined context (not used here)
C
C  Output Parameters:
C  A - Jacobian matrix
C  B - optionally different preconditioning matrix
C  flag - flag indicating matrix structure
C
      subroutine FormJacobian(snes,X,jac,B,flag,dummy)
      implicit none

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/pc.h"
#include "include/FINCLUDE/snes.h"

      SNES         snes
      Vec          X
      Mat          jac, B
      MatStructure flag
      Scalar       A(4)
      integer      ierr, idx(2), dummy(*)

C  Declarations for use with local arrays

      Scalar  lx_v(1)
      integer lx_i

C  Get pointer to vector data

      call VecGetArray(x,lx_v,lx_i,ierr)

C  Compute Jacobian entries and insert into matrix.
C   - Since this is such a small problem, we set all entries for
C     the matrix at once.
C   - Note that MatSetValues() uses 0-based row and column numbers
C     in Fortran as well as in C (as set here in the array idx).

      idx(1) = 0
      idx(2) = 1
      A(1) = 2.0*lx_a(1) + lx_a(2)
      A(2) = lx_a(1)
      A(3) = lx_a(2)
      A(4) = lx_a(1) + 2.0*lx_a(2)
      call MatSetValues(jac,2,idx,2,idx,A,INSERT_VALUES,ierr)
      flag = SAME_NONZERO_PATTERN

C  Restore vector

      call VecRestoreArray(x,lx_v,lx_i,ierr)

C  Assemble matrix

      call MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY,ierr)

      return
      end


