!
! "$Id: ex1f.F,v 1.33 2001/08/07 03:04:16 balay Exp $";
!
!  Description: Uses the Newton method to solve a two-variable system.
!
!/*T
!  Concepts: SNES^basic uniprocessor example
!  Processors: 1
!T*/
!
! -----------------------------------------------------------------------

      program main
      implicit none

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                    Include files
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  The following include statements are generally used in SNES Fortran
!  programs:
!     petsc.h       - base PETSc routines
!     petscvec.h    - vectors
!     petscmat.h    - matrices
!     petscksp.h    - Krylov subspace methods
!     petscpc.h     - preconditioners
!     petscsles.h   - SLES interface
!     petscsnes.h   - SNES interface
!  Other include statements may be needed if using additional PETSc
!  routines in a Fortran program, e.g.,
!     petscviewer.h - viewers
!     petscis.h     - index sets
!
#include "include/finclude/petsc.h"
#include "include/finclude/petscvec.h"
#include "include/finclude/petscmat.h"
#include "include/finclude/petscksp.h"
#include "include/finclude/petscpc.h"
#include "include/finclude/petscsles.h"
#include "include/finclude/petscsnes.h"
!
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Variables:
!     snes        - nonlinear solver
!     sles        - linear solver
!     pc          - preconditioner context
!     ksp         - Krylov subspace method context
!     x, r        - solution, residual vectors
!     J           - Jacobian matrix
!     its         - iterations for convergence
!
      SNES     snes
      SLES     sles
      PC       pc
      KSP      ksp
      Vec      x,r
      Mat      J
      integer  ierr,its,size,rank
      PetscScalar   pfive
      double precision   tol
      PetscTruth  setls

!  Note: Any user-defined Fortran routines (such as FormJacobian)
!  MUST be declared as external.

      external FormFunction, FormJacobian, MyLineSearch

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!                   Macro definitions
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!
!  Macros to make clearer the process of setting values in vectors and
!  getting values from vectors.  These vectors are used in the routines
!  FormFunction() and FormJacobian().
!   - The element lx_a(ib) is element ib in the vector x
!
#define lx_a(ib) lx_v(lx_i + (ib))
#define lf_a(ib) lf_v(lf_i + (ib))
!
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
      if (size .ne. 1) then
         if (rank .eq. 0) then
            write(6,*) 'This is a uniprocessor example only!'
         endif
         SETERRQ(1,' ',ierr)
      endif

! - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - - 
!  Create nonlinear solver context
! - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - - 

      call SNESCreate(PETSC_COMM_WORLD,snes,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create matrix and vector data structures; set corresponding routines
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Create vectors for solution and nonlinear function

      call VecCreateSeq(PETSC_COMM_SELF,2,x,ierr)
      call VecDuplicate(x,r,ierr)

!  Create Jacobian matrix data structure

      call MatCreate(PETSC_COMM_SELF,PETSC_DECIDE,PETSC_DECIDE,2,2,J,   &
     &               ierr)
      call MatSetFromOptions(J,ierr)

!  Set function evaluation routine and vector

      call SNESSetFunction(snes,r,FormFunction,PETSC_NULL_OBJECT,ierr)

!  Set Jacobian matrix data structure and Jacobian evaluation routine

      call SNESSetJacobian(snes,J,J,FormJacobian,PETSC_NULL_OBJECT,     &
     &     ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Customize nonlinear solver; set runtime options
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Set linear solver defaults for this problem. By extracting the
!  SLES, KSP, and PC contexts from the SNES context, we can then
!  directly call any SLES, KSP, and PC routines to set various options.

      call SNESGetSLES(snes,sles,ierr)
      call SLESGetKSP(sles,ksp,ierr)
      call KSPGetPC(ksp,pc,ierr)
      call PCSetType(pc,PCNONE,ierr)
      tol = 1.e-4
      call KSPSetTolerances(ksp,tol,PETSC_DEFAULT_DOUBLE_PRECISION,     &
     &                      PETSC_DEFAULT_DOUBLE_PRECISION,20,ierr)

!  Set SNES/SLES/KSP/PC runtime options, e.g.,
!      -snes_view -snes_monitor -ksp_type <ksp> -pc_type <pc>
!  These options will override those specified above as long as
!  SNESSetFromOptions() is called _after_ any other customization
!  routines.


      call SNESSetFromOptions(snes,ierr)

      call PetscOptionsHasName(PETSC_NULL_CHARACTER,'-setls',setls,ierr)

      if (setls .eq. PETSC_TRUE) then
        call SNESSetLineSearch(snes,MyLineSearch,                       &
     &                         PETSC_NULL_OBJECT,ierr)
      endif

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Evaluate initial guess; then solve nonlinear system
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Note: The user should initialize the vector, x, with the initial guess
!  for the nonlinear solver prior to calling SNESSolve().  In particular,
!  to employ an initial guess of zero, the user should explicitly set
!  this vector to zero by calling VecSet().

      pfive = 0.5
      call VecSet(pfive,x,ierr)
      call SNESSolve(snes,x,its,ierr)
      if (rank .eq. 0) then
         write(6,100) its
      endif
  100 format('Number of Newton iterations = ',i5)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call VecDestroy(x,ierr)
      call VecDestroy(r,ierr)
      call MatDestroy(J,ierr)
      call SNESDestroy(snes,ierr)
      call PetscFinalize(ierr)
      end
! ---------------------------------------------------------------------
!
!  FormFunction - Evaluates nonlinear function, F(x).
!
!  Input Parameters:
!  snes - the SNES context
!  x - input vector
!  dummy - optional user-defined context (not used here)
!
!  Output Parameter:
!  f - function vector
!
      subroutine FormFunction(snes,x,f,dummy,ierr)
      implicit none

#include "include/finclude/petsc.h"
#include "include/finclude/petscvec.h"
#include "include/finclude/petscsnes.h"

      SNES     snes
      Vec      x,f
      integer  ierr,dummy(*)

!  Declarations for use with local arrays

      PetscScalar  lx_v(1),lf_v(1)
      PetscOffset  lx_i,lf_i 

!  Get pointers to vector data.
!    - For default PETSc vectors, VecGetArray() returns a pointer to
!      the data array.  Otherwise, the routine is implementation dependent.
!    - You MUST call VecRestoreArray() when you no longer need access to
!      the array.
!    - Note that the Fortran interface to VecGetArray() differs from the
!      C version.  See the Fortran chapter of the users manual for details.

      call VecGetArray(x,lx_v,lx_i,ierr)
      call VecGetArray(f,lf_v,lf_i,ierr)

!  Compute function

      lf_a(1) = lx_a(1)*lx_a(1)                                         &
     &          + lx_a(1)*lx_a(2) - 3.0
      lf_a(2) = lx_a(1)*lx_a(2)                                         &
     &          + lx_a(2)*lx_a(2) - 6.0

!  Restore vectors

      call VecRestoreArray(x,lx_v,lx_i,ierr)
      call VecRestoreArray(f,lf_v,lf_i,ierr)

      return
      end

! ---------------------------------------------------------------------
!
!  FormJacobian - Evaluates Jacobian matrix.
!
!  Input Parameters:
!  snes - the SNES context
!  x - input vector
!  dummy - optional user-defined context (not used here)
!
!  Output Parameters:
!  A - Jacobian matrix
!  B - optionally different preconditioning matrix
!  flag - flag indicating matrix structure
!
      subroutine FormJacobian(snes,X,jac,B,flag,dummy,ierr)
      implicit none

#include "include/finclude/petsc.h"
#include "include/finclude/petscvec.h"
#include "include/finclude/petscmat.h"
#include "include/finclude/petscpc.h"
#include "include/finclude/petscsnes.h"

      SNES         snes
      Vec          X
      Mat          jac,B
      MatStructure flag
      PetscScalar  A(4)
      integer      ierr,idx(2),dummy(*)

!  Declarations for use with local arrays

      PetscScalar lx_v(1)
      PetscOffset lx_i

!  Get pointer to vector data

      call VecGetArray(x,lx_v,lx_i,ierr)

!  Compute Jacobian entries and insert into matrix.
!   - Since this is such a small problem, we set all entries for
!     the matrix at once.
!   - Note that MatSetValues() uses 0-based row and column numbers
!     in Fortran as well as in C (as set here in the array idx).

      idx(1) = 0
      idx(2) = 1
      A(1) = 2.0*lx_a(1) + lx_a(2)
      A(2) = lx_a(1)
      A(3) = lx_a(2)
      A(4) = lx_a(1) + 2.0*lx_a(2)
      call MatSetValues(jac,2,idx,2,idx,A,INSERT_VALUES,ierr)
      flag = SAME_NONZERO_PATTERN

!  Restore vector

      call VecRestoreArray(x,lx_v,lx_i,ierr)

!  Assemble matrix

      call MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY,ierr)

      return
      end


      subroutine MyLineSearch(snes,lctx,x,f,g,y,w,fnorm,ynorm,gnorm,     &
     &                        flag,ierr)
#include "include/finclude/petsc.h"
#include "include/finclude/petscvec.h"
#include "include/finclude/petscmat.h"
#include "include/finclude/petscksp.h"
#include "include/finclude/petscpc.h"
#include "include/finclude/petscsles.h"
#include "include/finclude/petscsnes.h"

      SNES              snes
      integer           lctx
      Vec               x, f,g, y, w
      double  precision fnorm,ynorm,gnorm
      integer           flag,ierr

      PetscScalar       mone

      mone = -1.0d0
      flag = 0
      call VecNorm(y,NORM_2,ynorm,ierr)
      call VecAYPX(mone,x,y,ierr)
      call SNESComputeFunction(snes,y,g,ierr)
      call VecNorm(g,NORM_2,gnorm,ierr)
      return 
      end
