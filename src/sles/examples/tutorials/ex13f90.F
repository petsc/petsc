!
!    "$Id: ex13f.F,v 1.7 1998/03/30 22:03:29 balay Exp balay $";
!
!
!/*T
!   Concepts: SLES^Solving a system of linear equations (basic sequential example)
!   Concepts: SLES^Laplacian, 2d
!   Concepts: Laplacian, 2d
!   Routines: SLESCreate(); SLESSetOperators(); SLESSetFromOptions();
!   Routines: SLESSolve(); SLESGetKSP(); SLESGetPC(); MatCreateSeqAIJ();
!   Routines: KSPSetTolerances(); PCSetType();
!   Routines: VecGetArray(); VecPlaceArray();
!   Processors: 1
!T*/
! -----------------------------------------------------------------------

      program main
      implicit none

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                    Include files
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  The following include statements are required for SLES Fortran programs:
!     petsc.h  - base PETSc routines
!     vec.h    - vectors
!     mat.h    - matrices
!     ksp.h    - Krylov subspace methods
!     pc.h     - preconditioners
!     sles.h   - SLES interface
!
#include "include/finclude/petsc.h"
#include "include/finclude/vec.h"
#include "include/finclude/mat.h"
#include "include/finclude/ksp.h"
#include "include/finclude/pc.h"
#include "include/finclude/sles.h"

!    User-defined context that contains all the data structures used
!    in the linear solution process.

!   Vec    x,b;      /* solution vector, right hand side vector and work vector */
!   Mat    A;        /* sparse matrix */
!   SLES   sles;     /* linear solver context */
!   int    m,n;      /* grid dimensions */
!   Scalar hx2,hy2;  /* 1/(m+1)*(m+1) and 1/(n+1)*(n+1) */

!  Note: Any user-defined Fortran routines MUST be declared as external.

      external UserInitializeLinearSolver, UserFinalizeLinearSolver
      external UserDoLinearSolver

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Variables:
!     sles     - linear solver context
!     ksp      - Krylov subspace method context
!     pc       - preconditioner context
!     x, b, u  - approx solution, right-hand-side, exact solution vectors
!     A        - matrix that defines linear system
!     its      - iterations for convergence
!     norm     - norm of error in solution


      Vec     x, b, u
      Mat     A 
      KSP     ksp
      PC      pc
      SLES    sles
      Scalar  *rho, *solution, *userb, hx, hy, x, y
      integer ierr, m, n, t, tmax, flg, i, I, j, N
      integer userctx(6), size, rank
      double precision  enorm

      tmax = 2
      m = 6
      n = 7

      common /param/ hx2, hy2
      double precision hxy, hy2


! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr)
      if (size .ne. 1) then
         call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
         if (rank .eq. 0) then
            write(6,*) 'This is a uniprocessor example only!'
         endif
         SETERRA(1,0,' ')
      endif

!  The next two lines are for testing only; these allow the user to
!  decide the grid size at runtime.

      call OptionsGetInt(PETSC_NULL_CHARACTER,'-m',m,flg,ierr)
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-n',n,flg,ierr)

!  Create the empty sparse matrix and linear solver data structures

      call UserInitializeLinearSolver(m,n,userctx,ierr)
      Ntot = m*n

!  Allocate arrays to hold the solution to the linear system.
!  This is not normally done in PETSc programs, but in this case, 
!  since we are calling these routines from a non-PETSc program, we 
!  would like to reuse the data structures from another code. So in 
!  the context of a larger application these would be provided by
!  other (non-PETSc) parts of the application code.

  userx    = (Scalar *) PetscMalloc(Ntot*sizeof(Scalar)); CHKPTRA(userx);
  userb    = (Scalar *) PetscMalloc(Ntot*sizeof(Scalar)); CHKPTRA(userb);
  solution = (Scalar *) PetscMalloc(Ntot*sizeof(Scalar)); CHKPTRA(solution);

!  Allocate an array to hold the coefficients in the elliptic operator

  rho = (Scalar *) PetscMalloc(Ntot*sizeof(Scalar)); CHKERRA(ierr);

!  Fill up the array rho[] with the function rho(x,y) = x; fill the
!  right-hand-side b[] and the solution with a known problem for testing.

      hx = 1.0/(m+1)
      hy = 1.0/(n+1)
      y  = hy
      do 20 j=1,n
         x = hx
         do 10 i=1,m
            rho(i,j)      = x
            solution(i,j) = sin(2.*PETSC_PI*x)*sin(2.*PETSC_PI*y)
            userb(i,j)    = -2*PETSC_PI*cos(2*PETSC_PI*x)*              &
     &                sin(2*PETSC_PI*y) +                               &
     &                8*PETSC_PI*PETSC_PI*x*                            &
     &                sin(2*PETSC_PI*x)*sin(2*PETSC_PI*y)
           x = x + hx
 10      continue
         y = y + hy
 20   continue

!  Loop over a bunch of timesteps, setting up and solver the linear
!  system for each time-step.
!
!  Note this is somewhat artificial. It is intended to demonstrate how
!  one may reuse the linear solver stuff in each time-step.

      do 100 t=1,tmax
         call UserDoLinearSolver(rho,userctx,userb,userx,ierr)

!        Compute error: Note that this could (and usually should) all be done
!        using the PETSc vector operations. Here we demonstrate using more 
!        standard programming practices to show how they may be mixed with 
!        PETSc.
         enorm = 0.0
         do 90 i=1,N
            enorm = enorm +                                             &
     &       PetscReal(PetscConj(solution[i]-userx[i])                  &
     &           *(solution[i]-userx[i]))
 90      continue
         enorm = enorm * PetscReal(hx*hy);
         write(6,*) 'm, n, error norm =',m,n,enorm

!  We are all finished solving linear systems, so we clean up the
!  data structures.

  PetscFree(rho);
  PetscFree(solution);
  PetscFree(userx);
  PetscFree(userb);
      ierr = UserFinalizeLinearSolver(&userctx); CHKERRA(ierr);
      call PetscFinalize(ierr)
      end
! ----------------------------------------------------------------
integer function UserInitializeLinearSolver(m,n,userctx,ierr)

      implicit none

#include "include/finclude/petsc.h"
#include "include/finclude/vec.h"
#include "include/finclude/mat.h"
#include "include/finclude/ksp.h"
#include "include/finclude/pc.h"
#include "include/finclude/sles.h"

      common /param/ hx2, hy2
      double precision hxy, hy2

      integer m, n, user(*)
      integer Ntot, ierr

!  Here we assume use of a grid of size m x n, with all points on the
!  interior of the domain, i.e., we do not include the points corresponding 
!  to homogeneous Dirichlet boundary conditions.  We assume that the domain
!  is [0,1]x[0,1].

      hx2 = (m+1)*(m+1)
      hy2 = (n+1)*(n+1)
      Ntot = m*n

!  Create the sparse matrix. Preallocate 5 nonzeros per row.

      call MatCreateSeqAIJ(PETSC_COMM_SELF,Ntot,Ntot,5,0,A,ierr)

!  Create vectors

      call VecCreateSeq(PETSC_COMM_SELF,N,b,ierr)
      call VecDuplicate(b,x,ierr)

!  Create linear solver context. This will be used repeatedly for all 
!  the linear solves needed.

      call SLESCreate(PETSC_COMM_SELF,sles,ierr)

      user(1) = x
      user(2) = b
      user(3) = A
      user(4) = sles
      user(5) = m
      user(6) = n

      return
      end
! -----------------------------------------------------------------------

!   Solves -div ( rho grad psi) = F using finite differences.
!   rho is a 2-dimensional array of size m by n, stored in Fortran
!   style by columns. userb is a standard one-dimensional array.

      integer function UserDoLinearSolver(rho,userctx,                  &
     &                                    userb,userx,ierr)

  Scalar *rho,UserCtx *userctx,Scalar *userb,Scalar *userx)
  int    ierr,i,j,I,J, m = userctx->m, n = userctx->n,its;
  Mat    A = userctx->A;
  PC     pc;
  Scalar v,*tmpx,*tmpb, hx2 = userctx->hx2, hy2 = userctx->hy2;

      common /param/ hx2, hy2
      double precision hxy, hy2

!  This is not the most efficient way of generating the matrix 
!  but let's not worry about it. We should have separate code for
!  the four corners, each edge and then the interior. Then we won't
!  have the slow if-tests inside the loop.
!
!  Computes the operator 
!          -div rho grad 
!  on an m by n grid with zero Dirichlet boundary conditions. The rho
!  is assumed to be given on the same grid as the finite difference 
!  stencil is applied.  For a staggered grid, one would have to change
!  things slightly.

  I = 0;
  for ( j=0; j<n; j++ ) {
    for ( i=0; i<m; i++) {
      if ( j>0 )   {
        J = I - m; 
        v = -.5*(rho[I] + rho[J])*hy2;
        MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);
      }
      if ( j<n-1 ) {
        J = I + m; 
        v = -.5*(rho[I] + rho[J])*hy2;
        MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);
      }
      if ( i>0 )   {
        J = I - 1; 
        v = -.5*(rho[I] + rho[J])*hx2;
        MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);
      }
      if ( i<m-1 ) {
        J = I + 1; 
        v = -.5*(rho[I] + rho[J])*hx2;
        MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);
      }
      v = 2*rho[I]*(hx2+hy2);
      MatSetValues(A,1,&I,1,&I,&v,INSERT_VALUES);     
      I++;
    }
  }

  /* 
     Assemble matrix
  */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  /* 
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix. Since all the matrices
     will have the same nonzero pattern here, we indicate this so the
     linear solvers can take advantage of this.
  */
  ierr = SLESSetOperators(userctx->sles,A,A,SAME_NONZERO_PATTERN); CHKERRQ(ierr);

  /* 
     Set linear solver defaults for this problem (optional).
     - Here we set it to use direct LU factorization for the solution
  */
  ierr = SLESGetPC(userctx->sles,&pc); CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU); CHKERRQ(ierr);

  /* 
     Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
     These options will override those specified above as long as
     SLESSetFromOptions() is called _after_ any other customization
     routines.
 
     Run the program with the option -help to see all the possible
     linear solver options.
  */
  ierr = SLESSetFromOptions(userctx->sles); CHKERRQ(ierr);

  /*
     This allows the PETSc linear solvers to compute the solution 
     directly in the user's array rather than in the PETSc vector.
 
     This is essentially a hack and not highly recommend unless you 
     are quite comfortable with using PETSc. In general, users should
     write their entire application using PETSc vectors rather than 
     arrays.
  */
  ierr = VecGetArray(userctx->x,&tmpx);
  ierr = VecGetArray(userctx->b,&tmpb);
  ierr = VecPlaceArray(userctx->x,userx); CHKERRQ(ierr);
  ierr = VecPlaceArray(userctx->b,userb); CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SLESSolve(userctx->sles,userctx->b,userctx->x,&its); CHKERRQ(ierr);

  /*
    Put back the PETSc array that belongs in the vector xuserctx->x
  */
  ierr = VecPlaceArray(userctx->x,tmpx);
  ierr = VecPlaceArray(userctx->b,tmpb);

  return 0;
}

/* ------------------------------------------------------------------------*/
int UserFinalizeLinearSolver(UserCtx *userctx)
{
  int ierr;
  /* 
     We are all done and don't need to solve any more linear systems, so
     we free the work space.  All PETSc objects should be destroyed when
     they are no longer needed.
  */
  ierr = SLESDestroy(userctx->sles); CHKERRQ(ierr);
  ierr = VecDestroy(userctx->x); CHKERRQ(ierr);
  ierr = VecDestroy(userctx->b); CHKERRQ(ierr);  
  ierr = MatDestroy(userctx->A); CHKERRQ(ierr);
  return 0;
}
