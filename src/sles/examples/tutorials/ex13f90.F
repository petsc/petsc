C
C    "$Id: ex14f.F,v 1.2 1997/04/25 01:44:22 curfman Exp curfman $";
C
C
C/*T
C   Concepts: SLES^Solving a system of linear equations (basic sequential example)
C   Concepts: SLES^Laplacian, 2d
C   Concepts: Laplacian, 2d
C   Routines: SLESCreate(); SLESSetOperators(); SLESSetFromOptions();
C   Routines: SLESSolve(); SLESGetKSP(); SLESGetPC(); MatCreateSeqAIJ();
C   Routines: KSPSetTolerances(); PCSetType();
C   Routines: VecGetArray(); VecPlaceArray();
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
C
#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/ksp.h"
#include "include/FINCLUDE/pc.h"
#include "include/FINCLUDE/sles.h"

C    User-defined context that contains all the data structures used
C    in the linear solution process.

C   Vec    x,b;      /* solution vector, right hand side vector and work vector */
C   Mat    A;        /* sparse matrix */
C   SLES   sles;     /* linear solver context */
C   int    m,n;      /* grid dimensions */
C   Scalar hx2,hy2;  /* 1/(m+1)*(m+1) and 1/(n+1)*(n+1) */

C  Note: Any user-defined Fortran routines MUST be declared as external.

      external UserInitializeLinearSolver, UserFinalizeLinearSolver
      external UserDoLinearSolver

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

#define PETSC_PI 3.14159265

      Vec      x, b, u
      Mat      A 
      KSP      ksp
      PC       pc
      SLES     sles
      Double   enorm
      Scalar  *rho, *solution, *userb, hx, hy, x, y
      integer  ierr, m, n, t, tmax, flg, i, I, j, N
      integer  userctx(6), size, rank

      tmax = 2
      m = 6
      n = 7

      common /param/ hx2, hy2
      Double hxy, hy2


C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C                 Beginning of program
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call MPI_Comm_size(MPI_COMM_WORLD,size,ierr)
      if (size .ne. 1) then
         call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
         if (rank .eq. 0)
     &      write(6,*) 'This is a uniprocessor example only!'
         SETERRA(1,0,' ')
      endif

C  The next two lines are for testing only; these allow the user to
C  decide the grid size at runtime.

      call OptionsGetInt(PETSC_NULL_CHARACTER,'-m',m,flg,ierr)
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-n',n,flg,ierr)

C  Create the empty sparse matrix and linear solver data structures

      call UserInitializeLinearSolver(m,n,userctx,ierr)
      Ntot = m*n

C  Allocate arrays to hold the solution to the linear system.
C  This is not normally done in PETSc programs, but in this case, 
C  since we are calling these routines from a non-PETSc program, we 
C  would like to reuse the data structures from another code. So in 
C  the context of a larger application these would be provided by
C  other (non-PETSc) parts of the application code.

  userx    = (Scalar *) PetscMalloc(Ntot*sizeof(Scalar)); CHKPTRA(userx);
  userb    = (Scalar *) PetscMalloc(Ntot*sizeof(Scalar)); CHKPTRA(userb);
  solution = (Scalar *) PetscMalloc(Ntot*sizeof(Scalar)); CHKPTRA(solution);

C  Allocate an array to hold the coefficients in the elliptic operator

  rho = (Scalar *) PetscMalloc(Ntot*sizeof(Scalar)); CHKERRA(ierr);

C  Fill up the array rho[] with the function rho(x,y) = x; fill the
C  right-hand-side b[] and the solution with a known problem for testing.

      hx = 1.0/(m+1)
      hy = 1.0/(n+1)
      y  = hy
      do 20 j=1,n
         x = hx
         do 10 i=1,m
            rho(i,j)      = x
            solution(i,j) = sin(2.*PETSC_PI*x)*sin(2.*PETSC_PI*y)
            userb(i,j)    = -2*PETSC_PI*cos(2*PETSC_PI*x)*
     &                sin(2*PETSC_PI*y) +
     &                8*PETSC_PI*PETSC_PI*x*
     &                sin(2*PETSC_PI*x)*sin(2*PETSC_PI*y)
           x = x + hx
 10      continue
         y = y + hy
 20   continue

C  Loop over a bunch of timesteps, setting up and solver the linear
C  system for each time-step.
C
C  Note this is somewhat artificial. It is intended to demonstrate how
C  one may reuse the linear solver stuff in each time-step.

      do 100 t=1,tmax
         call UserDoLinearSolver(rho,userctx,userb,userx,ierr)

C        Compute error: Note that this could (and usually should) all be done
C        using the PETSc vector operations. Here we demonstrate using more 
C        standard programming practices to show how they may be mixed with 
C        PETSc.
         enorm = 0.0
         do 90 i=1,N
            enorm = enorm + 
     &       PetscReal(PetscConj(solution[i]-userx[i])
     &           *(solution[i]-userx[i]));
 90      continue
         enorm = enorm * PetscReal(hx*hy);
         write(6,*) 'm, n, error norm =',m,n,enorm

C  We are all finished solving linear systems, so we clean up the
C  data structures.

  PetscFree(rho);
  PetscFree(solution);
  PetscFree(userx);
  PetscFree(userb);
      ierr = UserFinalizeLinearSolver(&userctx); CHKERRA(ierr);
      call PetscFinalize(ierr)

      stop
      end
C ----------------------------------------------------------------
integer function UserInitializeLinearSolver(m,n,userctx,ierr)

      implicit none

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/ksp.h"
#include "include/FINCLUDE/pc.h"
#include "include/FINCLUDE/sles.h"

      common /param/ hx2, hy2
      Double hxy, hy2

      integer m, n, user(*)
      integer Ntot, ierr

C  Here we assume use of a grid of size m x n, with all points on the
C  interior of the domain, i.e., we do not include the points corresponding 
C  to homogeneous Dirichlet boundary conditions.  We assume that the domain
C  is [0,1]x[0,1].

      hx2 = (m+1)*(m+1)
      hy2 = (n+1)*(n+1)
      Ntot = m*n

C  Create the sparse matrix. Preallocate 5 nonzeros per row.

      call MatCreateSeqAIJ(PETSC_COMM_SELF,Ntot,Ntot,5,0,A,ierr)

C  Create vectors

      call VecCreateSeq(PETSC_COMM_SELF,N,b,ierr)
      call VecDuplicate(b,x,ierr)

C  Create linear solver context. This will be used repeatedly for all 
C  the linear solves needed.

      call SLESCreate(PETSC_COMM_SELF,sles,ierr)

      user(1) = x
      user(2) = b
      user(3) = A
      user(4) = sles
      user(5) = m
      user(6) = n

      return
      end
C -----------------------------------------------------------------------

C   Solves -div ( rho grad psi) = F using finite differences.
C   rho is a 2-dimensional array of size m by n, stored in Fortran
C   style by columns. userb is a standard one-dimensional array.

      integer function UserDoLinearSolver(rho,userctx,
     &                                    userb,userx,ierr)

  Scalar *rho,UserCtx *userctx,Scalar *userb,Scalar *userx)
  int    ierr,i,j,I,J, m = userctx->m, n = userctx->n,its;
  Mat    A = userctx->A;
  PC     pc;
  Scalar v,*tmpx,*tmpb, hx2 = userctx->hx2, hy2 = userctx->hy2;

      common /param/ hx2, hy2
      Double hxy, hy2

C  This is not the most efficient way of generating the matrix 
C  but let's not worry about it. We should have separate code for
C  the four corners, each edge and then the interior. Then we won't
C  have the slow if-tests inside the loop.
C
C  Computes the operator 
C          -div rho grad 
C  on an m by n grid with zero Dirichlet boundary conditions. The rho
C  is assumed to be given on the same grid as the finite difference 
C  stencil is applied.  For a staggered grid, one would have to change
C  things slightly.

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
