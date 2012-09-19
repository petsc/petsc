
static char help[] = "Solves a variable Poisson problem with KSP.\n\n";

/*T
   Concepts: KSP^basic sequential example
   Concepts: KSP^Laplacian, 2d
   Concepts: Laplacian, 2d
   Processors: 1
T*/

/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/
#include <petscksp.h>

/*
    User-defined context that contains all the data structures used
    in the linear solution process.
*/
typedef struct {
   Vec         x,b;      /* solution vector, right-hand-side vector */
   Mat         A;         /* sparse matrix */
   KSP         ksp;      /* linear solver context */
   PetscInt    m,n;      /* grid dimensions */
   PetscScalar hx2,hy2;  /* 1/(m+1)*(m+1) and 1/(n+1)*(n+1) */
} UserCtx;

extern PetscErrorCode UserInitializeLinearSolver(PetscInt,PetscInt,UserCtx *);
extern PetscErrorCode UserFinalizeLinearSolver(UserCtx *);
extern PetscErrorCode UserDoLinearSolver(PetscScalar *,UserCtx *userctx,PetscScalar *b,PetscScalar *x);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  UserCtx        userctx;
  PetscErrorCode ierr;
  PetscInt       m = 6,n = 7,t,tmax = 2,i,Ii,j,N;
  PetscScalar    *userx,*rho,*solution,*userb,hx,hy,x,y;
  PetscReal      enorm;
  /*
     Initialize the PETSc libraries
  */
  PetscInitialize(&argc,&args,(char *)0,help);

  /*
     The next two lines are for testing only; these allow the user to
     decide the grid size at runtime.
  */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);

  /*
     Create the empty sparse matrix and linear solver data structures
  */
  ierr = UserInitializeLinearSolver(m,n,&userctx);CHKERRQ(ierr);
  N    = m*n;

  /*
     Allocate arrays to hold the solution to the linear system.
     This is not normally done in PETSc programs, but in this case,
     since we are calling these routines from a non-PETSc program, we
     would like to reuse the data structures from another code. So in
     the context of a larger application these would be provided by
     other (non-PETSc) parts of the application code.
  */
  ierr = PetscMalloc(N*sizeof(PetscScalar),&userx);CHKERRQ(ierr);
  ierr = PetscMalloc(N*sizeof(PetscScalar),&userb);CHKERRQ(ierr);
  ierr = PetscMalloc(N*sizeof(PetscScalar),&solution);CHKERRQ(ierr);

  /*
      Allocate an array to hold the coefficients in the elliptic operator
  */
  ierr = PetscMalloc(N*sizeof(PetscScalar),&rho);CHKERRQ(ierr);

  /*
     Fill up the array rho[] with the function rho(x,y) = x; fill the
     right-hand-side b[] and the solution with a known problem for testing.
  */
  hx = 1.0/(m+1);
  hy = 1.0/(n+1);
  y  = hy;
  Ii = 0;
  for (j=0; j<n; j++) {
    x = hx;
    for (i=0; i<m; i++) {
      rho[Ii]      = x;
      solution[Ii] = PetscSinScalar(2.*PETSC_PI*x)*PetscSinScalar(2.*PETSC_PI*y);
      userb[Ii]    = -2*PETSC_PI*PetscCosScalar(2*PETSC_PI*x)*PetscSinScalar(2*PETSC_PI*y) +
                    8*PETSC_PI*PETSC_PI*x*PetscSinScalar(2*PETSC_PI*x)*PetscSinScalar(2*PETSC_PI*y);
      x += hx;
      Ii++;
    }
    y += hy;
  }

  /*
     Loop over a bunch of timesteps, setting up and solver the linear
     system for each time-step.

     Note this is somewhat artificial. It is intended to demonstrate how
     one may reuse the linear solver stuff in each time-step.
  */
  for (t=0; t<tmax; t++) {
    ierr =  UserDoLinearSolver(rho,&userctx,userb,userx);CHKERRQ(ierr);

    /*
        Compute error: Note that this could (and usually should) all be done
        using the PETSc vector operations. Here we demonstrate using more
        standard programming practices to show how they may be mixed with
        PETSc.
    */
    enorm = 0.0;
    for (i=0; i<N; i++) {
      enorm += PetscRealPart(PetscConj(solution[i]-userx[i])*(solution[i]-userx[i]));
    }
    enorm *= PetscRealPart(hx*hy);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"m %D n %D error norm %G\n",m,n,enorm);CHKERRQ(ierr);
  }

  /*
     We are all finished solving linear systems, so we clean up the
     data structures.
  */
  ierr = PetscFree(rho);CHKERRQ(ierr);
  ierr = PetscFree(solution);CHKERRQ(ierr);
  ierr = PetscFree(userx);CHKERRQ(ierr);
  ierr = PetscFree(userb);CHKERRQ(ierr);
  ierr = UserFinalizeLinearSolver(&userctx);CHKERRQ(ierr);
  ierr = PetscFinalize();

  return 0;
}

/* ------------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "UserInitializedLinearSolve"
PetscErrorCode UserInitializeLinearSolver(PetscInt m,PetscInt n,UserCtx *userctx)
{
  PetscErrorCode ierr;
  PetscInt       N;

  /*
     Here we assume use of a grid of size m x n, with all points on the
     interior of the domain, i.e., we do not include the points corresponding
     to homogeneous Dirichlet boundary conditions.  We assume that the domain
     is [0,1]x[0,1].
  */
  userctx->m   = m;
  userctx->n   = n;
  userctx->hx2 = (m+1)*(m+1);
  userctx->hy2 = (n+1)*(n+1);
  N            = m*n;

  /*
     Create the sparse matrix. Preallocate 5 nonzeros per row.
  */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,N,N,5,0,&userctx->A);CHKERRQ(ierr);

  /*
     Create vectors. Here we create vectors with no memory allocated.
     This way, we can use the data structures already in the program
     by using VecPlaceArray() subroutine at a later stage.
  */
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,N,PETSC_NULL,&userctx->b);CHKERRQ(ierr);
  ierr = VecDuplicate(userctx->b,&userctx->x);CHKERRQ(ierr);

  /*
     Create linear solver context. This will be used repeatedly for all
     the linear solves needed.
  */
  ierr = KSPCreate(PETSC_COMM_SELF,&userctx->ksp);CHKERRQ(ierr);

  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "UserDoLinearSolve"
/*
   Solves -div (rho grad psi) = F using finite differences.
   rho is a 2-dimensional array of size m by n, stored in Fortran
   style by columns. userb is a standard one-dimensional array.
*/
/* ------------------------------------------------------------------------*/
PetscErrorCode UserDoLinearSolver(PetscScalar *rho,UserCtx *userctx,PetscScalar *userb,PetscScalar *userx)
{
  PetscErrorCode ierr;
  PetscInt       i,j,Ii,J,m = userctx->m,n = userctx->n;
  Mat            A = userctx->A;
  PC             pc;
  PetscScalar    v,hx2 = userctx->hx2,hy2 = userctx->hy2;

  /*
     This is not the most efficient way of generating the matrix
     but let's not worry about it. We should have separate code for
     the four corners, each edge and then the interior. Then we won't
     have the slow if-tests inside the loop.

     Computes the operator
             -div rho grad
     on an m by n grid with zero Dirichlet boundary conditions. The rho
     is assumed to be given on the same grid as the finite difference
     stencil is applied.  For a staggered grid, one would have to change
     things slightly.
  */
  Ii = 0;
  for (j=0; j<n; j++) {
    for (i=0; i<m; i++) {
      if (j>0)   {
        J    = Ii - m;
        v    = -.5*(rho[Ii] + rho[J])*hy2;
        ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
      }
      if (j<n-1) {
        J    = Ii + m;
        v    = -.5*(rho[Ii] + rho[J])*hy2;
        ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
      }
      if (i>0)   {
        J    = Ii - 1;
        v    = -.5*(rho[Ii] + rho[J])*hx2;
        ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
      }
      if (i<m-1) {
        J    = Ii + 1;
        v    = -.5*(rho[Ii] + rho[J])*hx2;
        ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
      }
      v    = 2.0*rho[Ii]*(hx2+hy2);
      ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
      Ii++;
    }
  }

  /*
     Assemble matrix
  */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /*
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix. Since all the matrices
     will have the same nonzero pattern here, we indicate this so the
     linear solvers can take advantage of this.
  */
  ierr = KSPSetOperators(userctx->ksp,A,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);

  /*
     Set linear solver defaults for this problem (optional).
     - Here we set it to use direct LU factorization for the solution
  */
  ierr = KSPGetPC(userctx->ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);

  /*
     Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
     These options will override those specified above as long as
     KSPSetFromOptions() is called _after_ any other customization
     routines.

     Run the program with the option -help to see all the possible
     linear solver options.
  */
  ierr = KSPSetFromOptions(userctx->ksp);CHKERRQ(ierr);

  /*
     This allows the PETSc linear solvers to compute the solution
     directly in the user's array rather than in the PETSc vector.

     This is essentially a hack and not highly recommend unless you
     are quite comfortable with using PETSc. In general, users should
     write their entire application using PETSc vectors rather than
     arrays.
  */
  ierr = VecPlaceArray(userctx->x,userx);CHKERRQ(ierr);
  ierr = VecPlaceArray(userctx->b,userb);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = KSPSolve(userctx->ksp,userctx->b,userctx->x);CHKERRQ(ierr);

  /*
    Put back the PETSc array that belongs in the vector xuserctx->x
  */
  ierr = VecResetArray(userctx->x);CHKERRQ(ierr);
  ierr = VecResetArray(userctx->b);CHKERRQ(ierr);

  return 0;
}

/* ------------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "UserFinalizeLinearSolve"
PetscErrorCode UserFinalizeLinearSolver(UserCtx *userctx)
{
  PetscErrorCode ierr;
  /*
     We are all done and don't need to solve any more linear systems, so
     we free the work space.  All PETSc objects should be destroyed when
     they are no longer needed.
  */
  ierr = KSPDestroy(&userctx->ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&userctx->x);CHKERRQ(ierr);
  ierr = VecDestroy(&userctx->b);CHKERRQ(ierr);
  ierr = MatDestroy(&userctx->A);CHKERRQ(ierr);
  return 0;
}
