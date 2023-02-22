
static char help[] = "Solves a variable Poisson problem with KSP.\n\n";

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
  Vec         x, b;     /* solution vector, right-hand-side vector */
  Mat         A;        /* sparse matrix */
  KSP         ksp;      /* linear solver context */
  PetscInt    m, n;     /* grid dimensions */
  PetscScalar hx2, hy2; /* 1/(m+1)*(m+1) and 1/(n+1)*(n+1) */
} UserCtx;

extern PetscErrorCode UserInitializeLinearSolver(PetscInt, PetscInt, UserCtx *);
extern PetscErrorCode UserFinalizeLinearSolver(UserCtx *);
extern PetscErrorCode UserDoLinearSolver(PetscScalar *, UserCtx *userctx, PetscScalar *b, PetscScalar *x);

int main(int argc, char **args)
{
  UserCtx      userctx;
  PetscInt     m = 6, n = 7, t, tmax = 2, i, Ii, j, N;
  PetscScalar *userx, *rho, *solution, *userb, hx, hy, x, y;
  PetscReal    enorm;

  /*
     Initialize the PETSc libraries
  */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  /*
     The next two lines are for testing only; these allow the user to
     decide the grid size at runtime.
  */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

  /*
     Create the empty sparse matrix and linear solver data structures
  */
  PetscCall(UserInitializeLinearSolver(m, n, &userctx));
  N = m * n;

  /*
     Allocate arrays to hold the solution to the linear system.
     This is not normally done in PETSc programs, but in this case,
     since we are calling these routines from a non-PETSc program, we
     would like to reuse the data structures from another code. So in
     the context of a larger application these would be provided by
     other (non-PETSc) parts of the application code.
  */
  PetscCall(PetscMalloc1(N, &userx));
  PetscCall(PetscMalloc1(N, &userb));
  PetscCall(PetscMalloc1(N, &solution));

  /*
      Allocate an array to hold the coefficients in the elliptic operator
  */
  PetscCall(PetscMalloc1(N, &rho));

  /*
     Fill up the array rho[] with the function rho(x,y) = x; fill the
     right-hand-side b[] and the solution with a known problem for testing.
  */
  hx = 1.0 / (m + 1);
  hy = 1.0 / (n + 1);
  y  = hy;
  Ii = 0;
  for (j = 0; j < n; j++) {
    x = hx;
    for (i = 0; i < m; i++) {
      rho[Ii]      = x;
      solution[Ii] = PetscSinScalar(2. * PETSC_PI * x) * PetscSinScalar(2. * PETSC_PI * y);
      userb[Ii]    = -2 * PETSC_PI * PetscCosScalar(2 * PETSC_PI * x) * PetscSinScalar(2 * PETSC_PI * y) + 8 * PETSC_PI * PETSC_PI * x * PetscSinScalar(2 * PETSC_PI * x) * PetscSinScalar(2 * PETSC_PI * y);
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
  for (t = 0; t < tmax; t++) {
    PetscCall(UserDoLinearSolver(rho, &userctx, userb, userx));

    /*
        Compute error: Note that this could (and usually should) all be done
        using the PETSc vector operations. Here we demonstrate using more
        standard programming practices to show how they may be mixed with
        PETSc.
    */
    enorm = 0.0;
    for (i = 0; i < N; i++) enorm += PetscRealPart(PetscConj(solution[i] - userx[i]) * (solution[i] - userx[i]));
    enorm *= PetscRealPart(hx * hy);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "m %" PetscInt_FMT " n %" PetscInt_FMT " error norm %g\n", m, n, (double)enorm));
  }

  /*
     We are all finished solving linear systems, so we clean up the
     data structures.
  */
  PetscCall(PetscFree(rho));
  PetscCall(PetscFree(solution));
  PetscCall(PetscFree(userx));
  PetscCall(PetscFree(userb));
  PetscCall(UserFinalizeLinearSolver(&userctx));
  PetscCall(PetscFinalize());
  return 0;
}

/* ------------------------------------------------------------------------*/
PetscErrorCode UserInitializeLinearSolver(PetscInt m, PetscInt n, UserCtx *userctx)
{
  PetscInt N;

  PetscFunctionBeginUser;
  /*
     Here we assume use of a grid of size m x n, with all points on the
     interior of the domain, i.e., we do not include the points corresponding
     to homogeneous Dirichlet boundary conditions.  We assume that the domain
     is [0,1]x[0,1].
  */
  userctx->m   = m;
  userctx->n   = n;
  userctx->hx2 = (m + 1) * (m + 1);
  userctx->hy2 = (n + 1) * (n + 1);
  N            = m * n;

  /*
     Create the sparse matrix. Preallocate 5 nonzeros per row.
  */
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, N, N, 5, 0, &userctx->A));

  /*
     Create vectors. Here we create vectors with no memory allocated.
     This way, we can use the data structures already in the program
     by using VecPlaceArray() subroutine at a later stage.
  */
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, N, NULL, &userctx->b));
  PetscCall(VecDuplicate(userctx->b, &userctx->x));

  /*
     Create linear solver context. This will be used repeatedly for all
     the linear solves needed.
  */
  PetscCall(KSPCreate(PETSC_COMM_SELF, &userctx->ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Solves -div (rho grad psi) = F using finite differences.
   rho is a 2-dimensional array of size m by n, stored in Fortran
   style by columns. userb is a standard one-dimensional array.
*/
/* ------------------------------------------------------------------------*/
PetscErrorCode UserDoLinearSolver(PetscScalar *rho, UserCtx *userctx, PetscScalar *userb, PetscScalar *userx)
{
  PetscInt    i, j, Ii, J, m = userctx->m, n = userctx->n;
  Mat         A = userctx->A;
  PC          pc;
  PetscScalar v, hx2 = userctx->hx2, hy2 = userctx->hy2;

  PetscFunctionBeginUser;
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
  for (j = 0; j < n; j++) {
    for (i = 0; i < m; i++) {
      if (j > 0) {
        J = Ii - m;
        v = -.5 * (rho[Ii] + rho[J]) * hy2;
        PetscCall(MatSetValues(A, 1, &Ii, 1, &J, &v, INSERT_VALUES));
      }
      if (j < n - 1) {
        J = Ii + m;
        v = -.5 * (rho[Ii] + rho[J]) * hy2;
        PetscCall(MatSetValues(A, 1, &Ii, 1, &J, &v, INSERT_VALUES));
      }
      if (i > 0) {
        J = Ii - 1;
        v = -.5 * (rho[Ii] + rho[J]) * hx2;
        PetscCall(MatSetValues(A, 1, &Ii, 1, &J, &v, INSERT_VALUES));
      }
      if (i < m - 1) {
        J = Ii + 1;
        v = -.5 * (rho[Ii] + rho[J]) * hx2;
        PetscCall(MatSetValues(A, 1, &Ii, 1, &J, &v, INSERT_VALUES));
      }
      v = 2.0 * rho[Ii] * (hx2 + hy2);
      PetscCall(MatSetValues(A, 1, &Ii, 1, &Ii, &v, INSERT_VALUES));
      Ii++;
    }
  }

  /*
     Assemble matrix
  */
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  /*
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix. Since all the matrices
     will have the same nonzero pattern here, we indicate this so the
     linear solvers can take advantage of this.
  */
  PetscCall(KSPSetOperators(userctx->ksp, A, A));

  /*
     Set linear solver defaults for this problem (optional).
     - Here we set it to use direct LU factorization for the solution
  */
  PetscCall(KSPGetPC(userctx->ksp, &pc));
  PetscCall(PCSetType(pc, PCLU));

  /*
     Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
     These options will override those specified above as long as
     KSPSetFromOptions() is called _after_ any other customization
     routines.

     Run the program with the option -help to see all the possible
     linear solver options.
  */
  PetscCall(KSPSetFromOptions(userctx->ksp));

  /*
     This allows the PETSc linear solvers to compute the solution
     directly in the user's array rather than in the PETSc vector.

     This is essentially a hack and not highly recommend unless you
     are quite comfortable with using PETSc. In general, users should
     write their entire application using PETSc vectors rather than
     arrays.
  */
  PetscCall(VecPlaceArray(userctx->x, userx));
  PetscCall(VecPlaceArray(userctx->b, userb));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(KSPSolve(userctx->ksp, userctx->b, userctx->x));

  /*
    Put back the PETSc array that belongs in the vector xuserctx->x
  */
  PetscCall(VecResetArray(userctx->x));
  PetscCall(VecResetArray(userctx->b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------------------------------------------------------*/
PetscErrorCode UserFinalizeLinearSolver(UserCtx *userctx)
{
  /*
     We are all done and don't need to solve any more linear systems, so
     we free the work space.  All PETSc objects should be destroyed when
     they are no longer needed.
  */
  PetscFunctionBeginUser;
  PetscCall(KSPDestroy(&userctx->ksp));
  PetscCall(VecDestroy(&userctx->x));
  PetscCall(VecDestroy(&userctx->b));
  PetscCall(MatDestroy(&userctx->A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   test:
      args: -m 19 -n 20 -ksp_gmres_cgs_refinement_type refine_always

TEST*/
