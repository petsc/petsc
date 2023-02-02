
static char help[] = "A Chebyshev spectral method for the compressible Blasius boundary layer equations.\n\n";

/*
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
   Include "petscdt.h" so that we can have support for use of Quadrature formulas
*/
/*F
This examples solves the compressible Blasius boundary layer equations
2(\rho\muf'')' + ff'' = 0
(\rho\muh')' + Prfh' + Pr(\gamma-1)Ma^{2}\rho\muf''^{2} = 0
following Howarth-Dorodnitsyn transformation with boundary conditions
f(0) = f'(0) = 0, f'(\infty) = 1, h(\infty) = 1, h = \theta(0). Where \theta = T/T_{\infty}
Note: density (\rho) and viscosity (\mu) are treated as constants in this example
F*/
#include <petscsnes.h>
#include <petscdt.h>

/*
   User-defined routines
*/

extern PetscErrorCode FormFunction(SNES, Vec, Vec, void *);

typedef struct {
  PetscReal  Ma, Pr, h_0;
  PetscInt   N;
  PetscReal  dx_deta;
  PetscReal *x;
  PetscReal  gamma;
} Blasius;

int main(int argc, char **argv)
{
  SNES        snes; /* nonlinear solver context */
  Vec         x, r; /* solution, residual vectors */
  PetscMPIInt size;
  Blasius    *blasius;
  PetscReal   L, *weight; /* L is size of the domain */

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Example is only for sequential runs");

  // Read command-line arguments
  PetscCall(PetscCalloc1(1, &blasius));
  blasius->Ma    = 2;   /* Mach number */
  blasius->Pr    = 0.7; /* Prandtl number */
  blasius->h_0   = 2.;  /* relative temperature at the wall */
  blasius->N     = 10;  /* Number of Chebyshev terms */
  blasius->gamma = 1.4; /* specific heat ratio */
  L              = 5;
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Compressible Blasius boundary layer equations", "");
  PetscCall(PetscOptionsReal("-mach", "Mach number at freestream", "", blasius->Ma, &blasius->Ma, NULL));
  PetscCall(PetscOptionsReal("-prandtl", "Prandtl number", "", blasius->Pr, &blasius->Pr, NULL));
  PetscCall(PetscOptionsReal("-h_0", "Relative enthalpy at wall", "", blasius->h_0, &blasius->h_0, NULL));
  PetscCall(PetscOptionsReal("-gamma", "Ratio of specific heats", "", blasius->gamma, &blasius->gamma, NULL));
  PetscCall(PetscOptionsInt("-N", "Number of Chebyshev terms for f", "", blasius->N, &blasius->N, NULL));
  PetscCall(PetscOptionsReal("-L", "Extent of the domain", "", L, &L, NULL));
  PetscOptionsEnd();
  blasius->dx_deta = 2 / L; /* this helps to map [-1,1] to [0,L] */
  PetscCall(PetscMalloc2(blasius->N - 3, &blasius->x, blasius->N - 3, &weight));
  PetscCall(PetscDTGaussQuadrature(blasius->N - 3, -1., 1., blasius->x, weight));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix and vector data structures; set corresponding routines
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create vectors for solution and nonlinear function
  */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, 2 * blasius->N - 1));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &r));

  /*
      Set function evaluation routine and vector.
   */
  PetscCall(SNESSetFunction(snes, r, FormFunction, blasius));
  {
    KSP ksp;
    PC  pc;
    PetscCall(SNESGetKSP(snes, &ksp));
    PetscCall(KSPSetType(ksp, KSPPREONLY));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCLU));
  }
  /*
     Set SNES/KSP/KSP/PC runtime options, e.g.,
         -snes_view -snes_monitor -ksp_type <ksp> -pc_type <pc>
     These options will override those specified above as long as
     SNESSetFromOptions() is called _after_ any other customization
     routines.
  */
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(SNESSolve(snes, NULL, x));
  //PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PetscFree2(blasius->x, weight));
  PetscCall(PetscFree(blasius));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&r));
  PetscCall(SNESDestroy(&snes));
  PetscCall(PetscFinalize());
  return 0;
}

/*
   Helper function to evaluate Chebyshev polynomials with a set of coefficients
   with all their derivatives represented as a recurrence table
*/
static void ChebyshevEval(PetscInt N, const PetscScalar *Tf, PetscReal x, PetscReal dx_deta, PetscScalar *f)
{
  PetscScalar table[4][3] = {
    {1, x, 2 * x * x - 1},
    {0, 1, 4 * x        },
    {0, 0, 4            },
    {0, 0, 0            }  /* Chebyshev polynomials T_0, T_1, T_2 of the first kind in (-1,1)  */
  };
  for (int i = 0; i < 4; i++) { f[i] = table[i][0] * Tf[0] + table[i][1] * Tf[1] + table[i][2] * Tf[2]; /* i-th derivative of f */ }
  for (int i = 3; i < N; i++) {
    table[0][i % 3] = 2 * x * table[0][(i - 1) % 3] - table[0][(i - 2) % 3]; /* T_n(x) = 2xT_{n-1}(x) - T_{n-2}(x) */
    /* Differentiate Chebyshev polynomials with the recurrence relation */
    for (int j = 1; j < 4; j++) { table[j][i % 3] = i * (2 * table[j - 1][(i - 1) % 3] + table[j][(i - 2) % 3] / (i - 2)); /* T'_{n}(x)/n = 2T_{n-1}(x) + T'_{n-2}(x)/n-2 */ }
    for (int j = 0; j < 4; j++) f[j] += table[j][i % 3] * Tf[i];
  }
  for (int i = 1; i < 4; i++) {
    for (int j = 0; j < i; j++) f[i] *= dx_deta; /* Here happens the physics of the problem */
  }
}

/*
   FormFunction - Evaluates nonlinear function, F(x).

   Input Parameters:
.  snes - the SNES context
.  X    - input vector
.  ctx  - optional user-defined context

   Output Parameter:
.  R - function vector
 */
PetscErrorCode FormFunction(SNES snes, Vec X, Vec R, void *ctx)
{
  Blasius           *blasius = (Blasius *)ctx;
  const PetscScalar *Tf, *Th; /* Tf and Th are Chebyshev coefficients */
  PetscScalar       *r, f[4], h[4];
  PetscInt           N  = blasius->N;
  PetscReal          Ma = blasius->Ma, Pr = blasius->Pr;

  PetscFunctionBeginUser;
  /*
   Get pointers to vector data.
      - For default PETSc vectors, VecGetArray() returns a pointer to
        the data array.  Otherwise, the routine is implementation dependent.
      - You MUST call VecRestoreArray() when you no longer need access to
        the array.
   */
  PetscCall(VecGetArrayRead(X, &Tf));
  Th = Tf + N;
  PetscCall(VecGetArray(R, &r));

  /* Compute function */
  ChebyshevEval(N, Tf, -1., blasius->dx_deta, f);
  r[0] = f[0];
  r[1] = f[1];
  ChebyshevEval(N, Tf, 1., blasius->dx_deta, f);
  r[2] = f[1] - 1; /* Right end boundary condition */
  for (int i = 0; i < N - 3; i++) {
    ChebyshevEval(N, Tf, blasius->x[i], blasius->dx_deta, f);
    r[3 + i] = 2 * f[3] + f[2] * f[0];
    ChebyshevEval(N - 1, Th, blasius->x[i], blasius->dx_deta, h);
    r[N + 2 + i] = h[2] + Pr * f[0] * h[1] + Pr * (blasius->gamma - 1) * PetscSqr(Ma * f[2]);
  }
  ChebyshevEval(N - 1, Th, -1., blasius->dx_deta, h);
  r[N] = h[0] - blasius->h_0; /* Left end boundary condition */
  ChebyshevEval(N - 1, Th, 1., blasius->dx_deta, h);
  r[N + 1] = h[0] - 1; /* Left end boundary condition */

  /* Restore vectors */
  PetscCall(VecRestoreArrayRead(X, &Tf));
  PetscCall(VecRestoreArray(R, &r));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   test:
      args: -snes_monitor -pc_type svd
      requires: !single

TEST*/
