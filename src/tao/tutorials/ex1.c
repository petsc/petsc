static char help[] = "One-Shot Multigrid for Parameter Estimation Problem for the Poisson Equation.\n\
Using the Interior Point Method.\n\n\n";

/*F
  We are solving the parameter estimation problem for the Laplacian. We will ask to minimize a Lagrangian
function over $a$ and $u$, given by
\begin{align}
  L(u, a, \lambda) = \frac{1}{2} || Qu - d ||^2 + \frac{1}{2} || L (a - a_r) ||^2 + \lambda F(u; a)
\end{align}
where $Q$ is a sampling operator, $L$ is a regularization operator, $F$ defines the PDE.

Currently, we have perfect information, meaning $Q = I$, and then we need no regularization, $L = I$. We
also give the exact control for the reference $a_r$.

The PDE will be the Laplace equation with homogeneous boundary conditions
\begin{align}
  -nabla \cdot a \nabla u = f
\end{align}

F*/

#include <petsc.h>
#include <petscfe.h>

typedef enum {RUN_FULL, RUN_TEST} RunType;

typedef struct {
  RunType runType;  /* Whether to run tests, or solve the full problem */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  const char    *runTypes[2] = {"full", "test"};
  PetscInt       run;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->runType = RUN_FULL;

  ierr = PetscOptionsBegin(comm, "", "Inverse Problem Options", "DMPLEX");CHKERRQ(ierr);
  run  = options->runType;
  CHKERRQ(PetscOptionsEList("-run_type", "The run type", "ex1.c", runTypes, 2, runTypes[options->runType], &run, NULL));
  options->runType = (RunType) run;
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

/* u - (x^2 + y^2) */
void f0_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = u[0] - (x[0]*x[0] + x[1]*x[1]);
}
/* a \nabla\lambda */
void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u[1]*u_x[dim*2+d];
}
/* I */
void g0_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = 1.0;
}
/* \nabla */
void g2_ua(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g2[d] = u_x[dim*2+d];
}
/* a */
void g3_ul(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d*dim+d] = u[1];
}
/* a - (x + y) */
void f0_a(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = u[1] - (x[0] + x[1]);
}
/* \lambda \nabla u */
void f1_a(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u[2]*u_x[d];
}
/* I */
void g0_aa(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = 1.0;
}
/* 6 (x + y) */
void f0_l(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = 6.0*(x[0] + x[1]);
}
/* a \nabla u */
void f1_l(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u[1]*u_x[d];
}
/* \nabla u */
void g2_la(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g2[d] = u_x[d];
}
/* a */
void g3_lu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d*dim+d] = u[1];
}

/*
  In 2D for Dirichlet conditions with a variable coefficient, we use exact solution:

    u  = x^2 + y^2
    f  = 6 (x + y)
    kappa(a) = a = (x + y)

  so that

    -\div \kappa(a) \grad u + f = -6 (x + y) + 6 (x + y) = 0
*/
PetscErrorCode quadratic_u_2d(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  *u = x[0]*x[0] + x[1]*x[1];
  return 0;
}
PetscErrorCode linear_a_2d(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt Nf, PetscScalar *a, void *ctx)
{
  *a = x[0] + x[1];
  return 0;
}
PetscErrorCode zero(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt Nf, PetscScalar *l, void *ctx)
{
  *l = 0.0;
  return 0;
}

PetscErrorCode SetupProblem(DM dm, AppCtx *user)
{
  PetscDS        ds;
  DMLabel        label;
  const PetscInt id = 1;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetDS(dm, &ds));
  CHKERRQ(PetscDSSetResidual(ds, 0, f0_u, f1_u));
  CHKERRQ(PetscDSSetResidual(ds, 1, f0_a, f1_a));
  CHKERRQ(PetscDSSetResidual(ds, 2, f0_l, f1_l));
  CHKERRQ(PetscDSSetJacobian(ds, 0, 0, g0_uu, NULL, NULL, NULL));
  CHKERRQ(PetscDSSetJacobian(ds, 0, 1, NULL, NULL, g2_ua, NULL));
  CHKERRQ(PetscDSSetJacobian(ds, 0, 2, NULL, NULL, NULL, g3_ul));
  CHKERRQ(PetscDSSetJacobian(ds, 1, 1, g0_aa, NULL, NULL, NULL));
  CHKERRQ(PetscDSSetJacobian(ds, 2, 1, NULL, NULL, g2_la, NULL));
  CHKERRQ(PetscDSSetJacobian(ds, 2, 0, NULL, NULL, NULL, g3_lu));

  CHKERRQ(PetscDSSetExactSolution(ds, 0, quadratic_u_2d, NULL));
  CHKERRQ(PetscDSSetExactSolution(ds, 1, linear_a_2d, NULL));
  CHKERRQ(PetscDSSetExactSolution(ds, 2, zero, NULL));
  CHKERRQ(DMGetLabel(dm, "marker", &label));
  CHKERRQ(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) quadratic_u_2d, NULL, user, NULL));
  CHKERRQ(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 1, 0, NULL, (void (*)(void)) linear_a_2d, NULL, user, NULL));
  CHKERRQ(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 2, 0, NULL, (void (*)(void)) zero, NULL, user, NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM              cdm = dm;
  const PetscInt  dim = 2;
  PetscFE         fe[3];
  PetscInt        f;
  MPI_Comm        comm;

  PetscFunctionBeginUser;
  /* Create finite element */
  CHKERRQ(PetscObjectGetComm((PetscObject) dm, &comm));
  CHKERRQ(PetscFECreateDefault(comm, dim, 1, PETSC_TRUE, "potential_", -1, &fe[0]));
  CHKERRQ(PetscObjectSetName((PetscObject) fe[0], "potential"));
  CHKERRQ(PetscFECreateDefault(comm, dim, 1, PETSC_TRUE, "conductivity_", -1, &fe[1]));
  CHKERRQ(PetscObjectSetName((PetscObject) fe[1], "conductivity"));
  CHKERRQ(PetscFECopyQuadrature(fe[0], fe[1]));
  CHKERRQ(PetscFECreateDefault(comm, dim, 1, PETSC_TRUE, "multiplier_", -1, &fe[2]));
  CHKERRQ(PetscObjectSetName((PetscObject) fe[2], "multiplier"));
  CHKERRQ(PetscFECopyQuadrature(fe[0], fe[2]));
  /* Set discretization and boundary conditions for each mesh */
  for (f = 0; f < 3; ++f) CHKERRQ(DMSetField(dm, f, NULL, (PetscObject) fe[f]));
  CHKERRQ(DMCreateDS(dm));
  CHKERRQ(SetupProblem(dm, user));
  while (cdm) {
    CHKERRQ(DMCopyDisc(dm, cdm));
    CHKERRQ(DMGetCoarseDM(cdm, &cdm));
  }
  for (f = 0; f < 3; ++f) CHKERRQ(PetscFEDestroy(&fe[f]));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  SNES           snes;
  Vec            u, r;
  AppCtx         user;

  CHKERRQ(PetscInitialize(&argc, &argv, NULL,help));
  CHKERRQ(ProcessOptions(PETSC_COMM_WORLD, &user));
  CHKERRQ(SNESCreate(PETSC_COMM_WORLD, &snes));
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  CHKERRQ(SNESSetDM(snes, dm));
  CHKERRQ(SetupDiscretization(dm, &user));

  CHKERRQ(DMCreateGlobalVector(dm, &u));
  CHKERRQ(PetscObjectSetName((PetscObject) u, "solution"));
  CHKERRQ(VecDuplicate(u, &r));
  CHKERRQ(DMPlexSetSNESLocalFEM(dm,&user,&user,&user));
  CHKERRQ(SNESSetFromOptions(snes));

  CHKERRQ(DMSNESCheckFromOptions(snes, u));
  if (user.runType == RUN_FULL) {
    PetscDS          ds;
    PetscErrorCode (*exactFuncs[3])(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
    PetscErrorCode (*initialGuess[3])(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx);
    PetscReal        error;

    CHKERRQ(DMGetDS(dm, &ds));
    CHKERRQ(PetscDSGetExactSolution(ds, 0, &exactFuncs[0], NULL));
    CHKERRQ(PetscDSGetExactSolution(ds, 1, &exactFuncs[1], NULL));
    CHKERRQ(PetscDSGetExactSolution(ds, 2, &exactFuncs[2], NULL));
    initialGuess[0] = zero;
    initialGuess[1] = zero;
    initialGuess[2] = zero;
    CHKERRQ(DMProjectFunction(dm, 0.0, initialGuess, NULL, INSERT_VALUES, u));
    CHKERRQ(VecViewFromOptions(u, NULL, "-initial_vec_view"));
    CHKERRQ(DMComputeL2Diff(dm, 0.0, exactFuncs, NULL, u, &error));
    if (error < 1.0e-11) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Initial L_2 Error: < 1.0e-11\n"));
    else                 CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Initial L_2 Error: %g\n", error));
    CHKERRQ(SNESSolve(snes, NULL, u));
    CHKERRQ(DMComputeL2Diff(dm, 0.0, exactFuncs, NULL, u, &error));
    if (error < 1.0e-11) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Final L_2 Error: < 1.0e-11\n"));
    else                 CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Final L_2 Error: %g\n", error));
  }
  CHKERRQ(VecViewFromOptions(u, NULL, "-sol_vec_view"));

  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&r));
  CHKERRQ(SNESDestroy(&snes));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: !complex

  test:
    suffix: 0
    requires: triangle
    args: -run_type test -dmsnes_check -potential_petscspace_degree 2 -conductivity_petscspace_degree 1 -multiplier_petscspace_degree 2

  test:
    suffix: 1
    requires: triangle
    args: -potential_petscspace_degree 2 -conductivity_petscspace_degree 1 -multiplier_petscspace_degree 2 -snes_monitor -pc_type fieldsplit -pc_fieldsplit_0_fields 0,1 -pc_fieldsplit_1_fields 2 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition selfp -fieldsplit_0_pc_type lu -fieldsplit_multiplier_ksp_rtol 1.0e-10 -fieldsplit_multiplier_pc_type lu -sol_vec_view

TEST*/
