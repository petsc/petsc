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

typedef enum {
  RUN_FULL,
  RUN_TEST
} RunType;

typedef struct {
  RunType runType; /* Whether to run tests, or solve the full problem */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  const char *runTypes[2] = {"full", "test"};
  PetscInt    run;

  PetscFunctionBeginUser;
  options->runType = RUN_FULL;
  PetscOptionsBegin(comm, "", "Inverse Problem Options", "DMPLEX");
  run = options->runType;
  PetscCall(PetscOptionsEList("-run_type", "The run type", "ex1.c", runTypes, 2, runTypes[options->runType], &run, NULL));
  options->runType = (RunType)run;
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* u - (x^2 + y^2) */
void f0_u(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = u[0] - (x[0] * x[0] + x[1] * x[1]);
}
/* a \nabla\lambda */
void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u[1] * u_x[dim * 2 + d];
}
/* I */
void g0_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = 1.0;
}
/* \nabla */
void g2_ua(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g2[d] = u_x[dim * 2 + d];
}
/* a */
void g3_ul(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d * dim + d] = u[1];
}
/* a - (x + y) */
void f0_a(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = u[1] - (x[0] + x[1]);
}
/* \lambda \nabla u */
void f1_a(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u[2] * u_x[d];
}
/* I */
void g0_aa(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = 1.0;
}
/* 6 (x + y) */
void f0_l(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = 6.0 * (x[0] + x[1]);
}
/* a \nabla u */
void f1_l(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u[1] * u_x[d];
}
/* \nabla u */
void g2_la(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g2[d] = u_x[d];
}
/* a */
void g3_lu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d * dim + d] = u[1];
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
  *u = x[0] * x[0] + x[1] * x[1];
  return PETSC_SUCCESS;
}
PetscErrorCode linear_a_2d(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt Nf, PetscScalar *a, void *ctx)
{
  *a = x[0] + x[1];
  return PETSC_SUCCESS;
}
PetscErrorCode zero(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt Nf, PetscScalar *l, void *ctx)
{
  *l = 0.0;
  return PETSC_SUCCESS;
}

PetscErrorCode SetupProblem(DM dm, AppCtx *user)
{
  PetscDS        ds;
  DMLabel        label;
  const PetscInt id = 1;

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetResidual(ds, 0, f0_u, f1_u));
  PetscCall(PetscDSSetResidual(ds, 1, f0_a, f1_a));
  PetscCall(PetscDSSetResidual(ds, 2, f0_l, f1_l));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, g0_uu, NULL, NULL, NULL));
  PetscCall(PetscDSSetJacobian(ds, 0, 1, NULL, NULL, g2_ua, NULL));
  PetscCall(PetscDSSetJacobian(ds, 0, 2, NULL, NULL, NULL, g3_ul));
  PetscCall(PetscDSSetJacobian(ds, 1, 1, g0_aa, NULL, NULL, NULL));
  PetscCall(PetscDSSetJacobian(ds, 2, 1, NULL, NULL, g2_la, NULL));
  PetscCall(PetscDSSetJacobian(ds, 2, 0, NULL, NULL, NULL, g3_lu));

  PetscCall(PetscDSSetExactSolution(ds, 0, quadratic_u_2d, NULL));
  PetscCall(PetscDSSetExactSolution(ds, 1, linear_a_2d, NULL));
  PetscCall(PetscDSSetExactSolution(ds, 2, zero, NULL));
  PetscCall(DMGetLabel(dm, "marker", &label));
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void))quadratic_u_2d, NULL, user, NULL));
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 1, 0, NULL, (void (*)(void))linear_a_2d, NULL, user, NULL));
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 2, 0, NULL, (void (*)(void))zero, NULL, user, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM             cdm = dm;
  const PetscInt dim = 2;
  PetscFE        fe[3];
  PetscInt       f;
  MPI_Comm       comm;

  PetscFunctionBeginUser;
  /* Create finite element */
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(PetscFECreateDefault(comm, dim, 1, PETSC_TRUE, "potential_", -1, &fe[0]));
  PetscCall(PetscObjectSetName((PetscObject)fe[0], "potential"));
  PetscCall(PetscFECreateDefault(comm, dim, 1, PETSC_TRUE, "conductivity_", -1, &fe[1]));
  PetscCall(PetscObjectSetName((PetscObject)fe[1], "conductivity"));
  PetscCall(PetscFECopyQuadrature(fe[0], fe[1]));
  PetscCall(PetscFECreateDefault(comm, dim, 1, PETSC_TRUE, "multiplier_", -1, &fe[2]));
  PetscCall(PetscObjectSetName((PetscObject)fe[2], "multiplier"));
  PetscCall(PetscFECopyQuadrature(fe[0], fe[2]));
  /* Set discretization and boundary conditions for each mesh */
  for (f = 0; f < 3; ++f) PetscCall(DMSetField(dm, f, NULL, (PetscObject)fe[f]));
  PetscCall(DMCreateDS(dm));
  PetscCall(SetupProblem(dm, user));
  while (cdm) {
    PetscCall(DMCopyDisc(dm, cdm));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  for (f = 0; f < 3; ++f) PetscCall(PetscFEDestroy(&fe[f]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM     dm;
  SNES   snes;
  Vec    u, r;
  AppCtx user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(SetupDiscretization(dm, &user));

  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(PetscObjectSetName((PetscObject)u, "solution"));
  PetscCall(VecDuplicate(u, &r));
  PetscCall(DMPlexSetSNESLocalFEM(dm, &user, &user, &user));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(DMSNESCheckFromOptions(snes, u));
  if (user.runType == RUN_FULL) {
    PetscDS ds;
    PetscErrorCode (*exactFuncs[3])(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
    PetscErrorCode (*initialGuess[3])(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx);
    PetscReal error;

    PetscCall(DMGetDS(dm, &ds));
    PetscCall(PetscDSGetExactSolution(ds, 0, &exactFuncs[0], NULL));
    PetscCall(PetscDSGetExactSolution(ds, 1, &exactFuncs[1], NULL));
    PetscCall(PetscDSGetExactSolution(ds, 2, &exactFuncs[2], NULL));
    initialGuess[0] = zero;
    initialGuess[1] = zero;
    initialGuess[2] = zero;
    PetscCall(DMProjectFunction(dm, 0.0, initialGuess, NULL, INSERT_VALUES, u));
    PetscCall(VecViewFromOptions(u, NULL, "-initial_vec_view"));
    PetscCall(DMComputeL2Diff(dm, 0.0, exactFuncs, NULL, u, &error));
    if (error < 1.0e-11) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Initial L_2 Error: < 1.0e-11\n"));
    else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Initial L_2 Error: %g\n", (double)error));
    PetscCall(SNESSolve(snes, NULL, u));
    PetscCall(DMComputeL2Diff(dm, 0.0, exactFuncs, NULL, u, &error));
    if (error < 1.0e-11) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Final L_2 Error: < 1.0e-11\n"));
    else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Final L_2 Error: %g\n", (double)error));
  }
  PetscCall(VecViewFromOptions(u, NULL, "-sol_vec_view"));

  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&r));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
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
