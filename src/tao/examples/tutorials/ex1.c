static char help[] = "One-Shot Multigrid for Parameter Estimation Problem for the Poisson Equation.\n\
Using the Interior Point Method.\n\n\n";

/*F
  We are solving the parameter estimation problem for the Laplacian. We will ask to minimize a Lagrangian
function over $y$ and $u$, given by
\begin{align}
  L(u, a, \lambda) = \frac{1}{2} || Qu - d ||^2 + \frac{1}{2} || L (u - u_r) ||^2 + \lambda F(u; a)
\end{align}
where $Q$ is a sampling operator, $L$ is a regularization operator, $F$ defines the PDE.

Currently, we have perfect information, meaning $Q = I$, and then we need no regularization, $L = I$. We
also give the exact control for the reference $u_r$.

F*/

#include <petsc.h>
#include <petscfe.h>

PetscInt spatialDim = 0;

typedef enum {RUN_FULL, RUN_TEST, RUN_PERF} RunType;

typedef struct {
  PetscFEM       fem;               /* REQUIRED to use DMPlexComputeResidualFEM() */
  PetscMPIInt    rank;              /* The process rank */
  PetscMPIInt    numProcs;          /* The number of processes */
  RunType        runType;           /* Whether to run tests, or solve the full problem */
  /* Domain and mesh definition */
  PetscInt       dim;               /* The topological mesh dimension */
  /* Finite Element definition */
  PetscFE       *fe;
  /* Problem definition */
  void (*f0Funcs[3])(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f0[]);
  void (*f1Funcs[3])(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f1[]);
  void (*g0Funcs[9])(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g0[]);
  void (*g1Funcs[9])(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g1[]);
  void (*g2Funcs[9])(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g2[]);
  void (*g3Funcs[9])(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g3[]);
  void (*exactFuncs[3])(const PetscReal x[], PetscScalar *u, void *ctx);
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  const char    *runTypes[3] = {"full", "test", "perf"};
  PetscInt       run;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->runType = RUN_FULL;
  options->dim     = 2;

  options->fem.f0Funcs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[])) &options->f0Funcs;
  options->fem.f1Funcs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[])) &options->f1Funcs;
  options->fem.g0Funcs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[])) &options->g0Funcs;
  options->fem.g1Funcs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[])) &options->g1Funcs;
  options->fem.g2Funcs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[])) &options->g2Funcs;
  options->fem.g3Funcs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[])) &options->g3Funcs;

  ierr = MPI_Comm_size(comm, &options->numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &options->rank);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(comm, "", "Poisson Problem Options", "DMPLEX");CHKERRQ(ierr);
  run  = options->runType;
  ierr = PetscOptionsEList("-run_type", "The run type", "magma.c", runTypes, 3, runTypes[options->runType], &run, NULL);CHKERRQ(ierr);
  options->runType = (RunType) run;
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "magma.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  spatialDim = options->dim;
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  DM             distributedMesh = NULL;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexCreateBoxMesh(comm, user->dim, PETSC_TRUE, dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMPlexDistribute(*dm, NULL, 0, NULL, &distributedMesh);CHKERRQ(ierr);
  if (distributedMesh) {
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = distributedMesh;
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

void f0_u(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f0[])
{
  f0[0] = u[0] - (x[0]*x[0] + x[1]*x[1]);
}
void f1_u(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < spatialDim; ++d) f1[d] = u[1]*gradU[spatialDim*2+d];
}
void g0_uu(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g0[])
{
  g0[0] = 1.0;
}
void g2_ua(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g2[])
{
  PetscInt d;
  for (d = 0; d < spatialDim; ++d) g2[d] = gradU[spatialDim*2+d];
}
void g3_ul(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < spatialDim; ++d) g3[d*spatialDim+d] = u[1];
}

void f0_a(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f0[])
{
  f0[0] = u[1] - (x[0] + x[1]);
}
void f1_a(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < spatialDim; ++d) f1[d] = u[2]*gradU[d];
}
void g0_aa(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g0[])
{
  g0[0] = 1.0;
}

void f0_l(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f0[])
{
  f0[0] = 6.0*(x[0] + x[1]);
}
void f1_l(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < spatialDim; ++d) f1[d] = u[1]*gradU[d];
}
void g2_la(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g2[])
{
  PetscInt d;
  for (d = 0; d < spatialDim; ++d) g2[d] = gradU[d];
}
void g3_lu(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < spatialDim; ++d) g3[d*spatialDim+d] = u[1];
}

/*
  In 2D for Dirichlet conditions with a variable coefficient, we use exact solution:

    u  = x^2 + y^2
    f  = 6 (x + y)
    kappa(a) = a = (x + y)

  so that

    -\div \kappa(a) \grad u + f = -6 (x + y) + 6 (x + y) = 0
*/
void quadratic_u_2d(const PetscReal x[], PetscScalar *u, void *ctx)
{
  *u = x[0]*x[0] + x[1]*x[1];
}
void linear_a_2d(const PetscReal x[], PetscScalar *a, void *ctx)
{
  *a = x[0] + x[1];
}
void zero(const PetscReal x[], PetscScalar *u, void *ctx)
{
  *u = 0.0;
}

#undef __FUNCT__
#define __FUNCT__ "SetupProblem"
static PetscErrorCode SetupProblem(DM dm, AppCtx *user)
{
  PetscFEM *fem = &user->fem;
  PetscInt  f, g;

  PetscFunctionBeginUser;
  for (f = 0; f < 3; ++f) {
    fem->f0Funcs[f] = fem->f1Funcs[f] = NULL;
    for (g = 0; g < 3; ++g) fem->g0Funcs[f*3+g] = fem->g1Funcs[f*3+g] = fem->g2Funcs[f*3+g] = fem->g3Funcs[f*3+g] = NULL;
  }
  fem->f0Funcs[0] = f0_u;
  fem->f0Funcs[1] = f0_a;
  fem->f0Funcs[2] = f0_l;
  fem->f1Funcs[0] = f1_u;
  fem->f1Funcs[1] = f1_a;
  fem->f1Funcs[2] = f1_l;
  fem->g0Funcs[0*3+0] = g0_uu;
  fem->g2Funcs[0*3+1] = g2_ua;
  fem->g3Funcs[0*3+2] = g3_ul;
  fem->g0Funcs[1*3+1] = g0_aa;
  fem->g2Funcs[2*3+1] = g2_la;
  fem->g3Funcs[2*3+0] = g3_lu;
  switch (user->dim) {
  case 2:
    user->exactFuncs[0] = quadratic_u_2d;
    user->exactFuncs[1] = linear_a_2d;
    user->exactFuncs[2] = zero;
    break;
  default:
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension %d", user->dim);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupSection"
static PetscErrorCode SetupSection(DM dm, PetscInt numFields, PetscFE fe[], AppCtx *user)
{
  DMLabel        label;
  PetscInt       id = 1;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectSetName((PetscObject) user->fe[0], "potential");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) user->fe[1], "conductivity");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) user->fe[2], "multiplier");CHKERRQ(ierr);
  ierr = DMSetNumFields(dm, 3);CHKERRQ(ierr);
  ierr = DMSetField(dm, 0, (PetscObject) user->fe[0]);CHKERRQ(ierr);
  ierr = DMSetField(dm, 1, (PetscObject) user->fe[1]);CHKERRQ(ierr);
  ierr = DMSetField(dm, 2, (PetscObject) user->fe[2]);CHKERRQ(ierr);
  ierr = DMPlexGetLabel(dm, "marker", &label);CHKERRQ(ierr);
  if (label) {
    ierr = DMPlexAddBoundary(dm, PETSC_TRUE, "marker", 0, 0, NULL, (void (*)()) user->exactFuncs[0], 1, &id, user);CHKERRQ(ierr);
    ierr = DMPlexAddBoundary(dm, PETSC_TRUE, "marker", 1, 0, NULL, (void (*)()) user->exactFuncs[1], 1, &id, user);CHKERRQ(ierr);
    ierr = DMPlexAddBoundary(dm, PETSC_TRUE, "marker", 2, 0, NULL, (void (*)()) user->exactFuncs[2], 1, &id, user);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  DM             dm;
  SNES           snes;
  PetscSpace     P;
  Mat            J, M;
  Vec            u, r;
  PetscInt       order;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);

  ierr = PetscMalloc1(3,&user.fe);CHKERRQ(ierr);
  user.fem.fe    = user.fe;
  user.fem.feBd  = NULL;
  user.fem.feAux = NULL;
  ierr = PetscFECreateDefault(dm, user.dim, 1, PETSC_TRUE, "potential_", -1, &user.fe[0]);CHKERRQ(ierr);
  ierr = PetscFEGetBasisSpace(user.fe[0], &P);CHKERRQ(ierr);
  ierr = PetscSpaceGetOrder(P, &order);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dm, user.dim, 1, PETSC_TRUE, "conductivity_", order, &user.fe[1]);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dm, user.dim, 1, PETSC_TRUE, "multiplier_", order, &user.fe[2]);CHKERRQ(ierr);
  ierr = SetupProblem(dm, &user);CHKERRQ(ierr);
  ierr = SetupSection(dm, 3, user.fe, &user);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "solution");CHKERRQ(ierr);
  ierr = VecDuplicate(u, &r);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm, &J);CHKERRQ(ierr);
  M    = J;
  ierr = DMSNESSetFunctionLocal(dm, DMPlexComputeResidualFEM, &user);CHKERRQ(ierr);
  ierr = DMSNESSetJacobianLocal(dm, DMPlexComputeJacobianFEM, &user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = DMPlexProjectFunction(dm, user.fe, user.exactFuncs, NULL, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
  if (user.runType == RUN_FULL) {
    void    (*initialGuess[3])(const PetscReal x[], PetscScalar *, void *ctx);
    PetscReal error;

    initialGuess[0] = zero;
    initialGuess[1] = zero;
    initialGuess[2] = zero;
    ierr = DMPlexProjectFunction(dm, user.fe, initialGuess, NULL, INSERT_VALUES, u);CHKERRQ(ierr);
    ierr = VecViewFromOptions(u, NULL, "-initial_sol_view");CHKERRQ(ierr);
    ierr = DMPlexComputeL2Diff(dm, user.fe, user.exactFuncs, NULL, u, &error);CHKERRQ(ierr);
    if (error < 1.0e-11) {ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial L_2 Error: < 1.0e-11\n");CHKERRQ(ierr);}
    else                 {ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial L_2 Error: %g\n", error);CHKERRQ(ierr);}
    ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
    ierr = DMPlexComputeL2Diff(dm, user.fe, user.exactFuncs, NULL, u, &error);CHKERRQ(ierr);
    if (error < 1.0e-11) {ierr = PetscPrintf(PETSC_COMM_WORLD, "Final L_2 Error: < 1.0e-11\n");CHKERRQ(ierr);}
    else                 {ierr = PetscPrintf(PETSC_COMM_WORLD, "Final L_2 Error: %g\n", error);CHKERRQ(ierr);}
  } else {
    PetscReal error = 0.0, res = 0.0;

    /* Check discretization error */
    ierr = VecViewFromOptions(u, NULL, "-initial_sol_view");CHKERRQ(ierr);
    ierr = DMPlexComputeL2Diff(dm, user.fe, user.exactFuncs, NULL, u, &error);CHKERRQ(ierr);
    if (error >= 1.0e-11) {ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: %g\n", error);CHKERRQ(ierr);}
    else                  {ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: < 1.0e-11\n", error);CHKERRQ(ierr);}
    /* Check residual */
    ierr = SNESComputeFunction(snes, u, r);CHKERRQ(ierr);
    ierr = VecChop(r, 1.0e-10);CHKERRQ(ierr);
    ierr = VecViewFromOptions(r, NULL, "-initial_res_view");CHKERRQ(ierr);
    ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Residual: %g\n", res);CHKERRQ(ierr);
    /* Check Jacobian */
    {
      Vec b;

      ierr = SNESComputeJacobian(snes, u, M, M);CHKERRQ(ierr);
      ierr = VecDuplicate(u, &b);CHKERRQ(ierr);
      ierr = VecSet(r, 0.0);CHKERRQ(ierr);
      ierr = SNESComputeFunction(snes, r, b);CHKERRQ(ierr);
      ierr = MatMult(M, u, r);CHKERRQ(ierr);
      ierr = VecAXPY(r, 1.0, b);CHKERRQ(ierr);
      ierr = VecDestroy(&b);CHKERRQ(ierr);
      ierr = VecChop(r, 1.0e-10);CHKERRQ(ierr);
      ierr = VecViewFromOptions(r, NULL, "-initial_res_view");CHKERRQ(ierr);
      ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Linear L_2 Residual: %g\n", res);CHKERRQ(ierr);
    }
  }
  ierr = VecViewFromOptions(u, "sol_", "-vec_view");CHKERRQ(ierr);

  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&user.fe[0]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&user.fe[1]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&user.fe[2]);CHKERRQ(ierr);
  ierr = PetscFree(user.fe);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
