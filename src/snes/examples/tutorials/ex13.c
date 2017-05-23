static char help[] = "Poisson Problem in 2d and 3d with finite elements.\n\
We solve the Poisson problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\
This example supports automatic convergence estimation\n\
and eventually adaptivity.\n\n\n";

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>

#include <petscblaslapack.h>

typedef struct {
  /* Domain and mesh definition */
  PetscInt  dim;               /* The topological mesh dimension */
  PetscBool simplex;           /* Simplicial mesh */
  PetscInt  cells[3];          /* The initial domain division */
} AppCtx;

static PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt c;
  for (c = 0; c < Nc; ++c) u[c] = 0.0;
  return 0;
}

static PetscErrorCode trig_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;
  *u = 0.0;
  for (d = 0; d < dim; ++d) *u += PetscSinReal(2.0*PETSC_PI*x[d]);
  return 0;
}

static void f0_trig_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f0[0] += -4.0*PetscSqr(PETSC_PI)*PetscSinReal(2.0*PETSC_PI*x[d]);
}

static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u_x[d];
}

static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d*dim+d] = 1.0;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt       n = 3;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim      = 2;
  options->cells[0] = 1;
  options->cells[1] = 1;
  options->cells[2] = 1;
  options->simplex  = PETSC_TRUE;

  ierr = PetscOptionsBegin(comm, "", "Poisson Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex13.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-cells", "The initial mesh division", "ex13.c", options->cells, &n, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Simplicial (true) or tensor (false) mesh", "ex13.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* Create box mesh */
  if (user->simplex) {ierr = DMPlexCreateBoxMesh(comm, user->dim, user->cells[0], PETSC_TRUE, dm);CHKERRQ(ierr);}
  else               {ierr = DMPlexCreateHexBoxMesh(comm, user->dim, user->cells, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, dm);CHKERRQ(ierr);}
  /* Distribute mesh over processes */
  {
    DM               dmDist = NULL;
    PetscPartitioner part;

    ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
    ierr = DMPlexDistribute(*dm, 0, NULL, &dmDist);CHKERRQ(ierr);
    if (dmDist) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = dmDist;
    }
  }
  /* TODO: This should be pulled into the library */
  {
    char      convType[256];
    PetscBool flg;

    ierr = PetscOptionsBegin(comm, "", "Mesh conversion options", "DMPLEX");CHKERRQ(ierr);
    ierr = PetscOptionsFList("-dm_plex_convert_type","Convert DMPlex to another format","ex12",DMList,DMPLEX,convType,256,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();
    if (flg) {
      DM dmConv;

      ierr = DMConvert(*dm,convType,&dmConv);CHKERRQ(ierr);
      if (dmConv) {
        ierr = DMDestroy(dm);CHKERRQ(ierr);
        *dm  = dmConv;
      }
    }
  }
  /* TODO: This should be pulled into the library */
  ierr = DMLocalizeCoordinates(*dm);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMSetApplicationContext(*dm, user);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  /* TODO: Add a hierachical viewer */
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupProblem(PetscDS prob, AppCtx *user)
{
  const PetscInt id = 1;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscDSSetResidual(prob, 0, f0_trig_u, f1_u);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "wall", "marker", 0, 0, NULL, (void (*)()) trig_u, 1, &id, user);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolution(prob, 0, trig_u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe;
  PetscDS        prob;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* Create finite element */
  ierr = PetscFECreateDefault(dm, user->dim, 1, user->simplex, NULL, -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, "potential");CHKERRQ(ierr);
  /* Set discretization and boundary conditions for each mesh */
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe);CHKERRQ(ierr);
  ierr = SetupProblem(prob, user);CHKERRQ(ierr);
  while (cdm) {
    ierr = DMSetDS(cdm, prob);CHKERRQ(ierr);
    /* TODO: Check whether the boundary of coarse meshes is marked */
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

struct _ConvergenceEstimator
{
  MPI_Comm          comm;
  /* Inputs */
  DM                idm;  /* Initial grid */
  SNES              snes; /* Solver */
  PetscInt          Nr;   /* The number of refinements */
  PetscInt          Nf;   /* The number of fields in the DM */
  PetscErrorCode (**initGuess)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);
  PetscErrorCode (**exactSol)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);
  /* Outputs */
  PetscBool  monitor;
  PetscReal *errors;
};
typedef struct _ConvergenceEstimator *ConvEst;

static PetscErrorCode ConvEstCreate(MPI_Comm comm, ConvEst *ce)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCalloc1(1, ce);CHKERRQ(ierr);
  (*ce)->comm    = comm;
  (*ce)->monitor = PETSC_FALSE;
  (*ce)->Nr      = 4;
  PetscFunctionReturn(0);
}

static PetscErrorCode ConvEstDestroy(ConvEst *ce)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree2((*ce)->initGuess, (*ce)->exactSol);CHKERRQ(ierr);
  ierr = PetscFree((*ce)->errors);CHKERRQ(ierr);
  ierr = PetscFree(*ce);CHKERRQ(ierr);
  *ce = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode ConvEstSetFromOptions(ConvEst ce)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(ce->comm, "", "Convergence Estimator Options", "ConvEst");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-num_refine", "The number of refinements for the convergence check", "ConvEst", ce->Nr, &ce->Nr, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode ConvEstSetSolver(ConvEst ce, SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ce->snes = snes;
  ierr = SNESGetDM(ce->snes, &ce->idm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ConvEstSetup(ConvEst ce)
{
  PetscDS        prob;
  PetscInt       f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDS(ce->idm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &ce->Nf);CHKERRQ(ierr);
  ierr = PetscMalloc1((ce->Nr+1)*ce->Nf, &ce->errors);CHKERRQ(ierr);
  ierr = PetscMalloc2(ce->Nf, &ce->initGuess, ce->Nf, &ce->exactSol);CHKERRQ(ierr);
  for (f = 0; f < ce->Nf; ++f) ce->initGuess[f] = zero;
  for (f = 0; f < ce->Nf; ++f) {
    ierr = PetscDSGetExactSolution(prob, f, &ce->exactSol[f]);CHKERRQ(ierr);
    if (!ce->exactSol[f]) SETERRQ1(ce->comm, PETSC_ERR_ARG_WRONG, "DS must contain exact solution functions in order to estimate convergence, missing for field %D", f);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ConvEstLinearRegression_Private(ConvEst ce, PetscInt n, const PetscReal x[], const PetscReal y[], PetscReal *slope, PetscReal *intercept)
{
  PetscReal *X, *Y, H[4], beta[2];
  PetscInt   i, j, k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc2(n*2, &X, n*2, &Y);CHKERRQ(ierr);
  for (k = 0; k < n; ++k) {
    /* X[n,2] = [1, x] */
    X[k*2+0] = 1.0;
    X[k*2+1] = x[k];
  }
  /* H = X^T X */
  for (i = 0; i < 2; ++i) {
    for (j = 0; j < 2; ++j) {
      H[i*2+j] = 0.0;
      for (k = 0; k < n; ++k) {
        H[i*2+j] += X[k*2+i] * X[k*2+j];
      }
    }
  }
  /* H = (X^T X)^{-1} */
  {
    PetscBLASInt two = 2, ipiv[2], info;
    PetscReal    work[2];

    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
    PetscStackCallBLAS("LAPACKgetrf", LAPACKgetrf_(&two, &two, H, &two, ipiv, &info));
    PetscStackCallBLAS("LAPACKgetri", LAPACKgetri_(&two, H, &two, ipiv, work, &two, &info));
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
  }
    /* Y = H X^T */
  for (i = 0; i < 2; ++i) {
    for (k = 0; k < n; ++k) {
      Y[i*n+k] = 0.0;
      for (j = 0; j < 2; ++j) {
        Y[i*n+k] += H[i*2+j] * X[k*2+j];
      }
    }
  }
  /* beta = Y error = [y-intercept, slope] */
  for (i = 0; i < 2; ++i) {
    beta[i] = 0.0;
    for (k = 0; k < n; ++k) {
      beta[i] += Y[i*n+k] * y[k];
    }
  }
  ierr = PetscFree2(X, Y);CHKERRQ(ierr);
  *intercept = beta[0];
  *slope     = beta[1];
  PetscFunctionReturn(0);
}

static PetscErrorCode ConvEstGetConvRate(ConvEst ce, PetscReal *alpha)
{
  DM            *dm;
  PetscDS        prob;
  PetscObject    disc;
  const char    *uname, *dmname;
  void          *ctx;
  Vec            u;
  PetscReal      t = 0.0, *x, *y, slope, intercept;
  PetscInt      *dof, dim, Nr = ce->Nr, r;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(ce->idm, &dim);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(ce->idm, &ctx);CHKERRQ(ierr);
  ierr = DMGetDS(ce->idm, &prob);CHKERRQ(ierr);
  ierr = DMPlexSetRefinementUniform(ce->idm, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscDSGetDiscretization(prob, 0, &disc);CHKERRQ(ierr);
  ierr = PetscObjectGetName(disc, &uname);CHKERRQ(ierr);
  ierr = PetscMalloc2((Nr+1), &dm, (Nr+1), &dof);CHKERRQ(ierr);
  dm[0]  = ce->idm;
  *alpha = 0.0;
  /* Loop over meshes */
  for (r = 0; r <= Nr; ++r) {
    if (r > 0) {
      ierr = DMRefine(dm[r-1], MPI_COMM_NULL, &dm[r]);CHKERRQ(ierr);
      ierr = DMSetCoarseDM(dm[r], dm[r-1]);CHKERRQ(ierr);
      ierr = DMSetDS(dm[r], prob);CHKERRQ(ierr);
      ierr = PetscObjectGetName((PetscObject) dm[r-1], &dmname);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) dm[r], dmname);CHKERRQ(ierr);
    }
    ierr = DMViewFromOptions(dm[r], NULL, "-conv_dm_view");CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm[r], 0, NULL, &dof[r]);CHKERRQ(ierr);
    /* Create solution */
    ierr = DMCreateGlobalVector(dm[r], &u);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) u, uname);CHKERRQ(ierr);
    /* Setup solver */
    ierr = SNESReset(ce->snes);CHKERRQ(ierr);
    ierr = SNESSetDM(ce->snes, dm[r]);CHKERRQ(ierr);
    ierr = DMPlexSetSNESLocalFEM(dm[r], ctx, ctx, ctx);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(ce->snes);CHKERRQ(ierr);
    /* Create initial guess */
    ierr = DMProjectFunction(dm[r], t, ce->initGuess, NULL, INSERT_VALUES, u);CHKERRQ(ierr);
    ierr = SNESSolve(ce->snes, NULL, u);CHKERRQ(ierr);
    ierr = DMComputeL2FieldDiff(dm[r], t, ce->exactSol, NULL, u, &ce->errors[r*ce->Nf]);CHKERRQ(ierr);
    /* Monitor */
    if (ce->monitor) {
      PetscReal *errors = &ce->errors[r*ce->Nf];
      PetscInt f;

      ierr = PetscPrintf(ce->comm, "L_2 Error: [");CHKERRQ(ierr);
      for (f = 0; f < ce->Nf; ++f) {
        if (f > 0) {ierr = PetscPrintf(ce->comm, ", ");CHKERRQ(ierr);}
        if (errors[f] < 1.0e-11) {ierr = PetscPrintf(ce->comm, "< 1e-11");CHKERRQ(ierr);}
        else                     {ierr = PetscPrintf(ce->comm, "%g", errors[f]);CHKERRQ(ierr);}
      }
      ierr = PetscPrintf(ce->comm, "]\n");CHKERRQ(ierr);
    }
    /* Cleanup */
    ierr = VecDestroy(&u);CHKERRQ(ierr);
  }
  for (r = 1; r <= Nr; ++r) {
    ierr = DMDestroy(&dm[r]);CHKERRQ(ierr);
  }
  /* Fit convergence rate */
  ierr = PetscMalloc2(Nr+1, &x, Nr+1, &y);CHKERRQ(ierr);
  for (r = 0; r <= Nr; ++r) {
    x[r] = PetscLog10Real(dof[r]);
    y[r] = PetscLog10Real(ce->errors[r*ce->Nf+0]);
  }
  ierr = ConvEstLinearRegression_Private(ce, Nr+1, x, y, &slope, &intercept);CHKERRQ(ierr);
  ierr = PetscFree2(x, y);CHKERRQ(ierr);
  /* Since h^{-dim} = N, lg err = s lg N + b = -s dim lg h + b */
  *alpha = -slope * dim;
  ierr = PetscFree2(dm, dof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;    /* Problem specification */
  ConvEst        conv;  /* Convergence estimator object */
  SNES           snes;  /* Nonlinear solver */
  AppCtx         user;  /* User-defined work context */
  PetscReal      alpha; /* Convergence rate of the solution error in the L_2 norm */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
  ierr = SetupDiscretization(dm, &user);CHKERRQ(ierr);

  ierr = ConvEstCreate(PETSC_COMM_WORLD, &conv);CHKERRQ(ierr);
  ierr = ConvEstSetSolver(conv, snes);CHKERRQ(ierr);
  ierr = ConvEstSetFromOptions(conv);CHKERRQ(ierr);
  ierr = ConvEstSetup(conv);CHKERRQ(ierr);
  ierr = ConvEstGetConvRate(conv, &alpha);CHKERRQ(ierr);
  ierr = ConvEstDestroy(&conv);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Convergence rate: %0.2g\n", alpha);CHKERRQ(ierr);

  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 2d_p1_0
    requires: triangle
    args: -petscspace_order 1 -dm_refine 2 -num_refine 4
  test:
    suffix: 2d_p2_0
    requires: triangle
    args: -petscspace_order 2 -dm_refine 2 -num_refine 4
  test:
    suffix: 2d_p3_0
    requires: triangle
    args: -petscspace_order 3 -dm_refine 2 -num_refine 4
  test:
    suffix: 2d_q1_0
    requires: triangle
    args: -simplex 0 -petscspace_poly_tensor -petscspace_order 1 -dm_refine 2 -num_refine 4
  test:
    suffix: 2d_q2_0
    requires: triangle
    args: -simplex 0 -petscspace_poly_tensor -petscspace_order 2 -dm_refine 2 -num_refine 4
  test:
    suffix: 2d_q3_0
    requires: triangle
    args: -simplex 0 -petscspace_poly_tensor -petscspace_order 3 -dm_refine 2 -num_refine 4

TEST*/
