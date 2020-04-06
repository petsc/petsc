static char help[] = "Poisson Problem in mixed form with 2d and 3d with finite elements.\n\
We solve the Poisson problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\
This example supports automatic convergence estimation\n\
and Hdiv elements.\n\n\n";

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscconvest.h>

typedef enum {SOL_LINEAR, SOL_QUADRATIC, SOL_QUARTIC, SOL_UNKNOWN, NUM_SOLTYPE} SolType;
const char *SolTypeNames[NUM_SOLTYPE+3] = {"linear", "quadratic", "quartic", "unknown", "SolType", "SOL_", NULL};

typedef struct {
  /* Domain and mesh definition */
  PetscInt  dim;     /* The topological mesh dimension */
  PetscBool simplex; /* Simplicial mesh */
  SolType   solType; /* The type of exact solution */
} AppCtx;

/* 2D Dirichlet potential example

  u = x
  q = <1, 0>
  f = 0

  We will need a boundary integral of u over \Gamma.
*/
static PetscErrorCode linear_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = x[0];
  return 0;
}
static PetscErrorCode linear_q(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt c;
  for (c = 0; c < Nc; ++c) u[c] = c ? 0.0 : 1.0;
  return 0;
}

/* 2D Dirichlet potential example

  u = x^2 + y^2
  q = <2x, 2y>
  f = 4

  We will need a boundary integral of u over \Gamma.
*/
static PetscErrorCode quadratic_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;

  u[0] = 0.0;
  for (d = 0; d < dim; ++d) u[0] += x[d]*x[d];
  return 0;
}
static PetscErrorCode quadratic_q(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt c;
  for (c = 0; c < Nc; ++c) u[c] = 2.0*x[c];
  return 0;
}

/* 2D Dirichlet potential example

  u = x (1-x) y (1-y)
  q = <(1-2x) y (1-y), x (1-x) (1-2y)>
  f = -y (1-y) - x (1-x)

  u|_\Gamma = 0 so that the boundary integral vanishes
*/
static PetscErrorCode quartic_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;

  u[0] = 1.0;
  for (d = 0; d < dim; ++d) u[0] *= x[d]*(1.0 - x[d]);
  return 0;
}
static PetscErrorCode quartic_q(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt c, d;

  for (c = 0; c < Nc; ++c) {
    u[c] = 1.0;
    for (d = 0; d < dim; ++d) {
      if (c == d) u[c] *= 1 - 2.0*x[d];
      else        u[c] *= x[d]*(1.0 - x[d]);
    }
  }
  return 0;
}

/* <v, -\nabla\cdot q> + <v, f> */
static void f0_linear_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                        PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = 0.0;
}
static void f0_bd_linear_q(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                           PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscScalar potential;
  PetscInt    d;

  linear_u(dim, t, x, dim, &potential, NULL);
  for (d = 0; d < dim; ++d) f0[d] = -potential*n[d];
}
static void f0_quadratic_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                           PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  f0[0] = 0.0;
  for (d = 0; d < dim; ++d) {
    f0[0] -= u_x[uOff_x[0]+d*dim+d];
  }
  f0[0] += 4.0;
}
static void f0_bd_quadratic_q(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                              PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscScalar potential;
  PetscInt    d;

  quadratic_u(dim, t, x, dim, &potential, NULL);
  for (d = 0; d < dim; ++d) f0[d] = -potential*n[d];
}
static void f0_quartic_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                        PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  f0[0] = 0.0;
  for (d = 0; d < dim; ++d) {
    f0[0] -= u_x[uOff_x[0]+d*dim+d];
    f0[0] += -2.0*x[d]*(1.0 - x[d]);
  }
}

/* <w, q> */
static void f0_q(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt c;

  for (c = 0; c < dim; ++c) {
    f0[c] = u[uOff[0]+c];
  }
}

/* <\nabla\cdot w, u> = <\nabla w, Iu> */
static void f1_q(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt c, d;
  for (c = 0; c < dim; ++c) {
    for (d = 0; d < dim; ++d) {
      if (c == d) f1[c*dim+d] = u[uOff[1]];
    }
  }
}

/* <w, q> */
static void g0_qq(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  PetscInt c;
  for (c = 0; c < dim; ++c) g0[c*dim+c] = 1.0;
}

/* <\nabla\cdot w, u> = <\nabla w, Iu> */
static void g2_qu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g2[d*dim+d] = 1.0;
}

/* <v, -\nabla\cdot q> */
static void g1_uq(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g1[d*dim+d] = -1.0;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim     = 2;
  options->simplex = PETSC_TRUE;
  options->solType = SOL_LINEAR;

  ierr = PetscOptionsBegin(comm, "", "Poisson Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex24.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Simplicial (true) or tensor (false) mesh", "ex24.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-sol_type", "Type of exact solution", "ex24.c", SolTypeNames, (PetscEnum) options->solType, (PetscEnum *) &options->solType, NULL);CHKERRQ(ierr);

  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (0) {
    DMLabel     label;
    const char *name = "marker";

    ierr = DMPlexCreateReferenceCell(comm, user->dim, user->simplex, dm);CHKERRQ(ierr);
    ierr = DMCreateLabel(*dm, name);CHKERRQ(ierr);
    ierr = DMGetLabel(*dm, name, &label);CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(*dm, 1, label);CHKERRQ(ierr);
    ierr = DMPlexLabelComplete(*dm, label);CHKERRQ(ierr);
  } else {
    /* Create box mesh */
    ierr = DMPlexCreateBoxMesh(comm, user->dim, user->simplex, NULL, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  }
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
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupPrimalProblem(DM dm, AppCtx *user)
{
  PetscDS        prob;
  const PetscInt id = 1;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 0, f0_q, f1_q);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, g0_qq, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 1, NULL, NULL, g2_qu, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 1, 0, NULL, g1_uq, NULL, NULL);CHKERRQ(ierr);
  switch (user->solType)
  {
    case SOL_LINEAR:
      ierr = PetscDSSetResidual(prob, 1, f0_linear_u, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetBdResidual(prob, 0, f0_bd_linear_q, NULL);CHKERRQ(ierr);
      ierr = DMAddBoundary(dm, DM_BC_NATURAL, "Dirichlet Bd Integral", "marker", 0, 0, NULL, (void (*)(void)) NULL, 1, &id, user);CHKERRQ(ierr);
      ierr = PetscDSSetExactSolution(prob, 0, linear_q, user);CHKERRQ(ierr);
      ierr = PetscDSSetExactSolution(prob, 1, linear_u, user);CHKERRQ(ierr);
      break;
    case SOL_QUADRATIC:
      ierr = PetscDSSetResidual(prob, 1, f0_quadratic_u, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetBdResidual(prob, 0, f0_bd_quadratic_q, NULL);CHKERRQ(ierr);
      ierr = DMAddBoundary(dm, DM_BC_NATURAL, "Dirichlet Bd Integral", "marker", 0, 0, NULL, (void (*)(void)) NULL, 1, &id, user);CHKERRQ(ierr);
      ierr = PetscDSSetExactSolution(prob, 0, quadratic_q, user);CHKERRQ(ierr);
      ierr = PetscDSSetExactSolution(prob, 1, quadratic_u, user);CHKERRQ(ierr);
      break;
    case SOL_QUARTIC:
      ierr = PetscDSSetResidual(prob, 1, f0_quartic_u, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetExactSolution(prob, 0, quartic_q, user);CHKERRQ(ierr);
      ierr = PetscDSSetExactSolution(prob, 1, quartic_u, user);CHKERRQ(ierr);
      break;
    default: SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Invalid exact solution type %s", SolTypeNames[PetscMin(user->solType, SOL_UNKNOWN)]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, PetscErrorCode (*setup)(DM, AppCtx *), AppCtx *user)
{
  DM              cdm = dm;
  PetscFE         feq, feu;
  const PetscInt  dim = user->dim;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  /* Create finite element */
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, dim, user->simplex, "field_",     -1, &feq);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) feq, "field");CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1,   user->simplex, "potential_", -1, &feu);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) feu, "potential");CHKERRQ(ierr);
  ierr = PetscFECopyQuadrature(feq, feu);CHKERRQ(ierr);
  /* Set discretization and boundary conditions for each mesh */
  ierr = DMSetField(dm, 0, NULL, (PetscObject) feq);CHKERRQ(ierr);
  ierr = DMSetField(dm, 1, NULL, (PetscObject) feu);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = (*setup)(dm, user);CHKERRQ(ierr);
  while (cdm) {
    ierr = DMCopyDisc(dm,cdm);CHKERRQ(ierr);
    /* TODO: Check whether the boundary of coarse meshes is marked */
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&feq);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&feu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;   /* Problem specification */
  SNES           snes; /* Nonlinear solver */
  Vec            u;    /* Solutions */
  AppCtx         user; /* User-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  /* Primal system */
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
  ierr = SetupDiscretization(dm, SetupPrimalProblem, &user);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = VecSet(u, 0.0);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "potential");CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(dm, &user, &user, &user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = DMSNESCheckFromOptions(snes, u, NULL, NULL);CHKERRQ(ierr);
  ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
  ierr = SNESGetSolution(snes, &u);CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-potential_view");CHKERRQ(ierr);
  /* Cleanup */
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 2d_bdm1_p0_0
    requires: triangle
    args: -sol_type linear \
          -field_petscspace_degree 1 -field_petscdualspace_type bdm -dm_refine 0 -convest_num_refine 1 -snes_convergence_estimate \
          -dmsnes_check .001 -snes_error_if_not_converged \
          -ksp_rtol 1e-10 -ksp_error_if_not_converged \
          -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition full \
            -fieldsplit_field_pc_type lu \
            -fieldsplit_potential_ksp_rtol 1e-10 -fieldsplit_potential_pc_type lu
  test:
    suffix: 2d_bdm1_p0_1
    requires: triangle
    args: -sol_type quadratic \
          -field_petscspace_degree 1 -field_petscdualspace_type bdm -dm_refine 0 -convest_num_refine 1 -snes_convergence_estimate \
          -dmsnes_check .001 -snes_error_if_not_converged \
          -ksp_rtol 1e-10 -ksp_error_if_not_converged \
          -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition full \
            -fieldsplit_field_pc_type lu \
            -fieldsplit_potential_ksp_rtol 1e-10 -fieldsplit_potential_pc_type lu
  test:
    suffix: 2d_bdm1_p0_2
    requires: triangle
    args: -sol_type quartic \
          -field_petscspace_degree 1 -field_petscdualspace_type bdm -dm_refine 0 -convest_num_refine 1 -snes_convergence_estimate \
          -snes_error_if_not_converged \
          -ksp_rtol 1e-10 -ksp_error_if_not_converged \
          -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition full \
            -fieldsplit_field_pc_type lu \
            -fieldsplit_potential_ksp_rtol 1e-10 -fieldsplit_potential_pc_type lu

  test:
    suffix: 2d_p2_p0_vtk
    requires: triangle
    args: -sol_type linear \
          -field_petscspace_degree 2 -dm_refine 0 -convest_num_refine 1 -snes_convergence_estimate \
          -dmsnes_check .001 -snes_error_if_not_converged \
          -ksp_rtol 1e-10 -ksp_error_if_not_converged \
          -potential_view vtk: -exact_vec_view vtk: \
          -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition full \
            -fieldsplit_field_pc_type lu \
            -fieldsplit_potential_ksp_rtol 1e-10 -fieldsplit_potential_pc_type lu

  test:
    suffix: 2d_p2_p0_vtu
    requires: triangle
    args: -sol_type linear \
          -field_petscspace_degree 2 -dm_refine 0 -convest_num_refine 1 -snes_convergence_estimate \
          -dmsnes_check .001 -snes_error_if_not_converged \
          -ksp_rtol 1e-10 -ksp_error_if_not_converged \
          -potential_view vtk:multifield.vtu -exact_vec_view vtk:exact.vtu \
          -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition full \
            -fieldsplit_field_pc_type lu \
            -fieldsplit_potential_ksp_rtol 1e-10 -fieldsplit_potential_pc_type lu
TEST*/

/* These tests will be active once tensor P^- is working

  test:
    suffix: 2d_bdmq1_p0_0
    requires: triangle
    args: -simplex 0 -sol_type linear \
          -field_petscspace_poly_type pminus_hdiv -field_petscspace_degree 1 -field_petscdualspace_type bdm -dm_refine 0 -convest_num_refine 3 -snes_convergence_estimate \
          -dmsnes_check .001 -snes_error_if_not_converged \
          -ksp_rtol 1e-10 -ksp_error_if_not_converged \
          -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition full \
            -fieldsplit_field_pc_type lu \
            -fieldsplit_potential_ksp_rtol 1e-10 -fieldsplit_potential_pc_type lu
  test:
    suffix: 2d_bdmq1_p0_2
    requires: triangle
    args: -simplex 0 -sol_type quartic \
          -field_petscspace_poly_type_no pminus_hdiv -field_petscspace_degree 1 -field_petscdualspace_type bdm -dm_refine 0 -convest_num_refine 3 -snes_convergence_estimate \
          -dmsnes_check .001 -snes_error_if_not_converged \
          -ksp_rtol 1e-10 -ksp_error_if_not_converged \
          -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -pc_fieldsplit_schur_precondition full \
            -fieldsplit_field_pc_type lu \
            -fieldsplit_potential_ksp_rtol 1e-10 -fieldsplit_potential_pc_type lu

*/
