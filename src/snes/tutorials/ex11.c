static char help[] = "Poisson problem with finite elements.\n\
We solve the Poisson problem using a parallel unstructured mesh to discretize it.\n\
This example is a simplified version of ex12.c that only solves the linear problem.\n\
It uses discretized auxiliary fields for coefficient and right-hand side, \n\
supports multilevel solvers and non-conforming mesh refinement.\n\n\n";

/*
Here we describe the PETSc approach to solve nonlinear problems arising from Finite Element (FE) discretizations.

The general model requires to solve the residual equations

    residual(u) := \int_\Omega \phi \cdot f_0(u, \nabla u, a, \nabla a) + \nabla \phi : f_1(u, \nabla u, a, \nabla a) = 0

where \phi is a test function, u is the sought FE solution, and a generically describes auxiliary data (for example material properties).

The functions f_0 (scalar) and f_1 (vector) describe the problem, while : is the usual contraction operator for tensors, i.e. A : B = \sum_{ij} A_{ij} B_{ij}.

The discrete residual is (with abuse of notation)

    F(u) := \sum_e E_e^T [ B^T_e W_{0,e} f_0(u_q, \nabla u_q, a_q, \nabla a_q) + D_e W_{1,e} f_1(u_q, \nabla u_q, a_q, \nabla a_q) ]

where E are element restriction matrices (can support non-conforming meshes), W are quadrature weights, B (resp. D) are basis function (resp. derivative of basis function) matrices, and u_q,a_q are vectors sampled at quadrature points.

Having the residual in the above form, it is straightforward to derive its Jacobian (again with abuse of notation)

    J(u) := \sum_e E_e^T [B^T_e D^T_e] W  J_e [ B_e^T, D_e^T ]^T E_e,

where J_e is the 2x2 matrix

   | \partial_u f_0, \partial_{\grad u} f_0 |
   | \partial_u f_1, \partial_{\grad u} f_1 |

Here we use a single-field approach, but the extension to the multi-field case is straightforward.

To keep the presentation simple, here we are interested in solving the Poisson problem in divergence form

   - \nabla \cdot (K * \nabla u) = g

with either u = 0 or K * \partial_n u = 0 on \partial \Omega.
The above problem possesses the weak form

   \int_\Omega \nabla \phi K \nabla u + \phi g = 0,

thus we have:

   f_0 = g, f_1 = K * \nabla u, and the only non-zero term of the Jacobian is J_{11} = K

See https://petsc.org/release/manual/fe the and the paper "Achieving High Performance with Unified Residual Evaluation" (available at https://arxiv.org/abs/1309.1204) for additional information.
*/

/* Include the necessary definitions */
#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>

/* The f_0 function: we read the right-hand side from the first field of the auxiliary data */
static void f_0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscScalar g = a[0];

  f0[0] = g;
}

/* The f_1 function: we read the conductivity tensor from the second field of the auxiliary data */
static void f_1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscScalar *K = &a[1];

  for (PetscInt d1 = 0; d1 < dim; ++d1) {
    PetscScalar v = 0;
    for (PetscInt d2 = 0; d2 < dim; ++d2) v += K[d1 * dim + d2] * u_x[d2];
    f1[d1] = v;
  }
}

/* The only non-zero term for the Jacobian is J_11 */
static void J_11(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J11[])
{
  const PetscScalar *K = &a[1];

  for (PetscInt d1 = 0; d1 < dim; ++d1) {
    for (PetscInt d2 = 0; d2 < dim; ++d2) J11[d1 * dim + d2] = K[d1 * dim + d2];
  }
}

/* The boundary conditions: Dirichlet (essential) or Neumann (natural) */
typedef enum {
  BC_DIRICHLET,
  BC_NEUMANN,
} bcType;

static const char *const bcTypes[] = {"DIRICHLET", "NEUMANN", "bcType", "BC_", 0};

/* The forcing term: constant or analytical */
typedef enum {
  RHS_CONSTANT,
  RHS_ANALYTICAL,
} rhsType;

static const char *const rhsTypes[] = {"CONSTANT", "ANALYTICAL", "rhsType", "RHS_", 0};

/* the constant case */
static PetscErrorCode rhs_constant(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *g, void *ctx)
{
  *g = 1.0;
  return PETSC_SUCCESS;
}

/* the analytical case */
static PetscErrorCode rhs_analytical(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *g, void *ctx)
{
  PetscScalar v = 1.0;
  for (PetscInt d = 0; d < dim; d++) v *= PetscSinReal(2.0 * PETSC_PI * x[d]);
  *g = v;
  return PETSC_SUCCESS;
}

/* For the Neumann BC case we need a functional to be integrated: average -> \int_\Omega u dx */
static void average(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar obj[])
{
  obj[0] = u[0];
}

/* The conductivity coefficient term: constant, checkerboard or analytical */
typedef enum {
  COEFF_CONSTANT,
  COEFF_CHECKERBOARD,
  COEFF_ANALYTICAL,
} coeffType;

static const char *const coeffTypes[] = {"CONSTANT", "CHECKERBOARD", "ANALYTICAL", "coeffType", "COEFF_", 0};

/* the constant coefficient case */
static PetscErrorCode coefficient_constant(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *K, void *ctx)
{
  for (PetscInt d = 0; d < dim; d++) K[d * dim + d] = 1.0;
  return PETSC_SUCCESS;
}

/* the checkerboard coefficient case: 10^2 in odd ranks, 10^-2 in even ranks */
static PetscErrorCode coefficient_checkerboard(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *K, void *ctx)
{
  PetscScalar exponent = PetscGlobalRank % 2 ? 2.0 : -2.0;
  for (PetscInt d = 0; d < dim; d++) K[d * dim + d] = PetscPowScalar(10.0, exponent);
  return PETSC_SUCCESS;
}

/* the analytical case (channels in diagonal with 4 order of magnitude in jumps) */
static PetscErrorCode coefficient_analytical(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *K, void *ctx)
{
  for (PetscInt d = 0; d < dim; d++) {
    const PetscReal v = PetscPowReal(10.0, 4.0 * PetscSinReal(4.0 * PETSC_PI * x[d]) * PetscCosReal(4.0 * PETSC_PI * x[d]));
    K[d * dim + d]    = v;
  }
  return PETSC_SUCCESS;
}

/* The application context that defines our problem */
typedef struct {
  bcType    bc;         /* type of boundary conditions */
  rhsType   rhs;        /* type of right-hand side */
  coeffType coeff;      /* type of conductivity coefficient */
  PetscInt  order;      /* the polynomial order for the solution */
  PetscInt  rhsOrder;   /* the polynomial order for the right-hand side */
  PetscInt  coeffOrder; /* the polynomial order for the coefficient */
  PetscBool p4est;      /* if we want to use non-conforming AMR */
} AppCtx;

/* Process command line options */
static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  options->bc         = BC_DIRICHLET;
  options->rhs        = RHS_CONSTANT;
  options->coeff      = COEFF_CONSTANT;
  options->order      = 1;
  options->rhsOrder   = 1;
  options->coeffOrder = 1;
  options->p4est      = PETSC_FALSE;

  PetscOptionsBegin(comm, "", "Poisson problem options", "DMPLEX");
  PetscCall(PetscOptionsEnum("-bc_type", "Type of boundary condition", __FILE__, bcTypes, (PetscEnum)options->bc, (PetscEnum *)&options->bc, NULL));
  PetscCall(PetscOptionsEnum("-rhs_type", "Type of forcing term", __FILE__, rhsTypes, (PetscEnum)options->rhs, (PetscEnum *)&options->rhs, NULL));
  PetscCall(PetscOptionsEnum("-coefficient_type", "Type of conductivity coefficient", __FILE__, coeffTypes, (PetscEnum)options->coeff, (PetscEnum *)&options->coeff, NULL));
  PetscCall(PetscOptionsInt("-order", "The polynomial order for the approximation of the solution", __FILE__, options->order, &options->order, NULL));
  PetscCall(PetscOptionsInt("-rhs_order", "The polynomial order for the approximation of the right-hand side", __FILE__, options->rhsOrder, &options->rhsOrder, NULL));
  PetscCall(PetscOptionsInt("-coefficient_order", "The polynomial order for the approximation of the coefficient", __FILE__, options->coeffOrder, &options->coeffOrder, NULL));
  PetscCall(PetscOptionsBool("-p4est", "Use p4est to represent the mesh", __FILE__, options->p4est, &options->p4est, NULL));
  PetscOptionsEnd();

  /* View processed options */
  PetscCall(PetscPrintf(comm, "Simulation parameters\n"));
  PetscCall(PetscPrintf(comm, "  polynomial order: %" PetscInt_FMT "\n", options->order));
  PetscCall(PetscPrintf(comm, "  boundary conditions: %s\n", bcTypes[options->bc]));
  PetscCall(PetscPrintf(comm, "  right-hand side: %s, order %" PetscInt_FMT "\n", rhsTypes[options->rhs], options->rhsOrder));
  PetscCall(PetscPrintf(comm, "  coefficient: %s, order %" PetscInt_FMT "\n", coeffTypes[options->coeff], options->coeffOrder));
  PetscCall(PetscPrintf(comm, "  non-conforming AMR: %d\n", options->p4est));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Create mesh from command line options */
static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  /* Create various types of meshes only with command line options */
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetOptionsPrefix(*dm, "initial_"));
  PetscCall(DMPlexDistributeSetDefault(*dm, PETSC_FALSE));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMLocalizeCoordinates(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscCall(DMPlexDistributeSetDefault(*dm, PETSC_TRUE));
  PetscCall(DMSetOptionsPrefix(*dm, "mesh_"));
  PetscCall(DMSetFromOptions(*dm));

  /* If requested convert to a format suitable for non-conforming adaptive mesh refinement */
  if (user->p4est) {
    PetscInt dim;
    DM       dmConv;

    PetscCall(DMGetDimension(*dm, &dim));
    PetscCheck(dim == 2 || dim == 3, comm, PETSC_ERR_SUP, "p4est support not for dimension %" PetscInt_FMT, dim);
    PetscCall(DMConvert(*dm, dim == 3 ? DMP8EST : DMP4EST, &dmConv));
    if (dmConv) {
      PetscCall(DMDestroy(dm));
      PetscCall(DMSetOptionsPrefix(dmConv, "mesh_"));
      PetscCall(DMSetFromOptions(dmConv));
      *dm = dmConv;
    }
  }
  PetscCall(DMSetUp(*dm));

  /* View the mesh.
     With a single call we can dump ASCII information, VTK data for visualization, store the mesh in HDF5 format, etc. */
  PetscCall(PetscObjectSetName((PetscObject)*dm, "Mesh"));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Setup the discrete problem */
static PetscErrorCode SetupProblem(DM dm, DM fdm, AppCtx *user)
{
  DM             plex, dmAux, cdm = NULL, coordDM;
  Vec            auxData, auxDataGlobal;
  PetscDS        ds;
  DMPolytopeType ct;
  PetscInt       dim, cdim, cStart;
  PetscFE        fe, fe_rhs, fe_K;
  PetscErrorCode (*auxFuncs[])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx) = {NULL, NULL};
  void *auxCtxs[]                                                                                                          = {NULL, NULL};

  PetscFunctionBeginUser;
  /* Create the Finite Element for the solution and pass it to the problem DM */
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMConvert(dm, DMPLEX, &plex));
  PetscCall(DMPlexGetHeightStratum(plex, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(plex, cStart, &ct));
  PetscCall(DMDestroy(&plex));
  PetscCall(PetscFECreateLagrangeByCell(PETSC_COMM_SELF, dim, 1, ct, user->order, PETSC_DETERMINE, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "potential"));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(dm));

  /* Set residual and Jacobian callbacks */
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetResidual(ds, 0, f_0, f_1));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, J_11));
  /* Tell DMPLEX we are going to use FEM callbacks */
  PetscCall(DMPlexSetSNESLocalFEM(dm, PETSC_FALSE, &user));

  /* Create the Finite Element for the auxiliary data */
  PetscCall(PetscFECreateLagrangeByCell(PETSC_COMM_SELF, dim, 1, ct, user->rhsOrder, PETSC_DETERMINE, &fe_rhs));
  PetscCall(PetscObjectSetName((PetscObject)fe_rhs, "g"));
  PetscCall(PetscFECopyQuadrature(fe, fe_rhs));
  PetscCall(DMGetCoordinateDM(dm, &coordDM));
  PetscCall(DMGetDimension(coordDM, &cdim));
  PetscCall(PetscFECreateLagrangeByCell(PETSC_COMM_SELF, dim, cdim * cdim, ct, user->coeffOrder, PETSC_DETERMINE, &fe_K));
  PetscCall(PetscObjectSetName((PetscObject)fe_K, "K"));
  PetscCall(PetscFECopyQuadrature(fe, fe_K));
  PetscCall(DMClone(dm, &dmAux));
  PetscCall(DMSetField(dmAux, 0, NULL, (PetscObject)fe_rhs));
  PetscCall(DMSetField(dmAux, 1, NULL, (PetscObject)fe_K));
  PetscCall(DMCreateDS(dmAux));

  /* Project the requested rhs and K to the auxiliary DM and pass it to the problem DM */
  PetscCall(DMCreateLocalVector(dmAux, &auxData));
  PetscCall(DMCreateGlobalVector(dmAux, &auxDataGlobal));
  PetscCall(PetscObjectSetName((PetscObject)auxData, ""));
  if (!fdm) {
    switch (user->rhs) {
    case RHS_CONSTANT:
      auxFuncs[0] = rhs_constant;
      auxCtxs[0]  = NULL;
      break;
    case RHS_ANALYTICAL:
      auxFuncs[0] = rhs_analytical;
      auxCtxs[0]  = NULL;
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported rhs type %s", rhsTypes[user->rhs]);
    }
    switch (user->coeff) {
    case COEFF_CONSTANT:
      auxFuncs[1] = coefficient_constant;
      auxCtxs[1]  = NULL;
      break;
    case COEFF_CHECKERBOARD:
      auxFuncs[1] = coefficient_checkerboard;
      auxCtxs[1]  = NULL;
      break;
    case COEFF_ANALYTICAL:
      auxFuncs[1] = coefficient_analytical;
      auxCtxs[1]  = NULL;
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported coefficient type %s", coeffTypes[user->coeff]);
    }
    PetscCall(DMGetDS(dmAux, &ds));
    PetscCall(DMProjectFunction(dmAux, 0.0, auxFuncs, auxCtxs, INSERT_ALL_VALUES, auxDataGlobal));
    if (user->bc == BC_NEUMANN) {
      PetscScalar vals[2];
      PetscInt    rhs_id = 0;
      IS          is;

      PetscCall(PetscDSSetObjective(ds, rhs_id, average));
      PetscCall(DMPlexComputeIntegralFEM(dmAux, auxDataGlobal, vals, NULL));
      PetscCall(DMCreateSubDM(dmAux, 1, &rhs_id, &is, NULL));
      PetscCall(VecISShift(auxDataGlobal, is, -vals[rhs_id]));
      PetscCall(DMPlexComputeIntegralFEM(dmAux, auxDataGlobal, vals, NULL));
      PetscCall(ISDestroy(&is));
    }
  } else {
    Mat J;
    Vec auxDataGlobalf, auxDataf, Jscale;
    DM  dmAuxf;

    PetscCall(DMGetAuxiliaryVec(fdm, NULL, 0, 0, &auxDataf));
    PetscCall(VecGetDM(auxDataf, &dmAuxf));
    PetscCall(DMSetCoarseDM(dmAuxf, dmAux));
    PetscCall(DMCreateGlobalVector(dmAuxf, &auxDataGlobalf));
    PetscCall(DMLocalToGlobal(dmAuxf, auxDataf, INSERT_VALUES, auxDataGlobalf));
    PetscCall(DMCreateInterpolation(dmAux, dmAuxf, &J, &Jscale));
    PetscCall(MatInterpolate(J, auxDataGlobalf, auxDataGlobal));
    PetscCall(VecPointwiseMult(auxDataGlobal, auxDataGlobal, Jscale));
    PetscCall(VecDestroy(&Jscale));
    PetscCall(VecDestroy(&auxDataGlobalf));
    PetscCall(MatDestroy(&J));
    PetscCall(DMSetCoarseDM(dmAuxf, NULL));
  }
  /* auxiliary data must be a local vector */
  PetscCall(DMGlobalToLocal(dmAux, auxDataGlobal, INSERT_VALUES, auxData));
  { /* view auxiliary data */
    PetscInt level;
    char     optionName[PETSC_MAX_PATH_LEN];

    PetscCall(DMGetRefineLevel(dm, &level));
    PetscCall(PetscSNPrintf(optionName, sizeof(optionName), "-aux_%" PetscInt_FMT "_vec_view", level));
    PetscCall(VecViewFromOptions(auxData, NULL, optionName));
  }
  PetscCall(DMSetAuxiliaryVec(dm, NULL, 0, 0, auxData));
  PetscCall(VecDestroy(&auxData));
  PetscCall(VecDestroy(&auxDataGlobal));
  PetscCall(DMDestroy(&dmAux));

  /* Setup boundary conditions
     Since we use homogeneous natural boundary conditions for the Neumann problem we
     only handle the Dirichlet case here */
  if (user->bc == BC_DIRICHLET) {
    DMLabel  label;
    PetscInt id = 1;

    /* Label faces on the mesh boundary */
    PetscCall(DMCreateLabel(dm, "boundary"));
    PetscCall(DMGetLabel(dm, "boundary", &label));
    PetscCall(DMPlexMarkBoundaryFaces(dm, 1, label));
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "dirichlet", label, 1, &id, 0, 0, NULL, NULL, NULL, NULL, NULL));
  }

  /* Iterate on coarser mesh if present */
  PetscCall(DMGetCoarseDM(dm, &cdm));
  if (cdm) PetscCall(SetupProblem(cdm, dm, user));

  PetscCall(PetscFEDestroy(&fe));
  PetscCall(PetscFEDestroy(&fe_rhs));
  PetscCall(PetscFEDestroy(&fe_K));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* We are now ready to run the simulation */
int main(int argc, char **argv)
{
  DM     dm;   /* problem specification */
  SNES   snes; /* nonlinear solver */
  KSP    ksp;  /* linear solver */
  Vec    u;    /* solution vector */
  AppCtx user; /* user-defined work context */

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(SetupProblem(dm, NULL, &user));

  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(SNESSetType(snes, SNESKSPONLY));
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPSetType(ksp, KSPCG));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_NATURAL));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(PetscObjectSetName((PetscObject)u, ""));
  PetscCall(VecSetRandom(u, NULL));
  PetscCall(SNESSolve(snes, NULL, u));
  PetscCall(VecDestroy(&u));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    requires: triangle !single
    args: -bc_type {{dirichlet neumann}separate output} -rhs_type {{constant analytical}separate output} -coefficient_type {{constant checkerboard analytical}separate output} -initial_dm_plex_simplex 1 -pc_type svd -snes_type newtonls -snes_error_if_not_converged

  test:
    requires: !single
    suffix: 0_quad
    args: -bc_type {{dirichlet neumann}separate output} -rhs_type {{constant analytical}separate output} -coefficient_type {{constant checkerboard analytical}separate output} -initial_dm_plex_simplex 0 -pc_type svd -snes_type newtonls -snes_error_if_not_converged

  test:
    suffix: 0_p4est
    requires: p4est !single
    args: -bc_type {{dirichlet neumann}separate output} -rhs_type {{constant analytical}separate output} -coefficient_type {{constant checkerboard analytical}separate output} -initial_dm_plex_simplex 0 -pc_type svd -snes_type newtonls -snes_error_if_not_converged

  testset:
    nsize: 2
    requires: hpddm slepc !single defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
    args: -pc_type hpddm -pc_hpddm_coarse_correction balanced -pc_hpddm_coarse_mat_type aij -pc_hpddm_levels_eps_nev 1 -pc_hpddm_levels_sub_pc_type lu -ksp_monitor -initial_dm_plex_simplex 0 -petscpartitioner_type simple
    output_file: output/ex11_hpddm.out
    test:
      suffix: hpddm
      args:
    test:
      suffix: hpddm_harmonic_overlap
      args: -pc_hpddm_harmonic_overlap 1 -pc_hpddm_has_neumann false -pc_hpddm_levels_1_pc_asm_type basic
    test:
      suffix: hpddm_p4est
      requires: p4est
      args: -p4est
      filter: sed -e "s/non-conforming AMR: 1/non-conforming AMR: 0/g"

  test:
    nsize: 4
    suffix: gdsw_corner
    requires: triangle
    args: -pc_type mg -pc_mg_galerkin -pc_mg_adapt_interp_coarse_space gdsw -pc_mg_levels 2 -mg_levels_pc_type asm -initial_dm_plex_shape ball -initial_dm_refine 2 -mesh_dm_mat_type {{aij is}} -mg_levels_sub_pc_type lu -mg_levels_pc_asm_type basic -petscpartitioner_type simple

  test:
    suffix: asm_seq
    args: -pc_type asm -pc_asm_dm_subdomains -sub_pc_type cholesky -snes_type newtonls -snes_error_if_not_converged -initial_dm_plex_simplex 0

TEST*/
