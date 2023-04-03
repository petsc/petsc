static char help[] = "Mixed element discretization of the Poisson equation.\n\n\n";

#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscds.h>
#include <petscsnes.h>
#include <petscconvest.h>
#include <petscbag.h>

/*
The Poisson equation

  -\Delta\phi = f

can be rewritten in first order form

  q - \nabla\phi  &= 0
  -\nabla \cdot q &= f
*/

typedef enum {
  SIGMA,
  NUM_CONSTANTS
} ConstantType;
typedef struct {
  PetscReal sigma; /* Nondimensional charge per length in x */
} Parameter;

typedef enum {
  SOL_CONST,
  SOL_LINEAR,
  SOL_QUADRATIC,
  SOL_TRIG,
  SOL_TRIGX,
  SOL_PARTICLES,
  NUM_SOL_TYPES
} SolType;
static const char *solTypes[] = {"const", "linear", "quadratic", "trig", "trigx", "particles"};

typedef struct {
  SolType   solType; /* MMS solution type */
  PetscBag  bag;     /* Problem parameters */
  PetscBool particleRHS;
  PetscInt  Np;
} AppCtx;

/* SOLUTION CONST: \phi = 1, q = 0, f = 0 */
static PetscErrorCode const_phi(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  *u = 1.0;
  return PETSC_SUCCESS;
}

static PetscErrorCode const_q(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  for (PetscInt d = 0; d < dim; ++d) u[d] = 0.0;
  return PETSC_SUCCESS;
}

/* SOLUTION LINEAR: \phi = 2y, q = <0, 2>, f = 0 */
static PetscErrorCode linear_phi(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 2. * x[1];
  return PETSC_SUCCESS;
}

static PetscErrorCode linear_q(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 0.;
  u[1] = 2.;
  return PETSC_SUCCESS;
}

/* SOLUTION QUADRATIC: \phi = x (2\pi - x) + (1 + y) (1 - y), q = <2\pi - 2 x, - 2 y> = <2\pi, 0> - 2 x, f = -4 */
static PetscErrorCode quadratic_phi(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = x[0] * (6.283185307179586 - x[0]) + (1. + x[1]) * (1. - x[1]);
  return PETSC_SUCCESS;
}

static PetscErrorCode quadratic_q(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 6.283185307179586 - 2. * x[0];
  u[1] = -2. * x[1];
  return PETSC_SUCCESS;
}

static PetscErrorCode quadratic_q_bc(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = x[1] > 0. ? -2. * x[1] : 2. * x[1];
  return PETSC_SUCCESS;
}

static void f0_quadratic_phi(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  for (PetscInt d = 0; d < dim; ++d) f0[0] -= -2.0;
}

/* SOLUTION TRIG: \phi = sin(x) + (1/3 - y^2), q = <cos(x), -2 y>, f = sin(x) + 2 */
static PetscErrorCode trig_phi(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = PetscSinReal(x[0]) + (1. / 3. - x[1] * x[1]);
  return PETSC_SUCCESS;
}

static PetscErrorCode trig_q(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = PetscCosReal(x[0]);
  u[1] = -2. * x[1];
  return PETSC_SUCCESS;
}

static PetscErrorCode trig_q_bc(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = x[1] > 0. ? -2. * x[1] : 2. * x[1];
  return PETSC_SUCCESS;
}

static void f0_trig_phi(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] += PetscSinReal(x[0]) + 2.;
}

/* SOLUTION TRIGX: \phi = sin(x), q = <cos(x), 0>, f = sin(x) */
static PetscErrorCode trigx_phi(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = PetscSinReal(x[0]);
  return PETSC_SUCCESS;
}

static PetscErrorCode trigx_q(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = PetscCosReal(x[0]);
  u[1] = 0.;
  return PETSC_SUCCESS;
}

static PetscErrorCode trigx_q_bc(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = x[1] > 0. ? -2. * x[1] : 2. * x[1];
  return PETSC_SUCCESS;
}

static void f0_trigx_phi(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] += PetscSinReal(x[0]);
}

static void f0_q(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  for (PetscInt d = 0; d < dim; ++d) f0[d] += u[uOff[0] + d];
}

static void f1_q(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  for (PetscInt d = 0; d < dim; ++d) f1[d * dim + d] = u[uOff[1]];
}

static void f0_phi(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  for (PetscInt d = 0; d < dim; ++d) f0[0] += u_x[uOff_x[0] + d * dim + d];
}

static void f0_phi_backgroundCharge(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] += constants[SIGMA];
  for (PetscInt d = 0; d < dim; ++d) f0[0] += u_x[uOff_x[0] + d * dim + d];
}

static void g0_qq(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  for (PetscInt d = 0; d < dim; ++d) g0[d * dim + d] = 1.0;
}

static void g2_qphi(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  for (PetscInt d = 0; d < dim; ++d) g2[d * dim + d] = 1.0;
}

static void g1_phiq(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  for (PetscInt d = 0; d < dim; ++d) g1[d * dim + d] = 1.0;
}

/* SOLUTION PARTICLES: \phi = sigma, q = <cos(x), 0>, f = sin(x) */
static PetscErrorCode particles_phi(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 0.0795775;
  return PETSC_SUCCESS;
}

static PetscErrorCode particles_q(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 0.;
  u[1] = 0.;
  return PETSC_SUCCESS;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt sol;

  PetscFunctionBeginUser;
  options->solType     = SOL_CONST;
  options->particleRHS = PETSC_FALSE;
  options->Np          = 100;

  PetscOptionsBegin(comm, "", "Mixed Poisson Options", "DMPLEX");
  PetscCall(PetscOptionsBool("-particleRHS", "Flag to user particle RHS and background charge", "ex9.c", options->particleRHS, &options->particleRHS, NULL));
  sol = options->solType;
  PetscCall(PetscOptionsEList("-sol_type", "The MMS solution type", "ex12.c", solTypes, NUM_SOL_TYPES, solTypes[sol], &sol, NULL));
  options->solType = (SolType)sol;
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMSetApplicationContext(*dm, user));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupPrimalProblem(DM dm, AppCtx *user)
{
  PetscDS        ds;
  PetscWeakForm  wf;
  DMLabel        label;
  const PetscInt id = 1;

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSGetWeakForm(ds, &wf));
  PetscCall(DMGetLabel(dm, "marker", &label));
  PetscCall(PetscDSSetResidual(ds, 0, f0_q, f1_q));
  if (user->particleRHS) {
    PetscCall(PetscDSSetResidual(ds, 1, f0_phi_backgroundCharge, NULL));
  } else {
    PetscCall(PetscDSSetResidual(ds, 1, f0_phi, NULL));
  }
  PetscCall(PetscDSSetJacobian(ds, 0, 0, g0_qq, NULL, NULL, NULL));
  PetscCall(PetscDSSetJacobian(ds, 0, 1, NULL, NULL, g2_qphi, NULL));
  PetscCall(PetscDSSetJacobian(ds, 1, 0, NULL, g1_phiq, NULL, NULL));
  switch (user->solType) {
  case SOL_CONST:
    PetscCall(PetscDSSetExactSolution(ds, 0, const_q, user));
    PetscCall(PetscDSSetExactSolution(ds, 1, const_phi, user));
    break;
  case SOL_LINEAR:
    PetscCall(PetscDSSetExactSolution(ds, 0, linear_q, user));
    PetscCall(PetscDSSetExactSolution(ds, 1, linear_phi, user));
    break;
  case SOL_QUADRATIC:
    PetscCall(PetscWeakFormAddResidual(wf, NULL, 0, 1, 0, f0_quadratic_phi, NULL));
    PetscCall(PetscDSSetExactSolution(ds, 0, quadratic_q, user));
    PetscCall(PetscDSSetExactSolution(ds, 1, quadratic_phi, user));
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void))quadratic_q_bc, NULL, user, NULL));
    break;
  case SOL_TRIG:
    PetscCall(PetscWeakFormAddResidual(wf, NULL, 0, 1, 0, f0_trig_phi, NULL));
    PetscCall(PetscDSSetExactSolution(ds, 0, trig_q, user));
    PetscCall(PetscDSSetExactSolution(ds, 1, trig_phi, user));
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void))trig_q_bc, NULL, user, NULL));
    break;
  case SOL_TRIGX:
    PetscCall(PetscWeakFormAddResidual(wf, NULL, 0, 1, 0, f0_trigx_phi, NULL));
    PetscCall(PetscDSSetExactSolution(ds, 0, trigx_q, user));
    PetscCall(PetscDSSetExactSolution(ds, 1, trigx_phi, user));
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void))trigx_q_bc, NULL, user, NULL));
    break;
  case SOL_PARTICLES:
    PetscCall(PetscDSSetExactSolution(ds, 0, particles_q, user));
    PetscCall(PetscDSSetExactSolution(ds, 1, particles_phi, user));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Invalid solution type: %d", user->solType);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupDiscretization(DM dm, PetscInt Nf, const char *names[], PetscErrorCode (*setup)(DM, AppCtx *), AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe;
  DMPolytopeType ct;
  PetscInt       dim, cStart;
  char           prefix[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  for (PetscInt f = 0; f < Nf; ++f) {
    PetscCall(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", names[f]));
    PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, 1, ct, prefix, -1, &fe));
    PetscCall(PetscObjectSetName((PetscObject)fe, names[f]));
    if (f > 0) {
      PetscFE fe0;

      PetscCall(DMGetField(dm, 0, NULL, (PetscObject *)&fe0));
      PetscCall(PetscFECopyQuadrature(fe0, fe));
    }
    PetscCall(DMSetField(dm, f, NULL, (PetscObject)fe));
    PetscCall(PetscFEDestroy(&fe));
  }
  PetscCall(DMCreateDS(dm));
  PetscCall((*setup)(dm, user));
  while (cdm) {
    PetscCall(DMCopyDisc(dm, cdm));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitializeParticlesAndWeights(DM sw, AppCtx *user)
{
  DM           dm;
  PetscScalar *weight;
  PetscInt     Np, Npc, p, dim, c, cStart, cEnd, q, *cellid;
  PetscReal    weightsum = 0.0;
  PetscMPIInt  size, rank;
  Parameter   *param;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)sw), &size));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)sw), &rank));
  PetscOptionsBegin(PetscObjectComm((PetscObject)sw), "", "DMSwarm Options", "DMSWARM");
  PetscCall(PetscOptionsInt("-dm_swarm_num_particles", "The target number of particles", "", user->Np, &user->Np, NULL));
  PetscOptionsEnd();

  Np = user->Np;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Np = %" PetscInt_FMT "\n", Np));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetCellDM(sw, &dm));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));

  PetscCall(PetscBagGetData(user->bag, (void **)&param));
  PetscCall(DMSwarmSetLocalSizes(sw, Np, 0));

  Npc = Np / (cEnd - cStart);
  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_cellid, NULL, NULL, (void **)&cellid));
  for (c = 0, p = 0; c < cEnd - cStart; ++c) {
    for (q = 0; q < Npc; ++q, ++p) cellid[p] = c;
  }
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_cellid, NULL, NULL, (void **)&cellid));

  PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&weight));
  PetscCall(DMSwarmSortGetAccess(sw));
  for (p = 0; p < Np; ++p) {
    weight[p] = 1.0 / Np;
    weightsum += PetscRealPart(weight[p]);
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "weightsum = %1.10f\n", (double)weightsum));
  PetscCall(DMSwarmSortRestoreAccess(sw));
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&weight));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateSwarm(DM dm, AppCtx *user, DM *sw)
{
  PetscInt dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMCreate(PetscObjectComm((PetscObject)dm), sw));
  PetscCall(DMSetType(*sw, DMSWARM));
  PetscCall(DMSetDimension(*sw, dim));
  PetscCall(DMSwarmSetType(*sw, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*sw, dm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "w_q", 1, PETSC_SCALAR));

  PetscCall(DMSwarmFinalizeFieldRegister(*sw));

  PetscCall(InitializeParticlesAndWeights(*sw, user));

  PetscCall(DMSetFromOptions(*sw));
  PetscCall(DMSetApplicationContext(*sw, user));
  PetscCall(PetscObjectSetName((PetscObject)*sw, "Particles"));
  PetscCall(DMViewFromOptions(*sw, NULL, "-sw_view"));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupParameters(MPI_Comm comm, AppCtx *ctx)
{
  PetscBag   bag;
  Parameter *p;

  PetscFunctionBeginUser;
  /* setup PETSc parameter bag */
  PetscCall(PetscBagGetData(ctx->bag, (void **)&p));
  PetscCall(PetscBagSetName(ctx->bag, "par", "Parameters"));
  bag = ctx->bag;
  PetscCall(PetscBagRegisterScalar(bag, &p->sigma, 1.0, "sigma", "Charge per unit area, C/m^3"));
  PetscCall(PetscBagSetFromOptions(bag));
  {
    PetscViewer       viewer;
    PetscViewerFormat format;
    PetscBool         flg;

    PetscCall(PetscOptionsGetViewer(comm, NULL, NULL, "-param_view", &viewer, &format, &flg));
    if (flg) {
      PetscCall(PetscViewerPushFormat(viewer, format));
      PetscCall(PetscBagView(bag, viewer));
      PetscCall(PetscViewerFlush(viewer));
      PetscCall(PetscViewerPopFormat(viewer));
      PetscCall(PetscViewerDestroy(&viewer));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitializeConstants(DM sw, AppCtx *user)
{
  DM         dm;
  PetscReal *weight, totalCharge, totalWeight = 0., gmin[3], gmax[3];
  PetscInt   Np, p, dim;

  PetscFunctionBegin;
  PetscCall(DMSwarmGetCellDM(sw, &dm));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMGetBoundingBox(dm, gmin, gmax));
  PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&weight));
  for (p = 0; p < Np; ++p) totalWeight += weight[p];
  totalCharge = -1.0 * totalWeight;
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&weight));
  {
    Parameter *param;
    PetscReal  Area;

    PetscCall(PetscBagGetData(user->bag, (void **)&param));
    switch (dim) {
    case 1:
      Area = (gmax[0] - gmin[0]);
      break;
    case 2:
      Area = (gmax[0] - gmin[0]) * (gmax[1] - gmin[1]);
      break;
    case 3:
      Area = (gmax[0] - gmin[0]) * (gmax[1] - gmin[1]) * (gmax[2] - gmin[2]);
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Dimension %" PetscInt_FMT " not supported", dim);
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "dim = %" PetscInt_FMT "\ttotalWeight = %f\ttotalCharge = %f, Total Area = %f\n", dim, (double)totalWeight, (double)totalCharge, (double)Area));
    param->sigma = PetscAbsReal(totalCharge / (Area));

    PetscCall(PetscPrintf(PETSC_COMM_SELF, "sigma: %g\n", (double)param->sigma));
  }
  /* Setup Constants */
  {
    PetscDS    ds;
    Parameter *param;
    PetscCall(PetscBagGetData(user->bag, (void **)&param));
    PetscScalar constants[NUM_CONSTANTS];
    constants[SIGMA] = param->sigma;
    PetscCall(DMGetDS(dm, &ds));
    PetscCall(PetscDSSetConstants(ds, NUM_CONSTANTS, constants));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM          dm, sw;
  SNES        snes;
  Vec         u;
  AppCtx      user;
  const char *names[] = {"q", "phi"};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(SetupDiscretization(dm, 2, names, SetupPrimalProblem, &user));
  if (user.particleRHS) {
    PetscCall(PetscBagCreate(PETSC_COMM_SELF, sizeof(Parameter), &user.bag));
    PetscCall(CreateSwarm(dm, &user, &sw));
    PetscCall(SetupParameters(PETSC_COMM_WORLD, &user));
    PetscCall(InitializeConstants(sw, &user));
  }
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(PetscObjectSetName((PetscObject)u, "solution"));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(DMPlexSetSNESLocalFEM(dm, &user, &user, &user));
  PetscCall(DMSNESCheckFromOptions(snes, u));
  if (user.particleRHS) {
    DM       potential_dm;
    IS       potential_IS;
    Mat      M_p;
    Vec      rho, f, temp_rho;
    PetscInt fields = 1;

    PetscCall(DMGetGlobalVector(dm, &rho));
    PetscCall(PetscObjectSetName((PetscObject)rho, "rho"));
    PetscCall(DMCreateSubDM(dm, 1, &fields, &potential_IS, &potential_dm));
    PetscCall(DMCreateMassMatrix(sw, potential_dm, &M_p));
    PetscCall(MatViewFromOptions(M_p, NULL, "-mp_view"));
    PetscCall(DMGetGlobalVector(potential_dm, &temp_rho));
    PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &f));
    PetscCall(PetscObjectSetName((PetscObject)f, "particle weight"));
    PetscCall(VecViewFromOptions(f, NULL, "-weights_view"));
    PetscCall(MatMultTranspose(M_p, f, temp_rho));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &f));
    PetscCall(MatDestroy(&M_p));
    PetscCall(PetscObjectSetName((PetscObject)rho, "rho"));
    PetscCall(VecViewFromOptions(rho, NULL, "-poisson_rho_view"));
    PetscCall(VecISCopy(rho, potential_IS, SCATTER_FORWARD, temp_rho));
    PetscCall(VecViewFromOptions(temp_rho, NULL, "-rho_view"));
    PetscCall(DMRestoreGlobalVector(potential_dm, &temp_rho));
    PetscCall(DMDestroy(&potential_dm));
    PetscCall(ISDestroy(&potential_IS));

    PetscCall(SNESSolve(snes, rho, u));
    PetscCall(DMRestoreGlobalVector(dm, &rho));
  } else {
    PetscCall(SNESSolve(snes, NULL, u));
  }
  PetscCall(VecDestroy(&u));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&dm));
  if (user.particleRHS) {
    PetscCall(DMDestroy(&sw));
    PetscCall(PetscBagDestroy(&user.bag));
  }
  PetscCall(PetscFinalize());
  return PETSC_SUCCESS;
}

/*TEST

  # RT1-P0 on quads
  testset:
    args: -dm_plex_simplex 0 -dm_plex_box_bd periodic,none -dm_plex_box_faces 3,1 \
          -dm_plex_box_lower 0,-1 -dm_plex_box_upper 6.283185307179586,1\
          -phi_petscspace_degree 0 \
          -phi_petscdualspace_lagrange_use_moments \
          -phi_petscdualspace_lagrange_moment_order 2 \
          -q_petscfe_default_quadrature_order 1 \
          -q_petscspace_type sum \
          -q_petscspace_variables 2 \
          -q_petscspace_components 2 \
          -q_petscspace_sum_spaces 2 \
          -q_petscspace_sum_concatenate true \
          -q_sumcomp_0_petscspace_variables 2 \
          -q_sumcomp_0_petscspace_type tensor \
          -q_sumcomp_0_petscspace_tensor_spaces 2 \
          -q_sumcomp_0_petscspace_tensor_uniform false \
          -q_sumcomp_0_tensorcomp_0_petscspace_degree 1 \
          -q_sumcomp_0_tensorcomp_1_petscspace_degree 0 \
          -q_sumcomp_1_petscspace_variables 2 \
          -q_sumcomp_1_petscspace_type tensor \
          -q_sumcomp_1_petscspace_tensor_spaces 2 \
          -q_sumcomp_1_petscspace_tensor_uniform false \
          -q_sumcomp_1_tensorcomp_0_petscspace_degree 0 \
          -q_sumcomp_1_tensorcomp_1_petscspace_degree 1 \
          -q_petscdualspace_form_degree -1 \
          -q_petscdualspace_order 1 \
          -q_petscdualspace_lagrange_trimmed true \
          -ksp_error_if_not_converged \
          -pc_type fieldsplit -pc_fieldsplit_type schur \
          -pc_fieldsplit_schur_fact_type full -pc_fieldsplit_schur_precondition full

    # The Jacobian test is meaningless here
    test:
          suffix: quad_hdiv_0
          args: -dmsnes_check
          filter: sed -e "s/Taylor approximation converging at order.*''//"

    # The Jacobian test is meaningless here
    test:
          suffix: quad_hdiv_1
          args: -sol_type linear -dmsnes_check
          filter: sed -e "s/Taylor approximation converging at order.*''//"

    test:
          suffix: quad_hdiv_2
          args: -sol_type quadratic -dmsnes_check \
                -fieldsplit_q_pc_type lu -fieldsplit_phi_pc_type svd

    test:
          suffix: quad_hdiv_3
          args: -sol_type trig \
                -fieldsplit_q_pc_type lu -fieldsplit_phi_pc_type svd

    test:
          suffix: quad_hdiv_4
          requires: !single
          args: -sol_type trigx \
                -fieldsplit_q_pc_type lu -fieldsplit_phi_pc_type svd

    test:
          suffix: particle_hdiv_5
          requires: !complex
          args: -dm_swarm_num_particles 100 -particleRHS -sol_type particles \
                -fieldsplit_q_pc_type lu -fieldsplit_phi_pc_type svd

TEST*/
