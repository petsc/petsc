static char help[] = "Tests multifield and multicomponent L2 projection.\n";

#include <petscdmswarm.h>
#include <petscksp.h>
#include <petscdmplex.h>
#include <petscds.h>

typedef struct {
  PetscBool           grad;  // Test gradient projection
  PetscBool           pass;  // Don't fail when moments are wrong
  PetscBool           fv;    // Use an FV discretization, instead of FE
  PetscInt            Npc;   // The number of partices per cell
  PetscInt            field; // The field to project
  PetscInt            Nm;    // The number of moments to match
  PetscReal           mtol;  // Tolerance for checking moment conservation
  PetscSimplePointFn *func;  // Function used to set particle weights
} AppCtx;

typedef enum {
  FUNCTION_CONSTANT,
  FUNCTION_LINEAR,
  FUNCTION_SIN,
  FUNCTION_X2_X4,
  FUNCTION_UNKNOWN,
  NUM_FUNCTIONS
} FunctionType;
const char *const FunctionTypes[] = {"constant", "linear", "sin", "x2_x4", "unknown", "FunctionType", "FUNCTION_", NULL};

static PetscErrorCode constant(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 1.0;
  return PETSC_SUCCESS;
}

static PetscErrorCode linear(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  for (PetscInt d = 0; d < dim; ++d) u[0] += x[d];
  return PETSC_SUCCESS;
}

static PetscErrorCode sinx(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 1;
  for (PetscInt d = 0; d < dim; ++d) u[0] *= PetscSinReal(2. * PETSC_PI * x[d]);
  return PETSC_SUCCESS;
}

static PetscErrorCode x2_x4(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 1;
  for (PetscInt d = 0; d < dim; ++d) u[0] *= PetscSqr(x[d]) - PetscSqr(PetscSqr(x[d]));
  return PETSC_SUCCESS;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  FunctionType func = FUNCTION_LINEAR;
  PetscBool    flg;

  PetscFunctionBeginUser;
  options->grad  = PETSC_FALSE;
  options->pass  = PETSC_FALSE;
  options->fv    = PETSC_FALSE;
  options->Npc   = 1;
  options->field = 0;
  options->Nm    = 1;
  options->mtol  = 100. * PETSC_MACHINE_EPSILON;

  PetscOptionsBegin(comm, "", "L2 Projection Options", "DMPLEX");
  PetscCall(PetscOptionsBool("-grad", "Test gradient projection", __FILE__, options->grad, &options->grad, NULL));
  PetscCall(PetscOptionsBool("-pass", "Don't fail when moments are wrong", __FILE__, options->pass, &options->pass, NULL));
  PetscCall(PetscOptionsBool("-fv", "Use FV instead of FE", __FILE__, options->fv, &options->fv, NULL));
  PetscCall(PetscOptionsBoundedInt("-npc", "Number of particles per cell", __FILE__, options->Npc, &options->Npc, NULL, 0));
  PetscCall(PetscOptionsBoundedInt("-field", "The field to project", __FILE__, options->field, &options->field, NULL, 0));
  PetscCall(PetscOptionsBoundedInt("-moments", "Number of moments to match", __FILE__, options->Nm, &options->Nm, NULL, 0));
  PetscCheck(options->Nm < 4, comm, PETSC_ERR_ARG_OUTOFRANGE, "Cannot match %" PetscInt_FMT " > 3 moments", options->Nm);
  PetscCall(PetscOptionsReal("-mtol", "Tolerance for moment checks", "ex2.c", options->mtol, &options->mtol, NULL));
  PetscCall(PetscOptionsEnum("-func", "Type of particle weight function", __FILE__, FunctionTypes, (PetscEnum)func, (PetscEnum *)&func, &flg));
  switch (func) {
  case FUNCTION_CONSTANT:
    options->func = constant;
    break;
  case FUNCTION_LINEAR:
    options->func = linear;
    break;
  case FUNCTION_SIN:
    options->func = sinx;
    break;
  case FUNCTION_X2_X4:
    options->func = x2_x4;
    break;
  default:
    PetscCheck(flg, comm, PETSC_ERR_ARG_WRONG, "Cannot handle function \"%s\"", FunctionTypes[func]);
  }
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *user)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateDiscretization(DM dm, AppCtx *user)
{
  PetscFE        fe;
  PetscFV        fv;
  DMPolytopeType ct;
  PetscInt       dim, cStart;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  if (user->fv) {
    PetscCall(PetscFVCreate(PETSC_COMM_SELF, &fv));
    PetscCall(PetscObjectSetName((PetscObject)fv, "fv"));
    PetscCall(PetscFVSetNumComponents(fv, 1));
    PetscCall(PetscFVSetSpatialDimension(fv, dim));
    PetscCall(PetscFVCreateDualSpace(fv, ct));
    PetscCall(PetscFVSetFromOptions(fv));
    PetscCall(DMAddField(dm, NULL, (PetscObject)fv));
    PetscCall(PetscFVDestroy(&fv));
    PetscCall(PetscFVCreate(PETSC_COMM_SELF, &fv));
    PetscCall(PetscObjectSetName((PetscObject)fv, "fv2"));
    PetscCall(PetscFVSetNumComponents(fv, dim));
    PetscCall(PetscFVSetSpatialDimension(fv, dim));
    PetscCall(PetscFVCreateDualSpace(fv, ct));
    PetscCall(PetscFVSetFromOptions(fv));
    PetscCall(DMAddField(dm, NULL, (PetscObject)fv));
    PetscCall(PetscFVDestroy(&fv));
  } else {
    PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, 1, ct, NULL, -1, &fe));
    PetscCall(PetscObjectSetName((PetscObject)fe, "fe"));
    PetscCall(DMAddField(dm, NULL, (PetscObject)fe));
    PetscCall(PetscFEDestroy(&fe));
    PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, dim, ct, NULL, -1, &fe));
    PetscCall(PetscObjectSetName((PetscObject)fe, "fe2"));
    PetscCall(DMAddField(dm, NULL, (PetscObject)fe));
    PetscCall(PetscFEDestroy(&fe));
  }
  PetscCall(DMCreateDS(dm));
  if (user->fv) {
    DMLabel  label;
    PetscInt values[1] = {1};

    PetscCall(DMCreateLabel(dm, "ghost"));
    PetscCall(DMGetLabel(dm, "marker", &label));
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "dummy", label, 1, values, 0, 0, NULL, NULL, NULL, NULL, NULL));
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "dummy", label, 1, values, 1, 0, NULL, NULL, NULL, NULL, NULL));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateGradDiscretization(DM dm, AppCtx *user)
{
  PetscFE        fe;
  DMPolytopeType ct;
  PetscInt       dim, cStart;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, dim, ct, NULL, -1, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "fe"));
  PetscCall(DMAddField(dm, NULL, (PetscObject)fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, 2 * dim, ct, NULL, -1, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "fe2"));
  PetscCall(DMAddField(dm, NULL, (PetscObject)fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMCreateDS(dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateSwarm(DM dm, DM *sw, AppCtx *user)
{
  PetscScalar *coords, *wvals, *xvals;
  PetscInt     Npc = user->Npc, dim, Np;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));

  PetscCall(DMCreate(PetscObjectComm((PetscObject)dm), sw));
  PetscCall(PetscObjectSetName((PetscObject)*sw, "Particles"));
  PetscCall(DMSetType(*sw, DMSWARM));
  PetscCall(DMSetDimension(*sw, dim));
  PetscCall(DMSwarmSetType(*sw, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*sw, dm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "w_q", 1, PETSC_SCALAR));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "x_q", 2, PETSC_SCALAR));
  PetscCall(DMSwarmFinalizeFieldRegister(*sw));
  PetscCall(DMSwarmInsertPointsUsingCellDM(*sw, DMSWARMPIC_LAYOUT_GAUSS, Npc));
  PetscCall(DMSetFromOptions(*sw));

  PetscCall(DMSwarmGetLocalSize(*sw, &Np));
  PetscCall(DMSwarmGetField(*sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(*sw, "w_q", NULL, NULL, (void **)&wvals));
  PetscCall(DMSwarmGetField(*sw, "x_q", NULL, NULL, (void **)&xvals));
  for (PetscInt p = 0; p < Np; ++p) {
    PetscCall(user->func(dim, 0., &coords[p * dim], 1, &wvals[p], user));
    for (PetscInt c = 0; c < 2; ++c) PetscCall(user->func(dim, 0., &coords[p * dim], 1, &xvals[p * 2 + c], user));
  }
  PetscCall(DMSwarmRestoreField(*sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(*sw, "w_q", NULL, NULL, (void **)&wvals));
  PetscCall(DMSwarmRestoreField(*sw, "x_q", NULL, NULL, (void **)&xvals));

  PetscCall(DMViewFromOptions(*sw, NULL, "-sw_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode computeParticleMoments(DM sw, Vec u, PetscReal moments[3], AppCtx *user)
{
  DM                 dm;
  const PetscReal   *coords;
  const PetscScalar *w;
  PetscReal          mom[3] = {0.0, 0.0, 0.0};
  PetscInt           dim, cStart, cEnd, Nc;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetCellDM(sw, &dm));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMSwarmSortGetAccess(sw));
  PetscCall(DMSwarmGetFieldInfo(sw, user->field ? "x_q" : "w_q", &Nc, NULL));
  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(VecGetArrayRead(u, &w));
  for (PetscInt cell = cStart; cell < cEnd; ++cell) {
    PetscInt *pidx;
    PetscInt  Np;

    PetscCall(DMSwarmSortGetPointsPerCell(sw, cell, &Np, &pidx));
    for (PetscInt p = 0; p < Np; ++p) {
      const PetscInt   idx = pidx[p];
      const PetscReal *x   = &coords[idx * dim];

      for (PetscInt c = 0; c < Nc; ++c) {
        mom[0] += PetscRealPart(w[idx * Nc + c]);
        mom[1] += PetscRealPart(w[idx * Nc + c]) * x[0];
        for (PetscInt d = 0; d < dim; ++d) mom[2] += PetscRealPart(w[idx * Nc + c]) * PetscSqr(x[d]);
      }
    }
    PetscCall(DMSwarmSortRestorePointsPerCell(sw, cell, &Np, &pidx));
  }
  PetscCall(VecRestoreArrayRead(u, &w));
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmSortRestoreAccess(sw));
  PetscCallMPI(MPIU_Allreduce(mom, moments, 3, MPIU_REAL, MPI_SUM, PetscObjectComm((PetscObject)sw)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void f0_1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscInt Nc = uOff[1] - uOff[0];

  for (PetscInt c = 0; c < Nc; ++c) f0[0] += u[c];
}

static void f0_x(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscInt Nc = uOff[1] - uOff[0];

  for (PetscInt c = 0; c < Nc; ++c) f0[0] += x[0] * u[c];
}

static void f0_r2(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscInt Nc = uOff[1] - uOff[0];

  for (PetscInt c = 0; c < Nc; ++c)
    for (PetscInt d = 0; d < dim; ++d) f0[0] += PetscSqr(x[d]) * u[c];
}

static PetscErrorCode computeFieldMoments(DM dm, Vec u, PetscReal moments[3], AppCtx *user)
{
  PetscDS     ds;
  PetscScalar mom;

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetObjective(ds, 0, &f0_1));
  mom = 0.;
  PetscCall(DMPlexComputeIntegralFEM(dm, u, &mom, user));
  moments[0] = PetscRealPart(mom);
  PetscCall(PetscDSSetObjective(ds, 0, &f0_x));
  mom = 0.;
  PetscCall(DMPlexComputeIntegralFEM(dm, u, &mom, user));
  moments[1] = PetscRealPart(mom);
  PetscCall(PetscDSSetObjective(ds, 0, &f0_r2));
  mom = 0.;
  PetscCall(DMPlexComputeIntegralFEM(dm, u, &mom, user));
  moments[2] = PetscRealPart(mom);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestParticlesToField(DM sw, DM dm, Vec fhat, AppCtx *user)
{
  const char *fieldnames[1] = {user->field ? "x_q" : "w_q"};
  Vec         fields[1]     = {fhat}, f;
  PetscReal   pmoments[3]; // \int f, \int x f, \int r^2 f
  PetscReal   fmoments[3]; // \int \hat f, \int x \hat f, \int r^2 \hat f

  PetscFunctionBeginUser;
  PetscCall(DMSwarmProjectFields(sw, dm, 1, fieldnames, fields, SCATTER_FORWARD));

  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, fieldnames[0], &f));
  PetscCall(computeParticleMoments(sw, f, pmoments, user));
  PetscCall(VecViewFromOptions(f, NULL, "-f_view"));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, fieldnames[0], &f));
  PetscCall(computeFieldMoments(dm, fhat, fmoments, user));
  PetscCall(VecViewFromOptions(fhat, NULL, "-fhat_view"));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "L2 projection mass: %20.10e, x-momentum: %20.10e, energy: %20.10e.\n", fmoments[0], fmoments[1], fmoments[2]));
  for (PetscInt m = 0; m < user->Nm; ++m) {
    if (user->pass) {
      if (PetscAbsReal((fmoments[m] - pmoments[m]) / fmoments[m]) > user->mtol) PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "p  projection mass: %20.10e, x-momentum: %20.10e, energy: %20.10e.\n", pmoments[0], pmoments[1], pmoments[2]));
    } else {
      PetscCheck(PetscAbsReal((fmoments[m] - pmoments[m]) / fmoments[m]) <= user->mtol, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Moment %" PetscInt_FMT " error too large %g > %g", m, PetscAbsReal((fmoments[m] - pmoments[m]) / fmoments[m]),
                 user->mtol);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestFieldToParticles(DM sw, DM dm, Vec fhat, AppCtx *user)
{
  const char *fieldnames[1] = {user->field ? "x_q" : "w_q"};
  Vec         fields[1]     = {fhat}, f;
  PetscReal   pmoments[3]; // \int f, \int x f, \int r^2 f
  PetscReal   fmoments[3]; // \int \hat f, \int x \hat f, \int r^2 \hat f

  PetscFunctionBeginUser;
  PetscCall(DMSwarmProjectFields(sw, dm, 1, fieldnames, fields, SCATTER_REVERSE));

  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, fieldnames[0], &f));
  PetscCall(computeParticleMoments(sw, f, pmoments, user));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, fieldnames[0], &f));
  PetscCall(computeFieldMoments(dm, fhat, fmoments, user));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "L2 projection mass: %20.10e, x-momentum: %20.10e, energy: %20.10e.\n", fmoments[0], fmoments[1], fmoments[2]));
  for (PetscInt m = 0; m < user->Nm; ++m) {
    if (user->pass) {
      if (PetscAbsReal((fmoments[m] - pmoments[m]) / fmoments[m]) > user->mtol) PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "p  projection mass: %20.10e, x-momentum: %20.10e, energy: %20.10e.\n", pmoments[0], pmoments[1], pmoments[2]));
    } else {
      PetscCheck(PetscAbsReal((fmoments[m] - pmoments[m]) / fmoments[m]) <= user->mtol, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Moment %" PetscInt_FMT " error too large %g > %g", m, PetscAbsReal((fmoments[m] - pmoments[m]) / fmoments[m]),
                 user->mtol);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestParticlesToGradientField(DM sw, DM dm, Vec fhat, AppCtx *user)
{
  const char *fieldnames[1] = {"x_q"};
  Vec         fields[1]     = {fhat}, f;
  PetscReal   pmoments[3]; // \int f, \int x f, \int r^2 f
  PetscReal   fmoments[3]; // \int \hat f, \int x \hat f, \int r^2 \hat f

  PetscFunctionBeginUser;
  PetscCall(DMSwarmProjectGradientFields(sw, dm, 1, fieldnames, fields, SCATTER_FORWARD));

  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, fieldnames[0], &f));
  PetscCall(computeParticleMoments(sw, f, pmoments, user));
  PetscCall(VecViewFromOptions(f, NULL, "-f_view"));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, fieldnames[0], &f));
  PetscCall(computeFieldMoments(dm, fhat, fmoments, user));
  PetscCall(VecViewFromOptions(fhat, NULL, "-fhat_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  DM     dm, subdm, sw;
  Vec    fhat;
  IS     subis;
  AppCtx user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &dm, &user));
  PetscCall(CreateDiscretization(dm, &user));
  PetscCall(CreateSwarm(dm, &sw, &user));

  PetscCall(DMCreateSubDM(dm, 1, &user.field, &subis, &subdm));

  PetscCall(DMGetGlobalVector(subdm, &fhat));
  PetscCall(PetscObjectSetName((PetscObject)fhat, "FEM f"));
  PetscCall(TestParticlesToField(sw, subdm, fhat, &user));
  PetscCall(TestFieldToParticles(sw, subdm, fhat, &user));
  PetscCall(DMRestoreGlobalVector(subdm, &fhat));

  if (user.grad) {
    DM dmGrad, gsubdm;
    IS gsubis;

    PetscCall(DMClone(dm, &dmGrad));
    PetscCall(CreateGradDiscretization(dmGrad, &user));
    PetscCall(DMCreateSubDM(dmGrad, 1, &user.field, &gsubis, &gsubdm));

    PetscCall(DMGetGlobalVector(gsubdm, &fhat));
    PetscCall(PetscObjectSetName((PetscObject)fhat, "FEM grad f"));
    PetscCall(TestParticlesToGradientField(sw, subdm, fhat, &user));
    //PetscCall(TestFieldToParticles(sw, subdm, fhat, &user));
    PetscCall(DMRestoreGlobalVector(gsubdm, &fhat));
    PetscCall(ISDestroy(&gsubis));
    PetscCall(DMDestroy(&gsubdm));
    PetscCall(DMDestroy(&dmGrad));
  }

  PetscCall(ISDestroy(&subis));
  PetscCall(DMDestroy(&subdm));
  PetscCall(DMDestroy(&dm));
  PetscCall(DMDestroy(&sw));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  # Swarm does not handle complex or quad
  build:
    requires: !complex double

  testset:
    requires: triangle
    args: -dm_refine 1 -petscspace_degree 2 -moments 3 \
          -ptof_pc_type lu  \
          -ftop_ksp_rtol 1e-15 -ftop_ksp_type lsqr -ftop_pc_type none

    test:
      suffix: tri_fe_f0
      args: -field 0

    test:
      suffix: tri_fe_f1
      args: -field 1

    test:
      suffix: tri_fe_grad
      args: -field 0 -grad -gptof_ksp_type lsqr -gptof_pc_type none -gptof_ksp_rtol 1e-10

    # -gptof_ksp_converged_reason -gptof_ksp_lsqr_monitor to see the divergence solve
    test:
      suffix: quad_fe_f0
      args: -dm_plex_simplex 0 -field 0

    test:
      suffix: quad_fe_f1
      args: -dm_plex_simplex 0 -field 1

  testset:
    requires: triangle
    args: -dm_refine 1 -moments 1 -fv \
          -ptof_pc_type lu \
          -ftop_ksp_rtol 1e-15 -ftop_ksp_type lsqr -ftop_pc_type none

    test:
      suffix: tri_fv_f0
      args: -field 0

    test:
      suffix: tri_fv_f1
      args: -field 1

    test:
      suffix: quad_fv_f0
      args: -dm_plex_simplex 0 -field 0

    test:
      suffix: quad_fv_f1
      args: -dm_plex_simplex 0 -field 1

TEST*/
