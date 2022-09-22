static char help[] = "Tests L2 projection with DMSwarm using delta function particles.\n";

#include <petscdmplex.h>
#include <petscfe.h>
#include <petscdmswarm.h>
#include <petscds.h>
#include <petscksp.h>
#include <petsc/private/petscfeimpl.h>
typedef struct {
  PetscReal L[3];             /* Dimensions of the mesh bounding box */
  PetscInt  particlesPerCell; /* The number of partices per cell */
  PetscReal particleRelDx;    /* Relative particle position perturbation compared to average cell diameter h */
  PetscReal meshRelDx;        /* Relative vertex position perturbation compared to average cell diameter h */
  PetscInt  k;                /* Mode number for test function */
  PetscReal momentTol;        /* Tolerance for checking moment conservation */
  PetscBool useBlockDiagPrec; /* Use the block diagonal of the normal equations as a preconditioner */
  PetscErrorCode (*func)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *);
} AppCtx;

/* const char *const ex2FunctionTypes[] = {"linear","x2_x4","sin","ex2FunctionTypes","EX2_FUNCTION_",0}; */

static PetscErrorCode linear(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *a_ctx)
{
  AppCtx  *ctx = (AppCtx *)a_ctx;
  PetscInt d;

  u[0] = 0.0;
  for (d = 0; d < dim; ++d) u[0] += x[d] / (ctx->L[d]);
  return 0;
}

static PetscErrorCode x2_x4(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *a_ctx)
{
  AppCtx  *ctx = (AppCtx *)a_ctx;
  PetscInt d;

  u[0] = 1;
  for (d = 0; d < dim; ++d) u[0] *= PetscSqr(x[d]) * PetscSqr(ctx->L[d]) - PetscPowRealInt(x[d], 4);
  return 0;
}

static PetscErrorCode sinx(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *a_ctx)
{
  AppCtx *ctx = (AppCtx *)a_ctx;

  u[0] = PetscSinScalar(2 * PETSC_PI * ctx->k * x[0] / (ctx->L[0]));
  return 0;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  char      fstring[PETSC_MAX_PATH_LEN] = "linear";
  PetscBool flag;

  PetscFunctionBeginUser;
  options->particlesPerCell = 1;
  options->k                = 1;
  options->particleRelDx    = 1.e-20;
  options->meshRelDx        = 1.e-20;
  options->momentTol        = 100. * PETSC_MACHINE_EPSILON;
  options->useBlockDiagPrec = PETSC_FALSE;

  PetscOptionsBegin(comm, "", "L2 Projection Options", "DMPLEX");
  PetscCall(PetscOptionsInt("-k", "Mode number of test", "ex2.c", options->k, &options->k, NULL));
  PetscCall(PetscOptionsInt("-particlesPerCell", "Number of particles per cell", "ex2.c", options->particlesPerCell, &options->particlesPerCell, NULL));
  PetscCall(PetscOptionsReal("-particle_perturbation", "Relative perturbation of particles (0,1)", "ex2.c", options->particleRelDx, &options->particleRelDx, NULL));
  PetscCall(PetscOptionsReal("-mesh_perturbation", "Relative perturbation of mesh points (0,1)", "ex2.c", options->meshRelDx, &options->meshRelDx, NULL));
  PetscCall(PetscOptionsString("-function", "Name of test function", "ex2.c", fstring, fstring, sizeof(fstring), NULL));
  PetscCall(PetscStrcmp(fstring, "linear", &flag));
  if (flag) {
    options->func = linear;
  } else {
    PetscCall(PetscStrcmp(fstring, "sin", &flag));
    if (flag) {
      options->func = sinx;
    } else {
      PetscCall(PetscStrcmp(fstring, "x2_x4", &flag));
      options->func = x2_x4;
      PetscCheck(flag, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown function %s", fstring);
    }
  }
  PetscCall(PetscOptionsReal("-moment_tol", "Tolerance for moment checks", "ex2.c", options->momentTol, &options->momentTol, NULL));
  PetscCall(PetscOptionsBool("-block_diag_prec", "Use the block diagonal of the normal equations to precondition the particle projection", "ex2.c", options->useBlockDiagPrec, &options->useBlockDiagPrec, NULL));
  PetscOptionsEnd();

  PetscFunctionReturn(0);
}

static PetscErrorCode PerturbVertices(DM dm, AppCtx *user)
{
  PetscRandom  rnd;
  PetscReal    interval = user->meshRelDx;
  Vec          coordinates;
  PetscScalar *coords;
  PetscReal   *hh, low[3], high[3];
  PetscInt     d, cdim, cEnd, N, p, bs;

  PetscFunctionBeginUser;
  PetscCall(DMGetBoundingBox(dm, low, high));
  PetscCall(DMPlexGetHeightStratum(dm, 0, NULL, &cEnd));
  PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)dm), &rnd));
  PetscCall(PetscRandomSetInterval(rnd, -interval, interval));
  PetscCall(PetscRandomSetFromOptions(rnd));
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(PetscCalloc1(cdim, &hh));
  for (d = 0; d < cdim; ++d) hh[d] = (user->L[d]) / PetscPowReal(cEnd, 1. / cdim);
  PetscCall(VecGetLocalSize(coordinates, &N));
  PetscCall(VecGetBlockSize(coordinates, &bs));
  PetscCheck(bs == cdim, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_SIZ, "Coordinate vector has wrong block size %" PetscInt_FMT " != %" PetscInt_FMT, bs, cdim);
  PetscCall(VecGetArray(coordinates, &coords));
  for (p = 0; p < N; p += cdim) {
    PetscScalar *coord = &coords[p], value;

    for (d = 0; d < cdim; ++d) {
      PetscCall(PetscRandomGetValue(rnd, &value));
      coord[d] = PetscMax(low[d], PetscMin(high[d], PetscRealPart(coord[d] + value * hh[d])));
    }
  }
  PetscCall(VecRestoreArray(coordinates, &coords));
  PetscCall(PetscRandomDestroy(&rnd));
  PetscCall(PetscFree(hh));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *user)
{
  PetscReal low[3], high[3];
  PetscInt  cdim, d;

  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));

  PetscCall(DMGetCoordinateDim(*dm, &cdim));
  PetscCall(DMGetBoundingBox(*dm, low, high));
  for (d = 0; d < cdim; ++d) user->L[d] = high[d] - low[d];
  PetscCall(PerturbVertices(*dm, user));
  PetscFunctionReturn(0);
}

static void identity(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = 1.0;
}

static PetscErrorCode CreateFEM(DM dm, AppCtx *user)
{
  PetscFE        fe;
  PetscDS        ds;
  DMPolytopeType ct;
  PetscBool      simplex;
  PetscInt       dim, cStart;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct) + 1 ? PETSC_TRUE : PETSC_FALSE;
  PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject)dm), dim, 1, simplex, NULL, -1, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "fe"));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(PetscFEDestroy(&fe));
  /* Setup to form mass matrix */
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, identity, NULL, NULL, NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateParticles(DM dm, DM *sw, AppCtx *user)
{
  PetscRandom    rnd, rndp;
  PetscReal      interval = user->particleRelDx;
  DMPolytopeType ct;
  PetscBool      simplex;
  PetscScalar    value, *vals;
  PetscReal     *centroid, *coords, *xi0, *v0, *J, *invJ, detJ;
  PetscInt      *cellid;
  PetscInt       Ncell, Np = user->particlesPerCell, p, cStart, c, dim, d;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct) + 1 ? PETSC_TRUE : PETSC_FALSE;
  PetscCall(DMCreate(PetscObjectComm((PetscObject)dm), sw));
  PetscCall(DMSetType(*sw, DMSWARM));
  PetscCall(DMSetDimension(*sw, dim));

  PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)dm), &rnd));
  PetscCall(PetscRandomSetInterval(rnd, -1.0, 1.0));
  PetscCall(PetscRandomSetFromOptions(rnd));
  PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)dm), &rndp));
  PetscCall(PetscRandomSetInterval(rndp, -interval, interval));
  PetscCall(PetscRandomSetFromOptions(rndp));

  PetscCall(DMSwarmSetType(*sw, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*sw, dm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "w_q", 1, PETSC_SCALAR));
  PetscCall(DMSwarmFinalizeFieldRegister(*sw));
  PetscCall(DMPlexGetHeightStratum(dm, 0, NULL, &Ncell));
  PetscCall(DMSwarmSetLocalSizes(*sw, Ncell * Np, 0));
  PetscCall(DMSetFromOptions(*sw));
  PetscCall(DMSwarmGetField(*sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **)&cellid));
  PetscCall(DMSwarmGetField(*sw, "w_q", NULL, NULL, (void **)&vals));

  PetscCall(PetscMalloc5(dim, &centroid, dim, &xi0, dim, &v0, dim * dim, &J, dim * dim, &invJ));
  for (c = 0; c < Ncell; ++c) {
    if (Np == 1) {
      PetscCall(DMPlexComputeCellGeometryFVM(dm, c, NULL, centroid, NULL));
      cellid[c] = c;
      for (d = 0; d < dim; ++d) coords[c * dim + d] = centroid[d];
    } else {
      PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ)); /* affine */
      for (d = 0; d < dim; ++d) xi0[d] = -1.0;
      for (p = 0; p < Np; ++p) {
        const PetscInt n   = c * Np + p;
        PetscReal      sum = 0.0, refcoords[3];

        cellid[n] = c;
        for (d = 0; d < dim; ++d) {
          PetscCall(PetscRandomGetValue(rnd, &value));
          refcoords[d] = PetscRealPart(value);
          sum += refcoords[d];
        }
        if (simplex && sum > 0.0)
          for (d = 0; d < dim; ++d) refcoords[d] -= PetscSqrtReal(dim) * sum;
        CoordinatesRefToReal(dim, dim, xi0, v0, J, refcoords, &coords[n * dim]);
      }
    }
  }
  PetscCall(PetscFree5(centroid, xi0, v0, J, invJ));
  for (c = 0; c < Ncell; ++c) {
    for (p = 0; p < Np; ++p) {
      const PetscInt n = c * Np + p;

      for (d = 0; d < dim; ++d) {
        PetscCall(PetscRandomGetValue(rndp, &value));
        coords[n * dim + d] += PetscRealPart(value);
      }
      user->func(dim, 0.0, &coords[n * dim], 1, &vals[c], user);
    }
  }
  PetscCall(DMSwarmRestoreField(*sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **)&cellid));
  PetscCall(DMSwarmRestoreField(*sw, "w_q", NULL, NULL, (void **)&vals));
  PetscCall(PetscRandomDestroy(&rnd));
  PetscCall(PetscRandomDestroy(&rndp));
  PetscCall(PetscObjectSetName((PetscObject)*sw, "Particles"));
  PetscCall(DMViewFromOptions(*sw, NULL, "-sw_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode computeParticleMoments(DM sw, PetscReal moments[3], AppCtx *user)
{
  DM                 dm;
  const PetscReal   *coords;
  const PetscScalar *w;
  PetscReal          mom[3] = {0.0, 0.0, 0.0};
  PetscInt           cell, cStart, cEnd, dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetCellDM(sw, &dm));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMSwarmSortGetAccess(sw));
  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&w));
  for (cell = cStart; cell < cEnd; ++cell) {
    PetscInt *pidx;
    PetscInt  Np, p, d;

    PetscCall(DMSwarmSortGetPointsPerCell(sw, cell, &Np, &pidx));
    for (p = 0; p < Np; ++p) {
      const PetscInt   idx = pidx[p];
      const PetscReal *c   = &coords[idx * dim];

      mom[0] += PetscRealPart(w[idx]);
      mom[1] += PetscRealPart(w[idx]) * c[0];
      for (d = 0; d < dim; ++d) mom[2] += PetscRealPart(w[idx]) * c[d] * c[d];
    }
    PetscCall(PetscFree(pidx));
  }
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&w));
  PetscCall(DMSwarmSortRestoreAccess(sw));
  PetscCallMPI(MPI_Allreduce(mom, moments, 3, MPIU_REAL, MPI_SUM, PetscObjectComm((PetscObject)sw)));
  PetscFunctionReturn(0);
}

static void f0_1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = u[0];
}

static void f0_x(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = x[0] * u[0];
}

static void f0_r2(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;

  f0[0] = 0.0;
  for (d = 0; d < dim; ++d) f0[0] += PetscSqr(x[d]) * u[0];
}

static PetscErrorCode computeFEMMoments(DM dm, Vec u, PetscReal moments[3], AppCtx *user)
{
  PetscDS     prob;
  PetscScalar mom;

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &prob));
  PetscCall(PetscDSSetObjective(prob, 0, &f0_1));
  PetscCall(DMPlexComputeIntegralFEM(dm, u, &mom, user));
  moments[0] = PetscRealPart(mom);
  PetscCall(PetscDSSetObjective(prob, 0, &f0_x));
  PetscCall(DMPlexComputeIntegralFEM(dm, u, &mom, user));
  moments[1] = PetscRealPart(mom);
  PetscCall(PetscDSSetObjective(prob, 0, &f0_r2));
  PetscCall(DMPlexComputeIntegralFEM(dm, u, &mom, user));
  moments[2] = PetscRealPart(mom);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestL2ProjectionParticlesToField(DM dm, DM sw, AppCtx *user)
{
  MPI_Comm  comm;
  KSP       ksp;
  Mat       M;            /* FEM mass matrix */
  Mat       M_p;          /* Particle mass matrix */
  Vec       f, rhs, fhat; /* Particle field f, \int phi_i f, FEM field */
  PetscReal pmoments[3];  /* \int f, \int x f, \int r^2 f */
  PetscReal fmoments[3];  /* \int \hat f, \int x \hat f, \int r^2 \hat f */
  PetscInt  m;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(KSPCreate(comm, &ksp));
  PetscCall(KSPSetOptionsPrefix(ksp, "ptof_"));
  PetscCall(DMGetGlobalVector(dm, &fhat));
  PetscCall(DMGetGlobalVector(dm, &rhs));

  PetscCall(DMCreateMassMatrix(sw, dm, &M_p));
  PetscCall(MatViewFromOptions(M_p, NULL, "-M_p_view"));

  /* make particle weight vector */
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &f));

  /* create matrix RHS vector */
  PetscCall(MatMultTranspose(M_p, f, rhs));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &f));
  PetscCall(PetscObjectSetName((PetscObject)rhs, "rhs"));
  PetscCall(VecViewFromOptions(rhs, NULL, "-rhs_view"));

  PetscCall(DMCreateMatrix(dm, &M));
  PetscCall(DMPlexSNESComputeJacobianFEM(dm, fhat, M, M, user));
  PetscCall(MatViewFromOptions(M, NULL, "-M_view"));
  PetscCall(KSPSetOperators(ksp, M, M));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, rhs, fhat));
  PetscCall(PetscObjectSetName((PetscObject)fhat, "fhat"));
  PetscCall(VecViewFromOptions(fhat, NULL, "-fhat_view"));

  /* Check moments of field */
  PetscCall(computeParticleMoments(sw, pmoments, user));
  PetscCall(computeFEMMoments(dm, fhat, fmoments, user));
  PetscCall(PetscPrintf(comm, "L2 projection mass: %20.10e, x-momentum: %20.10e, energy: %20.10e.\n", fmoments[0], fmoments[1], fmoments[2]));
  for (m = 0; m < 3; ++m) {
    PetscCheck(PetscAbsReal((fmoments[m] - pmoments[m]) / fmoments[m]) <= user->momentTol, comm, PETSC_ERR_ARG_WRONG, "Moment %" PetscInt_FMT " error too large %g > %g", m, PetscAbsReal((fmoments[m] - pmoments[m]) / fmoments[m]), user->momentTol);
  }

  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&M));
  PetscCall(MatDestroy(&M_p));
  PetscCall(DMRestoreGlobalVector(dm, &fhat));
  PetscCall(DMRestoreGlobalVector(dm, &rhs));

  PetscFunctionReturn(0);
}

static PetscErrorCode TestL2ProjectionFieldToParticles(DM dm, DM sw, AppCtx *user)
{
  MPI_Comm  comm;
  KSP       ksp;
  Mat       M;            /* FEM mass matrix */
  Mat       M_p, PM_p;    /* Particle mass matrix M_p, and the preconditioning matrix, e.g. M_p M^T_p */
  Vec       f, rhs, fhat; /* Particle field f, \int phi_i fhat, FEM field */
  PetscReal pmoments[3];  /* \int f, \int x f, \int r^2 f */
  PetscReal fmoments[3];  /* \int \hat f, \int x \hat f, \int r^2 \hat f */
  PetscInt  m;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));

  PetscCall(KSPCreate(comm, &ksp));
  PetscCall(KSPSetOptionsPrefix(ksp, "ftop_"));
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(DMGetGlobalVector(dm, &fhat));
  PetscCall(DMGetGlobalVector(dm, &rhs));

  PetscCall(DMCreateMassMatrix(sw, dm, &M_p));
  PetscCall(MatViewFromOptions(M_p, NULL, "-M_p_view"));

  /* make particle weight vector */
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &f));

  /* create matrix RHS vector, in this case the FEM field fhat with the coefficients vector #alpha */
  PetscCall(PetscObjectSetName((PetscObject)rhs, "rhs"));
  PetscCall(VecViewFromOptions(rhs, NULL, "-rhs_view"));
  PetscCall(DMCreateMatrix(dm, &M));
  PetscCall(DMPlexSNESComputeJacobianFEM(dm, fhat, M, M, user));
  PetscCall(MatViewFromOptions(M, NULL, "-M_view"));
  PetscCall(MatMultTranspose(M, fhat, rhs));
  if (user->useBlockDiagPrec) PetscCall(DMSwarmCreateMassMatrixSquare(sw, dm, &PM_p));
  else {
    PetscCall(PetscObjectReference((PetscObject)M_p));
    PM_p = M_p;
  }

  PetscCall(KSPSetOperators(ksp, M_p, PM_p));
  PetscCall(KSPSolveTranspose(ksp, rhs, f));
  PetscCall(PetscObjectSetName((PetscObject)fhat, "fhat"));
  PetscCall(VecViewFromOptions(fhat, NULL, "-fhat_view"));

  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &f));

  /* Check moments */
  PetscCall(computeParticleMoments(sw, pmoments, user));
  PetscCall(computeFEMMoments(dm, fhat, fmoments, user));
  PetscCall(PetscPrintf(comm, "L2 projection mass: %20.10e, x-momentum: %20.10e, energy: %20.10e.\n", fmoments[0], fmoments[1], fmoments[2]));
  for (m = 0; m < 3; ++m) {
    PetscCheck(PetscAbsReal((fmoments[m] - pmoments[m]) / fmoments[m]) <= user->momentTol, comm, PETSC_ERR_ARG_WRONG, "Moment %" PetscInt_FMT " error too large %g > %g", m, PetscAbsReal((fmoments[m] - pmoments[m]) / fmoments[m]), user->momentTol);
  }
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&M));
  PetscCall(MatDestroy(&M_p));
  PetscCall(MatDestroy(&PM_p));
  PetscCall(DMRestoreGlobalVector(dm, &fhat));
  PetscCall(DMRestoreGlobalVector(dm, &rhs));
  PetscFunctionReturn(0);
}

/* Interpolate the gradient of an FEM (FVM) field. Code repurposed from DMPlexComputeGradientClementInterpolant */
static PetscErrorCode InterpolateGradient(DM dm, Vec locX, Vec locC)
{
  DM_Plex         *mesh  = (DM_Plex *)dm->data;
  PetscInt         debug = mesh->printFEM;
  DM               dmC;
  PetscSection     section;
  PetscQuadrature  quad = NULL;
  PetscScalar     *interpolant, *gradsum;
  PetscFEGeom      fegeom;
  PetscReal       *coords;
  const PetscReal *quadPoints, *quadWeights;
  PetscInt         dim, coordDim, numFields, numComponents = 0, qNc, Nq, cStart, cEnd, vStart, vEnd, v, field, fieldOffset;

  PetscFunctionBegin;
  PetscCall(VecGetDM(locC, &dmC));
  PetscCall(VecSet(locC, 0.0));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &coordDim));
  fegeom.dimEmbed = coordDim;
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(PetscSectionGetNumFields(section, &numFields));
  for (field = 0; field < numFields; ++field) {
    PetscObject  obj;
    PetscClassId id;
    PetscInt     Nc;

    PetscCall(DMGetField(dm, field, NULL, &obj));
    PetscCall(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE)obj;

      PetscCall(PetscFEGetQuadrature(fe, &quad));
      PetscCall(PetscFEGetNumComponents(fe, &Nc));
    } else if (id == PETSCFV_CLASSID) {
      PetscFV fv = (PetscFV)obj;

      PetscCall(PetscFVGetQuadrature(fv, &quad));
      PetscCall(PetscFVGetNumComponents(fv, &Nc));
    } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %" PetscInt_FMT, field);
    numComponents += Nc;
  }
  PetscCall(PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights));
  PetscCheck(!(qNc != 1) || !(qNc != numComponents), PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_SIZ, "Quadrature components %" PetscInt_FMT " != %" PetscInt_FMT " field components", qNc, numComponents);
  PetscCall(PetscMalloc6(coordDim * numComponents * 2, &gradsum, coordDim * numComponents, &interpolant, coordDim * Nq, &coords, Nq, &fegeom.detJ, coordDim * coordDim * Nq, &fegeom.J, coordDim * coordDim * Nq, &fegeom.invJ));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
  for (v = vStart; v < vEnd; ++v) {
    PetscInt *star = NULL;
    PetscInt  starSize, st, d, fc;

    PetscCall(PetscArrayzero(gradsum, coordDim * numComponents));
    PetscCall(DMPlexGetTransitiveClosure(dm, v, PETSC_FALSE, &starSize, &star));
    for (st = 0; st < starSize * 2; st += 2) {
      const PetscInt cell = star[st];
      PetscScalar   *grad = &gradsum[coordDim * numComponents];
      PetscScalar   *x    = NULL;

      if ((cell < cStart) || (cell >= cEnd)) continue;
      PetscCall(DMPlexComputeCellGeometryFEM(dm, cell, quad, coords, fegeom.J, fegeom.invJ, fegeom.detJ));
      PetscCall(DMPlexVecGetClosure(dm, NULL, locX, cell, NULL, &x));
      for (field = 0, fieldOffset = 0; field < numFields; ++field) {
        PetscObject  obj;
        PetscClassId id;
        PetscInt     Nb, Nc, q, qc = 0;

        PetscCall(PetscArrayzero(grad, coordDim * numComponents));
        PetscCall(DMGetField(dm, field, NULL, &obj));
        PetscCall(PetscObjectGetClassId(obj, &id));
        if (id == PETSCFE_CLASSID) {
          PetscCall(PetscFEGetNumComponents((PetscFE)obj, &Nc));
          PetscCall(PetscFEGetDimension((PetscFE)obj, &Nb));
        } else if (id == PETSCFV_CLASSID) {
          PetscCall(PetscFVGetNumComponents((PetscFV)obj, &Nc));
          Nb = 1;
        } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %" PetscInt_FMT, field);
        for (q = 0; q < Nq; ++q) {
          PetscCheck(fegeom.detJ[q] > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %" PetscInt_FMT ", quadrature points %" PetscInt_FMT, (double)fegeom.detJ[q], cell, q);
          if (id == PETSCFE_CLASSID) PetscCall(PetscFEInterpolateGradient_Static((PetscFE)obj, 1, &x[fieldOffset], &fegeom, q, interpolant));
          else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %" PetscInt_FMT, field);
          for (fc = 0; fc < Nc; ++fc) {
            const PetscReal wt = quadWeights[q * qNc + qc + fc];

            for (d = 0; d < coordDim; ++d) grad[fc * coordDim + d] += interpolant[fc * dim + d] * wt * fegeom.detJ[q];
          }
        }
        fieldOffset += Nb;
      }
      PetscCall(DMPlexVecRestoreClosure(dm, NULL, locX, cell, NULL, &x));
      for (fc = 0; fc < numComponents; ++fc) {
        for (d = 0; d < coordDim; ++d) gradsum[fc * coordDim + d] += grad[fc * coordDim + d];
      }
      if (debug) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "Cell %" PetscInt_FMT " gradient: [", cell));
        for (fc = 0; fc < numComponents; ++fc) {
          for (d = 0; d < coordDim; ++d) {
            if (fc || d > 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, ", "));
            PetscCall(PetscPrintf(PETSC_COMM_SELF, "%g", (double)PetscRealPart(grad[fc * coordDim + d])));
          }
        }
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "]\n"));
      }
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, v, PETSC_FALSE, &starSize, &star));
    PetscCall(DMPlexVecSetClosure(dmC, NULL, locC, v, gradsum, INSERT_VALUES));
  }
  PetscCall(PetscFree6(gradsum, interpolant, coords, fegeom.detJ, fegeom.J, fegeom.invJ));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestFieldGradientProjection(DM dm, DM sw, AppCtx *user)
{
  MPI_Comm  comm;
  KSP       ksp;
  Mat       M;                  /* FEM mass matrix */
  Mat       M_p;                /* Particle mass matrix */
  Vec       f, rhs, fhat, grad; /* Particle field f, \int phi_i f, FEM field */
  PetscReal pmoments[3];        /* \int f, \int x f, \int r^2 f */
  PetscReal fmoments[3];        /* \int \hat f, \int x \hat f, \int r^2 \hat f */
  PetscInt  m;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(KSPCreate(comm, &ksp));
  PetscCall(KSPSetOptionsPrefix(ksp, "ptof_"));
  PetscCall(DMGetGlobalVector(dm, &fhat));
  PetscCall(DMGetGlobalVector(dm, &rhs));
  PetscCall(DMGetGlobalVector(dm, &grad));

  PetscCall(DMCreateMassMatrix(sw, dm, &M_p));
  PetscCall(MatViewFromOptions(M_p, NULL, "-M_p_view"));

  /* make particle weight vector */
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &f));

  /* create matrix RHS vector */
  PetscCall(MatMultTranspose(M_p, f, rhs));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &f));
  PetscCall(PetscObjectSetName((PetscObject)rhs, "rhs"));
  PetscCall(VecViewFromOptions(rhs, NULL, "-rhs_view"));

  PetscCall(DMCreateMatrix(dm, &M));
  PetscCall(DMPlexSNESComputeJacobianFEM(dm, fhat, M, M, user));

  PetscCall(InterpolateGradient(dm, fhat, grad));

  PetscCall(MatViewFromOptions(M, NULL, "-M_view"));
  PetscCall(KSPSetOperators(ksp, M, M));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, rhs, grad));
  PetscCall(PetscObjectSetName((PetscObject)fhat, "fhat"));
  PetscCall(VecViewFromOptions(fhat, NULL, "-fhat_view"));

  /* Check moments of field */
  PetscCall(computeParticleMoments(sw, pmoments, user));
  PetscCall(computeFEMMoments(dm, grad, fmoments, user));
  PetscCall(PetscPrintf(comm, "L2 projection mass: %20.10e, x-momentum: %20.10e, energy: %20.10e.\n", fmoments[0], fmoments[1], fmoments[2]));
  for (m = 0; m < 3; ++m) {
    PetscCheck(PetscAbsReal((fmoments[m] - pmoments[m]) / fmoments[m]) <= user->momentTol, comm, PETSC_ERR_ARG_WRONG, "Moment %" PetscInt_FMT " error too large %g > %g", m, PetscAbsReal((fmoments[m] - pmoments[m]) / fmoments[m]), user->momentTol);
  }

  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&M));
  PetscCall(MatDestroy(&M_p));
  PetscCall(DMRestoreGlobalVector(dm, &fhat));
  PetscCall(DMRestoreGlobalVector(dm, &rhs));
  PetscCall(DMRestoreGlobalVector(dm, &grad));

  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  MPI_Comm comm;
  DM       dm, sw;
  AppCtx   user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(ProcessOptions(comm, &user));
  PetscCall(CreateMesh(comm, &dm, &user));
  PetscCall(CreateFEM(dm, &user));
  PetscCall(CreateParticles(dm, &sw, &user));
  PetscCall(TestL2ProjectionParticlesToField(dm, sw, &user));
  PetscCall(TestL2ProjectionFieldToParticles(dm, sw, &user));
  PetscCall(TestFieldGradientProjection(dm, sw, &user));
  PetscCall(DMDestroy(&dm));
  PetscCall(DMDestroy(&sw));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  # Swarm does not handle complex or quad
  build:
    requires: !complex double

  test:
    suffix: proj_tri_0
    requires: triangle
    args: -dm_plex_box_faces 1,1 -dm_view -sw_view -petscspace_degree 2 -petscfe_default_quadrature_order {{2 3}} -ptof_pc_type lu  -ftop_ksp_rtol 1e-15 -ftop_ksp_type lsqr -ftop_pc_type none
    filter: grep -v marker | grep -v atomic | grep -v usage

  test:
    suffix: proj_tri_2_faces
    requires: triangle
    args: -dm_plex_box_faces 2,2  -dm_view -sw_view -petscspace_degree 2 -petscfe_default_quadrature_order {{2 3}} -ptof_pc_type lu  -ftop_ksp_rtol 1e-15 -ftop_ksp_type lsqr -ftop_pc_type none
    filter: grep -v marker | grep -v atomic | grep -v usage

  test:
    suffix: proj_quad_0
    requires: triangle
    args: -dm_plex_simplex 0 -dm_plex_box_faces 1,1 -dm_view -sw_view -petscspace_degree 2 -petscfe_default_quadrature_order {{2 3}} -ptof_pc_type lu  -ftop_ksp_rtol 1e-15 -ftop_ksp_type lsqr -ftop_pc_type none
    filter: grep -v marker | grep -v atomic | grep -v usage

  test:
    suffix: proj_quad_2_faces
    requires: triangle
    args: -dm_plex_simplex 0 -dm_plex_box_faces 2,2 -dm_view -sw_view -petscspace_degree 2 -petscfe_default_quadrature_order {{2 3}} -ptof_pc_type lu  -ftop_ksp_rtol 1e-15 -ftop_ksp_type lsqr -ftop_pc_type none
    filter: grep -v marker | grep -v atomic | grep -v usage

  test:
    suffix: proj_tri_5P
    requires: triangle
    args: -dm_plex_box_faces 1,1 -particlesPerCell 5 -dm_view -sw_view -petscspace_degree 2 -petscfe_default_quadrature_order {{2 3}} -ptof_pc_type lu  -ftop_ksp_rtol 1e-15 -ftop_ksp_type lsqr -ftop_pc_type none
    filter: grep -v marker | grep -v atomic | grep -v usage

  test:
    suffix: proj_quad_5P
    requires: triangle
    args: -dm_plex_simplex 0 -dm_plex_box_faces 1,1 -particlesPerCell 5 -dm_view -sw_view -petscspace_degree 2 -petscfe_default_quadrature_order {{2 3}} -ptof_pc_type lu  -ftop_ksp_rtol 1e-15 -ftop_ksp_type lsqr -ftop_pc_type none
    filter: grep -v marker | grep -v atomic | grep -v usage

  test:
    suffix: proj_tri_mdx
    requires: triangle
    args: -dm_plex_box_faces 1,1 -mesh_perturbation 1.0e-1 -dm_view -sw_view -petscspace_degree 2 -petscfe_default_quadrature_order {{2 3}} -ptof_pc_type lu  -ftop_ksp_rtol 1e-15 -ftop_ksp_type lsqr -ftop_pc_type none
    filter: grep -v marker | grep -v atomic | grep -v usage

  test:
    suffix: proj_tri_mdx_5P
    requires: triangle
    args: -dm_plex_box_faces 1,1 -particlesPerCell 5 -mesh_perturbation 1.0e-1 -dm_view -sw_view -petscspace_degree 2 -petscfe_default_quadrature_order {{2 3}} -ptof_pc_type lu  -ftop_ksp_rtol 1e-15 -ftop_ksp_type lsqr -ftop_pc_type none
    filter: grep -v marker | grep -v atomic | grep -v usage

  test:
    suffix: proj_tri_3d
    requires: ctetgen
    args: -dm_plex_dim 3 -dm_plex_box_faces 1,1,1 -dm_view -sw_view -petscspace_degree 2 -petscfe_default_quadrature_order {{2 3}} -ptof_pc_type lu  -ftop_ksp_rtol 1e-15 -ftop_ksp_type lsqr -ftop_pc_type none
    filter: grep -v marker | grep -v atomic | grep -v usage

  test:
    suffix: proj_tri_3d_2_faces
    requires: ctetgen
    args: -dm_plex_dim 3 -dm_plex_box_faces 2,2,2 -dm_view -sw_view -petscspace_degree 2 -petscfe_default_quadrature_order {{2 3}} -ptof_pc_type lu  -ftop_ksp_rtol 1e-15 -ftop_ksp_type lsqr -ftop_pc_type none
    filter: grep -v marker | grep -v atomic | grep -v usage

  test:
    suffix: proj_tri_3d_5P
    requires: ctetgen
    args: -dm_plex_dim 3 -dm_plex_box_faces 1,1,1 -particlesPerCell 5 -dm_view -sw_view -petscspace_degree 2 -petscfe_default_quadrature_order {{2 3}} -ptof_pc_type lu  -ftop_ksp_rtol 1e-15 -ftop_ksp_type lsqr -ftop_pc_type none
    filter: grep -v marker | grep -v atomic | grep -v usage

  test:
    suffix: proj_tri_3d_mdx
    requires: ctetgen
    args: -dm_plex_dim 3 -dm_plex_box_faces 1,1,1 -mesh_perturbation 1.0e-1 -dm_view -sw_view -petscspace_degree 2 -petscfe_default_quadrature_order {{2 3}} -ptof_pc_type lu  -ftop_ksp_rtol 1e-15 -ftop_ksp_type lsqr -ftop_pc_type none
    filter: grep -v marker | grep -v atomic | grep -v usage

  test:
    suffix: proj_tri_3d_mdx_5P
    requires: ctetgen
    args: -dm_plex_dim 3 -dm_plex_box_faces 1,1,1 -particlesPerCell 5 -mesh_perturbation 1.0e-1 -dm_view -sw_view -petscspace_degree 2 -petscfe_default_quadrature_order {{2 3}} -ptof_pc_type lu  -ftop_ksp_rtol 1e-15 -ftop_ksp_type lsqr -ftop_pc_type none
    filter: grep -v marker | grep -v atomic | grep -v usage

  test:
    suffix: proj_tri_3d_mdx_2_faces
    requires: ctetgen
    args: -dm_plex_dim 3 -dm_plex_box_faces 2,2,2 -mesh_perturbation 1.0e-1 -dm_view -sw_view -petscspace_degree 2 -petscfe_default_quadrature_order {{2 3}} -ptof_pc_type lu  -ftop_ksp_rtol 1e-15 -ftop_ksp_type lsqr -ftop_pc_type none
    filter: grep -v marker | grep -v atomic | grep -v usage

  test:
    suffix: proj_tri_3d_mdx_5P_2_faces
    requires: ctetgen
    args: -dm_plex_dim 3 -dm_plex_box_faces 2,2,2 -particlesPerCell 5 -mesh_perturbation 1.0e-1 -dm_view -sw_view -petscspace_degree 2 -petscfe_default_quadrature_order {{2 3}} -ptof_pc_type lu  -ftop_ksp_rtol 1e-15 -ftop_ksp_type lsqr -ftop_pc_type none
    filter: grep -v marker | grep -v atomic | grep -v usage

  test:
    suffix: proj_quad_lsqr_scale
    requires: !complex
    args: -dm_plex_simplex 0 -dm_plex_box_faces 4,4 \
      -petscspace_degree 2 -petscfe_default_quadrature_order 3 \
      -particlesPerCell 200 \
      -ptof_pc_type lu  \
      -ftop_ksp_rtol 1e-17 -ftop_ksp_type lsqr -ftop_pc_type none

  test:
    suffix: proj_quad_lsqr_prec_scale
    requires: !complex
    args: -dm_plex_simplex 0 -dm_plex_box_faces 4,4 \
      -petscspace_degree 2 -petscfe_default_quadrature_order 3 \
      -particlesPerCell 200 \
      -ptof_pc_type lu  \
      -ftop_ksp_rtol 1e-17 -ftop_ksp_type lsqr -ftop_pc_type lu -ftop_pc_factor_shift_type nonzero -block_diag_prec

TEST*/
