static char help[] = "Tests L2 projection with DMSwarm using delta function particles.\n";

#include <petscdmplex.h>
#include <petscfe.h>
#include <petscdmswarm.h>
#include <petscds.h>
#include <petscksp.h>
#include <petsc/private/petscfeimpl.h>
typedef struct {
  char      meshFilename[PETSC_MAX_PATH_LEN]; /* Name of the mesh filename if any */
  PetscReal L[3];                             /* Dimensions of the mesh bounding box */
  PetscInt  particlesPerCell;                 /* The number of partices per cell */
  PetscReal particleRelDx;                    /* Relative particle position perturbation compared to average cell diameter h */
  PetscReal meshRelDx;                        /* Relative vertex position perturbation compared to average cell diameter h */
  PetscInt  k;                                /* Mode number for test function */
  PetscReal momentTol;                        /* Tolerance for checking moment conservation */
  PetscBool useBlockDiagPrec;                 /* Use the block diagonal of the normal equations as a preconditioner */
  PetscErrorCode (*func)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *);
} AppCtx;

/* const char *const ex2FunctionTypes[] = {"linear","x2_x4","sin","ex2FunctionTypes","EX2_FUNCTION_",0}; */

static PetscErrorCode linear(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *a_ctx)
{
  AppCtx  *ctx = (AppCtx *) a_ctx;
  PetscInt d;

  u[0] = 0.0;
  for (d = 0; d < dim; ++d) u[0] += x[d]/(ctx->L[d]);
  return 0;
}

static PetscErrorCode x2_x4(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *a_ctx)
{
  AppCtx  *ctx = (AppCtx *) a_ctx;
  PetscInt d;

  u[0] = 1;
  for (d = 0; d < dim; ++d) u[0] *= PetscSqr(x[d])*PetscSqr(ctx->L[d]) - PetscPowRealInt(x[d], 4);
  return 0;
}

static PetscErrorCode sinx(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *a_ctx)
{
  AppCtx *ctx = (AppCtx *) a_ctx;

  u[0] = PetscSinScalar(2*PETSC_PI*ctx->k*x[0]/(ctx->L[0]));
  return 0;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  char           fstring[PETSC_MAX_PATH_LEN] = "linear";
  PetscBool      flag;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->particlesPerCell = 1;
  options->k                = 1;
  options->particleRelDx    = 1.e-20;
  options->meshRelDx        = 1.e-20;
  options->momentTol        = 100.*PETSC_MACHINE_EPSILON;
  options->useBlockDiagPrec = PETSC_FALSE;
  ierr = PetscStrcpy(options->meshFilename, "");CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm, "", "L2 Projection Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsString("-mesh", "Name of the mesh filename if any", "ex2.c", options->meshFilename, options->meshFilename, sizeof(options->meshFilename), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-k", "Mode number of test", "ex2.c", options->k, &options->k, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-particlesPerCell", "Number of particles per cell", "ex2.c", options->particlesPerCell, &options->particlesPerCell, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-particle_perturbation", "Relative perturbation of particles (0,1)", "ex2.c", options->particleRelDx, &options->particleRelDx, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mesh_perturbation", "Relative perturbation of mesh points (0,1)", "ex2.c", options->meshRelDx, &options->meshRelDx, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-function", "Name of test function", "ex2.c", fstring, fstring, sizeof(fstring), NULL);CHKERRQ(ierr);
  ierr = PetscStrcmp(fstring, "linear", &flag);CHKERRQ(ierr);
  if (flag) {
    options->func = linear;
  } else {
    ierr = PetscStrcmp(fstring, "sin", &flag);CHKERRQ(ierr);
    if (flag) {
      options->func = sinx;
    } else {
      ierr = PetscStrcmp(fstring, "x2_x4", &flag);CHKERRQ(ierr);
      options->func = x2_x4;
      if (!flag) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown function %s",fstring);
    }
  }
  ierr = PetscOptionsReal("-moment_tol", "Tolerance for moment checks", "ex2.c", options->momentTol, &options->momentTol, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-block_diag_prec", "Use the block diagonal of the normal equations to precondition the particle projection", "ex2.c", options->useBlockDiagPrec, &options->useBlockDiagPrec, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  PetscFunctionReturn(0);
}

static PetscErrorCode PerturbVertices(DM dm, AppCtx *user)
{
  PetscRandom    rnd;
  PetscReal      interval = user->meshRelDx;
  Vec            coordinates;
  PetscScalar   *coords;
  PetscReal     *hh, low[3], high[3];
  PetscInt       d, cdim, cEnd, N, p, bs;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetBoundingBox(dm, low, high);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, NULL, &cEnd);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject) dm), &rnd);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rnd, -interval, interval);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rnd);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
  ierr = PetscCalloc1(cdim,&hh);CHKERRQ(ierr);
  for (d = 0; d < cdim; ++d) hh[d] = (user->L[d])/PetscPowReal(cEnd, 1./cdim);
  ierr = VecGetLocalSize(coordinates, &N);CHKERRQ(ierr);
  ierr = VecGetBlockSize(coordinates, &bs);CHKERRQ(ierr);
  if (bs != cdim) SETERRQ2(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_SIZ, "Coordinate vector has wrong block size %D != %D", bs, cdim);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  for (p = 0; p < N; p += cdim) {
    PetscScalar *coord = &coords[p], value;

    for (d = 0; d < cdim; ++d) {
      ierr = PetscRandomGetValue(rnd, &value);CHKERRQ(ierr);
      coord[d] = PetscMax(low[d], PetscMin(high[d], PetscRealPart(coord[d] + value*hh[d])));
    }
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rnd);CHKERRQ(ierr);
  ierr = PetscFree(hh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *user)
{
  PetscReal      low[3], high[3];
  PetscInt       cdim, d;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscStrcmp(user->meshFilename, "", &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DMPlexCreateBoxMesh(comm, 2, PETSC_TRUE, NULL, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateFromFile(comm, user->meshFilename, PETSC_TRUE, dm);CHKERRQ(ierr);
  }
  ierr = DMGetCoordinateDim(*dm, &cdim);CHKERRQ(ierr);
  ierr = DMGetBoundingBox(*dm, low, high);CHKERRQ(ierr);
  for (d = 0; d < cdim; ++d) user->L[d] = high[d] - low[d];
  {
    DM distributedMesh = NULL;

    ierr = DMPlexDistribute(*dm, 0, NULL, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    }
  }
  ierr = DMLocalizeCoordinates(*dm);CHKERRQ(ierr); /* needed for periodic */
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = PerturbVertices(*dm, user);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static void identity(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetCellType(dm, cStart, &ct);CHKERRQ(ierr);
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1, simplex, NULL, -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, "fe");CHKERRQ(ierr);
  ierr = DMSetField(dm, 0, NULL, (PetscObject) fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  /* Setup to form mass matrix */
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(ds, 0, 0, identity, NULL, NULL, NULL);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetCellType(dm, cStart, &ct);CHKERRQ(ierr);
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;
  ierr = DMCreate(PetscObjectComm((PetscObject) dm), sw);CHKERRQ(ierr);
  ierr = DMSetType(*sw, DMSWARM);CHKERRQ(ierr);
  ierr = DMSetDimension(*sw, dim);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PetscObjectComm((PetscObject) dm), &rnd);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rnd, -1.0, 1.0);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rnd);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject) dm), &rndp);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rndp, -interval, interval);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rndp);CHKERRQ(ierr);

  ierr = DMSwarmSetType(*sw, DMSWARM_PIC);CHKERRQ(ierr);
  ierr = DMSwarmSetCellDM(*sw, dm);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(*sw, "w_q", 1, PETSC_SCALAR);CHKERRQ(ierr);
  ierr = DMSwarmFinalizeFieldRegister(*sw);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, NULL, &Ncell);CHKERRQ(ierr);
  ierr = DMSwarmSetLocalSizes(*sw, Ncell * Np, 0);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*sw);CHKERRQ(ierr);
  ierr = DMSwarmGetField(*sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmGetField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
  ierr = DMSwarmGetField(*sw, "w_q", NULL, NULL, (void **) &vals);CHKERRQ(ierr);

  ierr = PetscMalloc5(dim, &centroid, dim, &xi0, dim, &v0, dim*dim, &J, dim*dim, &invJ);CHKERRQ(ierr);
  for (c = 0; c < Ncell; ++c) {
    if (Np == 1) {
      ierr = DMPlexComputeCellGeometryFVM(dm, c, NULL, centroid, NULL);CHKERRQ(ierr);
      cellid[c] = c;
      for (d = 0; d < dim; ++d) coords[c*dim+d] = centroid[d];
    } else {
      ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr); /* affine */
      for (d = 0; d < dim; ++d) xi0[d] = -1.0;
      for (p = 0; p < Np; ++p) {
        const PetscInt n   = c*Np + p;
        PetscReal      sum = 0.0, refcoords[3];

        cellid[n] = c;
        for (d = 0; d < dim; ++d) {ierr = PetscRandomGetValue(rnd, &value);CHKERRQ(ierr); refcoords[d] = PetscRealPart(value); sum += refcoords[d];}
        if (simplex && sum > 0.0) for (d = 0; d < dim; ++d) refcoords[d] -= PetscSqrtReal(dim)*sum;
        CoordinatesRefToReal(dim, dim, xi0, v0, J, refcoords, &coords[n*dim]);
      }
    }
  }
  ierr = PetscFree5(centroid, xi0, v0, J, invJ);CHKERRQ(ierr);
  for (c = 0; c < Ncell; ++c) {
    for (p = 0; p < Np; ++p) {
      const PetscInt n = c*Np + p;

      for (d = 0; d < dim; ++d) {ierr = PetscRandomGetValue(rndp, &value);CHKERRQ(ierr); coords[n*dim+d] += PetscRealPart(value);}
      user->func(dim, 0.0, &coords[n*dim], 1, &vals[c], user);
    }
  }
  ierr = DMSwarmRestoreField(*sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(*sw, "w_q", NULL, NULL, (void **) &vals);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rnd);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rndp);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *sw, "Particles");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*sw, NULL, "-sw_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode computeParticleMoments(DM sw, PetscReal moments[3], AppCtx *user)
{
  DM                 dm;
  const PetscReal   *coords;
  const PetscScalar *w;
  PetscReal          mom[3] = {0.0, 0.0, 0.0};
  PetscInt           cell, cStart, cEnd, dim;
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(sw, &dim);CHKERRQ(ierr);
  ierr = DMSwarmGetCellDM(sw, &dm);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMSwarmSortGetAccess(sw);CHKERRQ(ierr);
  ierr = DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmGetField(sw, "w_q", NULL, NULL, (void **) &w);CHKERRQ(ierr);
  for (cell = cStart; cell < cEnd; ++cell) {
    PetscInt *pidx;
    PetscInt  Np, p, d;

    ierr = DMSwarmSortGetPointsPerCell(sw, cell, &Np, &pidx);CHKERRQ(ierr);
    for (p = 0; p < Np; ++p) {
      const PetscInt   idx = pidx[p];
      const PetscReal *c   = &coords[idx*dim];

      mom[0] += PetscRealPart(w[idx]);
      mom[1] += PetscRealPart(w[idx]) * c[0];
      for (d = 0; d < dim; ++d) mom[2] += PetscRealPart(w[idx]) * c[d]*c[d];
    }
    ierr = PetscFree(pidx);CHKERRQ(ierr);
  }
  ierr = DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **) &w);CHKERRQ(ierr);
  ierr = DMSwarmSortRestoreAccess(sw);CHKERRQ(ierr);
  ierr = MPI_Allreduce(mom, moments, 3, MPIU_REAL, MPI_SUM, PetscObjectComm((PetscObject) sw));CHKERRMPI(ierr);
  PetscFunctionReturn(0);
}

static void f0_1(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = u[0];
}

static void f0_x(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = x[0]*u[0];
}

static void f0_r2(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;

  f0[0] = 0.0;
  for (d = 0; d < dim; ++d) f0[0] += PetscSqr(x[d])*u[0];
}

static PetscErrorCode computeFEMMoments(DM dm, Vec u, PetscReal moments[3], AppCtx *user)
{
  PetscDS        prob;
  PetscErrorCode ierr;
  PetscScalar    mom;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetObjective(prob, 0, &f0_1);CHKERRQ(ierr);
  ierr = DMPlexComputeIntegralFEM(dm, u, &mom, user);CHKERRQ(ierr);
  moments[0] = PetscRealPart(mom);
  ierr = PetscDSSetObjective(prob, 0, &f0_x);CHKERRQ(ierr);
  ierr = DMPlexComputeIntegralFEM(dm, u, &mom, user);CHKERRQ(ierr);
  moments[1] = PetscRealPart(mom);
  ierr = PetscDSSetObjective(prob, 0, &f0_r2);CHKERRQ(ierr);
  ierr = DMPlexComputeIntegralFEM(dm, u, &mom, user);CHKERRQ(ierr);
  moments[2] = PetscRealPart(mom);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestL2ProjectionParticlesToField(DM dm, DM sw, AppCtx *user)
{
  MPI_Comm       comm;
  KSP            ksp;
  Mat            M;            /* FEM mass matrix */
  Mat            M_p;          /* Particle mass matrix */
  Vec            f, rhs, fhat; /* Particle field f, \int phi_i f, FEM field */
  PetscReal      pmoments[3];  /* \int f, \int x f, \int r^2 f */
  PetscReal      fmoments[3];  /* \int \hat f, \int x \hat f, \int r^2 \hat f */
  PetscInt       m;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = KSPCreate(comm, &ksp);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ksp, "ptof_");CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &fhat);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &rhs);CHKERRQ(ierr);

  ierr = DMCreateMassMatrix(sw, dm, &M_p);CHKERRQ(ierr);
  ierr = MatViewFromOptions(M_p, NULL, "-M_p_view");CHKERRQ(ierr);

  /* make particle weight vector */
  ierr = DMSwarmCreateGlobalVectorFromField(sw, "w_q", &f);CHKERRQ(ierr);

  /* create matrix RHS vector */
  ierr = MatMultTranspose(M_p, f, rhs);CHKERRQ(ierr);
  ierr = DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &f);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) rhs,"rhs");CHKERRQ(ierr);
  ierr = VecViewFromOptions(rhs, NULL, "-rhs_view");CHKERRQ(ierr);

  ierr = DMCreateMatrix(dm, &M);CHKERRQ(ierr);
  ierr = DMPlexSNESComputeJacobianFEM(dm, fhat, M, M, user);CHKERRQ(ierr);
  ierr = MatViewFromOptions(M, NULL, "-M_view");CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, M, M);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp, rhs, fhat);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fhat,"fhat");CHKERRQ(ierr);
  ierr = VecViewFromOptions(fhat, NULL, "-fhat_view");CHKERRQ(ierr);

  /* Check moments of field */
  ierr = computeParticleMoments(sw, pmoments, user);CHKERRQ(ierr);
  ierr = computeFEMMoments(dm, fhat, fmoments, user);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "L2 projection mass: %20.10e, x-momentum: %20.10e, energy: %20.10e.\n", fmoments[0], fmoments[1], fmoments[2]);CHKERRQ(ierr);
  for (m = 0; m < 3; ++m) {
    if (PetscAbsReal((fmoments[m] - pmoments[m])/fmoments[m]) > user->momentTol) SETERRQ3(comm, PETSC_ERR_ARG_WRONG, "Moment %D error too large %g > %g", m, PetscAbsReal((fmoments[m] - pmoments[m])/fmoments[m]), user->momentTol);
  }

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&M);CHKERRQ(ierr);
  ierr = MatDestroy(&M_p);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &fhat);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &rhs);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


static PetscErrorCode TestL2ProjectionFieldToParticles(DM dm, DM sw, AppCtx *user)
{

  MPI_Comm       comm;
  KSP            ksp;
  Mat            M;            /* FEM mass matrix */
  Mat            M_p, PM_p;    /* Particle mass matrix M_p, and the preconditioning matrix, e.g. M_p M^T_p */
  Vec            f, rhs, fhat; /* Particle field f, \int phi_i fhat, FEM field */
  PetscReal      pmoments[3];  /* \int f, \int x f, \int r^2 f */
  PetscReal      fmoments[3];  /* \int \hat f, \int x \hat f, \int r^2 \hat f */
  PetscInt       m;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);

  ierr = KSPCreate(comm, &ksp);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ksp, "ftop_");CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  ierr = DMGetGlobalVector(dm, &fhat);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &rhs);CHKERRQ(ierr);

  ierr = DMCreateMassMatrix(sw, dm, &M_p);CHKERRQ(ierr);
  ierr = MatViewFromOptions(M_p, NULL, "-M_p_view");CHKERRQ(ierr);

  /* make particle weight vector */
  ierr = DMSwarmCreateGlobalVectorFromField(sw, "w_q", &f);CHKERRQ(ierr);

  /* create matrix RHS vector, in this case the FEM field fhat with the coefficients vector #alpha */
  ierr = PetscObjectSetName((PetscObject) rhs,"rhs");CHKERRQ(ierr);
  ierr = VecViewFromOptions(rhs, NULL, "-rhs_view");CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm, &M);CHKERRQ(ierr);
  ierr = DMPlexSNESComputeJacobianFEM(dm, fhat, M, M, user);CHKERRQ(ierr);
  ierr = MatViewFromOptions(M, NULL, "-M_view");CHKERRQ(ierr);
  ierr = MatMultTranspose(M, fhat, rhs);CHKERRQ(ierr);
  if (user->useBlockDiagPrec) {ierr = DMSwarmCreateMassMatrixSquare(sw, dm, &PM_p);CHKERRQ(ierr);}
  else                        {ierr = PetscObjectReference((PetscObject) M_p);CHKERRQ(ierr); PM_p = M_p;}

  ierr = KSPSetOperators(ksp, M_p, PM_p);CHKERRQ(ierr);
  ierr = KSPSolveTranspose(ksp, rhs, f);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fhat,"fhat");CHKERRQ(ierr);
  ierr = VecViewFromOptions(fhat, NULL, "-fhat_view");CHKERRQ(ierr);

  ierr = DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &f);CHKERRQ(ierr);

  /* Check moments */
  ierr = computeParticleMoments(sw, pmoments, user);CHKERRQ(ierr);
  ierr = computeFEMMoments(dm, fhat, fmoments, user);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "L2 projection mass: %20.10e, x-momentum: %20.10e, energy: %20.10e.\n", fmoments[0], fmoments[1], fmoments[2]);CHKERRQ(ierr);
  for (m = 0; m < 3; ++m) {
    if (PetscAbsReal((fmoments[m] - pmoments[m])/fmoments[m]) > user->momentTol) SETERRQ3(comm, PETSC_ERR_ARG_WRONG, "Moment %D error too large %g > %g", m, PetscAbsReal((fmoments[m] - pmoments[m])/fmoments[m]), user->momentTol);
  }
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&M);CHKERRQ(ierr);
  ierr = MatDestroy(&M_p);CHKERRQ(ierr);
  ierr = MatDestroy(&PM_p);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &fhat);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &rhs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Interpolate the gradient of an FEM (FVM) field. Code repurposed from DMPlexComputeGradientClementInterpolant */
static PetscErrorCode InterpolateGradient(DM dm, Vec locX, Vec locC){

  DM_Plex         *mesh  = (DM_Plex *) dm->data;
  PetscInt         debug = mesh->printFEM;
  DM               dmC;
  PetscSection     section;
  PetscQuadrature  quad = NULL;
  PetscScalar     *interpolant, *gradsum;
  PetscFEGeom      fegeom;
  PetscReal       *coords;
  const PetscReal *quadPoints, *quadWeights;
  PetscInt         dim, coordDim, numFields, numComponents = 0, qNc, Nq, cStart, cEnd, vStart, vEnd, v, field, fieldOffset;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = VecGetDM(locC, &dmC);CHKERRQ(ierr);
  ierr = VecSet(locC, 0.0);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &coordDim);CHKERRQ(ierr);
  fegeom.dimEmbed = coordDim;
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
  for (field = 0; field < numFields; ++field) {
    PetscObject  obj;
    PetscClassId id;
    PetscInt     Nc;

    ierr = DMGetField(dm, field, NULL, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {
      PetscFE fe = (PetscFE) obj;

      ierr = PetscFEGetQuadrature(fe, &quad);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
    } else if (id == PETSCFV_CLASSID) {
      PetscFV fv = (PetscFV) obj;

      ierr = PetscFVGetQuadrature(fv, &quad);CHKERRQ(ierr);
      ierr = PetscFVGetNumComponents(fv, &Nc);CHKERRQ(ierr);
    } else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %d", field);
    numComponents += Nc;
  }
  ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
  if ((qNc != 1) && (qNc != numComponents)) SETERRQ2(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_SIZ, "Quadrature components %D != %D field components", qNc, numComponents);
  ierr = PetscMalloc6(coordDim*numComponents*2,&gradsum,coordDim*numComponents,&interpolant,coordDim*Nq,&coords,Nq,&fegeom.detJ,coordDim*coordDim*Nq,&fegeom.J,coordDim*coordDim*Nq,&fegeom.invJ);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    PetscInt   *star = NULL;
    PetscInt    starSize, st, d, fc;

    ierr = PetscArrayzero(gradsum, coordDim*numComponents);CHKERRQ(ierr);
    ierr = DMPlexGetTransitiveClosure(dm, v, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
    for (st = 0; st < starSize*2; st += 2) {
      const PetscInt cell = star[st];
      PetscScalar   *grad = &gradsum[coordDim*numComponents];
      PetscScalar   *x    = NULL;

      if ((cell < cStart) || (cell >= cEnd)) continue;
      ierr = DMPlexComputeCellGeometryFEM(dm, cell, quad, coords, fegeom.J, fegeom.invJ, fegeom.detJ);CHKERRQ(ierr);
      ierr = DMPlexVecGetClosure(dm, NULL, locX, cell, NULL, &x);CHKERRQ(ierr);
      for (field = 0, fieldOffset = 0; field < numFields; ++field) {
        PetscObject  obj;
        PetscClassId id;
        PetscInt     Nb, Nc, q, qc = 0;

        ierr = PetscArrayzero(grad, coordDim*numComponents);CHKERRQ(ierr);
        ierr = DMGetField(dm, field, NULL, &obj);CHKERRQ(ierr);
        ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
        if (id == PETSCFE_CLASSID)      {ierr = PetscFEGetNumComponents((PetscFE) obj, &Nc);CHKERRQ(ierr);ierr = PetscFEGetDimension((PetscFE) obj, &Nb);CHKERRQ(ierr);}
        else if (id == PETSCFV_CLASSID) {ierr = PetscFVGetNumComponents((PetscFV) obj, &Nc);CHKERRQ(ierr);Nb = 1;}
        else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %d", field);
        for (q = 0; q < Nq; ++q) {
          if (fegeom.detJ[q] <= 0.0) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %D, quadrature points %D", (double)fegeom.detJ[q], cell, q);
          if (ierr) {
            PetscErrorCode ierr2;
            ierr2 = DMPlexVecRestoreClosure(dm, NULL, locX, cell, NULL, &x);CHKERRQ(ierr2);
            ierr2 = DMPlexRestoreTransitiveClosure(dm, v, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr2);
            ierr2 = PetscFree6(gradsum,interpolant,coords,fegeom.detJ,fegeom.J,fegeom.invJ);CHKERRQ(ierr2);
            CHKERRQ(ierr);
          }
          if (id == PETSCFE_CLASSID)      {ierr = PetscFEInterpolateGradient_Static((PetscFE) obj, &x[fieldOffset], &fegeom, q, interpolant);CHKERRQ(ierr);}
          else SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %d", field);
          for (fc = 0; fc < Nc; ++fc) {
            const PetscReal wt = quadWeights[q*qNc+qc+fc];

            for (d = 0; d < coordDim; ++d) grad[fc*coordDim+d] += interpolant[fc*dim+d]*wt*fegeom.detJ[q];
          }
        }
        fieldOffset += Nb;
      }
      ierr = DMPlexVecRestoreClosure(dm, NULL, locX, cell, NULL, &x);CHKERRQ(ierr);
      for (fc = 0; fc < numComponents; ++fc) {
        for (d = 0; d < coordDim; ++d) {
          gradsum[fc*coordDim+d] += grad[fc*coordDim+d];
        }
      }
      if (debug) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "Cell %D gradient: [", cell);CHKERRQ(ierr);
        for (fc = 0; fc < numComponents; ++fc) {
          for (d = 0; d < coordDim; ++d) {
            if (fc || d > 0) {ierr = PetscPrintf(PETSC_COMM_SELF, ", ");CHKERRQ(ierr);}
            ierr = PetscPrintf(PETSC_COMM_SELF, "%g", (double)PetscRealPart(grad[fc*coordDim+d]));CHKERRQ(ierr);
          }
        }
        ierr = PetscPrintf(PETSC_COMM_SELF, "]\n");CHKERRQ(ierr);
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, v, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
    ierr = DMPlexVecSetClosure(dmC, NULL, locC, v, gradsum, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree6(gradsum,interpolant,coords,fegeom.detJ,fegeom.J,fegeom.invJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestFieldGradientProjection(DM dm, DM sw, AppCtx *user)
{

  MPI_Comm       comm;
  KSP            ksp;
  Mat            M;                   /* FEM mass matrix */
  Mat            M_p;                 /* Particle mass matrix */
  Vec            f, rhs, fhat, grad;  /* Particle field f, \int phi_i f, FEM field */
  PetscReal      pmoments[3];         /* \int f, \int x f, \int r^2 f */
  PetscReal      fmoments[3];         /* \int \hat f, \int x \hat f, \int r^2 \hat f */
  PetscInt       m;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = KSPCreate(comm, &ksp);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ksp, "ptof_");CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &fhat);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &rhs);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &grad);CHKERRQ(ierr);

  ierr = DMCreateMassMatrix(sw, dm, &M_p);CHKERRQ(ierr);
  ierr = MatViewFromOptions(M_p, NULL, "-M_p_view");CHKERRQ(ierr);

  /* make particle weight vector */
  ierr = DMSwarmCreateGlobalVectorFromField(sw, "w_q", &f);CHKERRQ(ierr);

  /* create matrix RHS vector */
  ierr = MatMultTranspose(M_p, f, rhs);CHKERRQ(ierr);
  ierr = DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &f);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) rhs,"rhs");CHKERRQ(ierr);
  ierr = VecViewFromOptions(rhs, NULL, "-rhs_view");CHKERRQ(ierr);

  ierr = DMCreateMatrix(dm, &M);CHKERRQ(ierr);
  ierr = DMPlexSNESComputeJacobianFEM(dm, fhat, M, M, user);CHKERRQ(ierr);

  ierr = InterpolateGradient(dm, fhat, grad);CHKERRQ(ierr);

  ierr = MatViewFromOptions(M, NULL, "-M_view");CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, M, M);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp, rhs, grad);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fhat,"fhat");CHKERRQ(ierr);
  ierr = VecViewFromOptions(fhat, NULL, "-fhat_view");CHKERRQ(ierr);

  /* Check moments of field */
  ierr = computeParticleMoments(sw, pmoments, user);CHKERRQ(ierr);
  ierr = computeFEMMoments(dm, grad, fmoments, user);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "L2 projection mass: %20.10e, x-momentum: %20.10e, energy: %20.10e.\n", fmoments[0], fmoments[1], fmoments[2]);CHKERRQ(ierr);
  for (m = 0; m < 3; ++m) {
    if (PetscAbsReal((fmoments[m] - pmoments[m])/fmoments[m]) > user->momentTol) SETERRQ3(comm, PETSC_ERR_ARG_WRONG, "Moment %D error too large %g > %g", m, PetscAbsReal((fmoments[m] - pmoments[m])/fmoments[m]), user->momentTol);
  }

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&M);CHKERRQ(ierr);
  ierr = MatDestroy(&M_p);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &fhat);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &rhs);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &grad);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

int main (int argc, char * argv[]) {
  MPI_Comm       comm;
  DM             dm, sw;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &dm, &user);CHKERRQ(ierr);
  ierr = CreateFEM(dm, &user);CHKERRQ(ierr);
  ierr = CreateParticles(dm, &sw, &user);CHKERRQ(ierr);
  ierr = TestL2ProjectionParticlesToField(dm, sw, &user);CHKERRQ(ierr);
  ierr = TestL2ProjectionFieldToParticles(dm, sw, &user);CHKERRQ(ierr);
  ierr = TestFieldGradientProjection(dm, sw, &user);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&sw);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
    args: -dm_plex_box_simplex 0 -dm_plex_box_faces 1,1 -dm_view -sw_view -petscspace_degree 2 -petscfe_default_quadrature_order {{2 3}} -ptof_pc_type lu  -ftop_ksp_rtol 1e-15 -ftop_ksp_type lsqr -ftop_pc_type none
    filter: grep -v marker | grep -v atomic | grep -v usage

  test:
    suffix: proj_quad_2_faces
    requires: triangle
    args: -dm_plex_box_simplex 0 -dm_plex_box_faces 2,2 -dm_view -sw_view -petscspace_degree 2 -petscfe_default_quadrature_order {{2 3}} -ptof_pc_type lu  -ftop_ksp_rtol 1e-15 -ftop_ksp_type lsqr -ftop_pc_type none
    filter: grep -v marker | grep -v atomic | grep -v usage

  test:
    suffix: proj_tri_5P
    requires: triangle
    args: -dm_plex_box_faces 1,1 -particlesPerCell 5 -dm_view -sw_view -petscspace_degree 2 -petscfe_default_quadrature_order {{2 3}} -ptof_pc_type lu  -ftop_ksp_rtol 1e-15 -ftop_ksp_type lsqr -ftop_pc_type none
    filter: grep -v marker | grep -v atomic | grep -v usage

  test:
    suffix: proj_quad_5P
    requires: triangle
    args: -dm_plex_box_simplex 0 -dm_plex_box_faces 1,1 -particlesPerCell 5 -dm_view -sw_view -petscspace_degree 2 -petscfe_default_quadrature_order {{2 3}} -ptof_pc_type lu  -ftop_ksp_rtol 1e-15 -ftop_ksp_type lsqr -ftop_pc_type none
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
    args: -dm_plex_box_dim 3 -dm_plex_box_faces 1,1,1 -dm_view -sw_view -petscspace_degree 2 -petscfe_default_quadrature_order {{2 3}} -ptof_pc_type lu  -ftop_ksp_rtol 1e-15 -ftop_ksp_type lsqr -ftop_pc_type none
    filter: grep -v marker | grep -v atomic | grep -v usage

  test:
    suffix: proj_tri_3d_2_faces
    requires: ctetgen
    args: -dm_plex_box_dim 3 -dm_plex_box_faces 2,2,2 -dm_view -sw_view -petscspace_degree 2 -petscfe_default_quadrature_order {{2 3}} -ptof_pc_type lu  -ftop_ksp_rtol 1e-15 -ftop_ksp_type lsqr -ftop_pc_type none
    filter: grep -v marker | grep -v atomic | grep -v usage

  test:
    suffix: proj_tri_3d_5P
    requires: ctetgen
    args: -dm_plex_box_dim 3 -dm_plex_box_faces 1,1,1 -particlesPerCell 5 -dm_view -sw_view -petscspace_degree 2 -petscfe_default_quadrature_order {{2 3}} -ptof_pc_type lu  -ftop_ksp_rtol 1e-15 -ftop_ksp_type lsqr -ftop_pc_type none
    filter: grep -v marker | grep -v atomic | grep -v usage

  test:
    suffix: proj_tri_3d_mdx
    requires: ctetgen
    args: -dm_plex_box_dim 3 -dm_plex_box_faces 1,1,1 -mesh_perturbation 1.0e-1 -dm_view -sw_view -petscspace_degree 2 -petscfe_default_quadrature_order {{2 3}} -ptof_pc_type lu  -ftop_ksp_rtol 1e-15 -ftop_ksp_type lsqr -ftop_pc_type none
    filter: grep -v marker | grep -v atomic | grep -v usage

  test:
    suffix: proj_tri_3d_mdx_5P
    requires: ctetgen
    args: -dm_plex_box_dim 3 -dm_plex_box_faces 1,1,1 -particlesPerCell 5 -mesh_perturbation 1.0e-1 -dm_view -sw_view -petscspace_degree 2 -petscfe_default_quadrature_order {{2 3}} -ptof_pc_type lu  -ftop_ksp_rtol 1e-15 -ftop_ksp_type lsqr -ftop_pc_type none
    filter: grep -v marker | grep -v atomic | grep -v usage

  test:
    suffix: proj_tri_3d_mdx_2_faces
    requires: ctetgen
    args: -dm_plex_box_dim 3 -dm_plex_box_faces 2,2,2 -mesh_perturbation 1.0e-1 -dm_view -sw_view -petscspace_degree 2 -petscfe_default_quadrature_order {{2 3}} -ptof_pc_type lu  -ftop_ksp_rtol 1e-15 -ftop_ksp_type lsqr -ftop_pc_type none
    filter: grep -v marker | grep -v atomic | grep -v usage

  test:
    suffix: proj_tri_3d_mdx_5P_2_faces
    requires: ctetgen
    args: -dm_plex_box_dim 3 -dm_plex_box_faces 2,2,2 -particlesPerCell 5 -mesh_perturbation 1.0e-1 -dm_view -sw_view -petscspace_degree 2 -petscfe_default_quadrature_order {{2 3}} -ptof_pc_type lu  -ftop_ksp_rtol 1e-15 -ftop_ksp_type lsqr -ftop_pc_type none
    filter: grep -v marker | grep -v atomic | grep -v usage

  test:
    suffix: proj_quad_lsqr_scale
    requires: !complex
    args: -dm_plex_box_simplex 0 -dm_plex_box_faces 4,4 \
      -petscspace_degree 2 -petscfe_default_quadrature_order 3 \
      -particlesPerCell 200 \
      -ptof_pc_type lu  \
      -ftop_ksp_rtol 1e-17 -ftop_ksp_type lsqr -ftop_pc_type none

  test:
    suffix: proj_quad_lsqr_prec_scale
    requires: !complex
    args: -dm_plex_box_simplex 0 -dm_plex_box_faces 4,4 \
      -petscspace_degree 2 -petscfe_default_quadrature_order 3 \
      -particlesPerCell 200 \
      -ptof_pc_type lu  \
      -ftop_ksp_rtol 1e-17 -ftop_ksp_type lsqr -ftop_pc_type lu -ftop_pc_factor_shift_type nonzero -block_diag_prec

TEST*/
