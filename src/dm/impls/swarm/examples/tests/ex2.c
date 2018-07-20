static char help[] = "Tests L2 projection with DMSwarm using delta function particles.\n";

#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscds.h>
#include <petscksp.h>
#include <petsc/private/petscfeimpl.h>

typedef struct {
  PetscInt       dim;                              /* The topological mesh dimension */
  PetscBool      simplex;                          /* Flag for simplices or tensor cells */
  char           meshFilename[PETSC_MAX_PATH_LEN]; /* Name of the mesh filename if any */
  PetscInt       faces;                            /* Number of faces per edge if unit square/cube generated */
  PetscReal      domain_lo[3], domain_hi[3];       /* Lower left and upper right mesh corners */
  DMBoundaryType boundary[3];                      /* The domain boundary type, e.g. periodic */
  PetscInt       particlesPerCell;                 /* The number of partices per cell */
  PetscReal      particleRelDx;                    /* Relative particle position perturbation compared to average cell diameter h */
  PetscReal      meshRelDx;                        /* Relative vertex position perturbation compared to average cell diameter h */
  PetscInt       k;                                /* Mode number for test function */
  PetscReal      momentTol;                        /* Tolerance for checking moment conservation */
  PetscErrorCode (*func)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *);
} AppCtx;

/* const char *const ex2FunctionTypes[] = {"linear","x2_x4","sin","ex2FunctionTypes","EX2_FUNCTION_",0}; */

static PetscErrorCode linear(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *a_ctx)
{
  AppCtx  *ctx = (AppCtx *) a_ctx;
  PetscInt d;

  u[0] = 0.0;
  for (d = 0; d < dim; ++d) u[0] += x[d]/(ctx->domain_hi[d] - ctx->domain_lo[d]);
  return 0;
}

static PetscErrorCode x2_x4(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *a_ctx)
{
  AppCtx  *ctx = (AppCtx *) a_ctx;
  PetscInt d;

  u[0] = 1;
  for (d = 0; d < dim; ++d) u[0] *= PetscSqr(x[d])*PetscSqr(ctx->domain_hi[d]) - PetscPowRealInt(x[d], 4);
  return 0;
}

static PetscErrorCode sinx(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *a_ctx)
{
  AppCtx *ctx = (AppCtx *) a_ctx;

  u[0] = sin(2*PETSC_PI*ctx->k*x[0]/(ctx->domain_hi[0] - ctx->domain_lo[0]));
  return 0;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt       ii, bd;
  char           fstring[PETSC_MAX_PATH_LEN] = "linear";
  PetscBool      flag;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim              = 2;
  options->simplex          = PETSC_TRUE;
  options->faces            = 1;
  options->domain_lo[0]     = 0.0;
  options->domain_lo[1]     = 0.0;
  options->domain_lo[2]     = 0.0;
  options->domain_hi[0]     = 1.0;
  options->domain_hi[1]     = 1.0;
  options->domain_hi[2]     = 1.0;
  options->boundary[0]      = DM_BOUNDARY_NONE; /* PERIODIC (plotting does not work in parallel, moments not conserved) */
  options->boundary[1]      = DM_BOUNDARY_NONE;
  options->boundary[2]      = DM_BOUNDARY_NONE;
  options->particlesPerCell = 1;
  options->k                = 1;
  options->particleRelDx    = 1.e-20;
  options->meshRelDx        = 1.e-20;
  options->momentTol        = 10.*PETSC_MACHINE_EPSILON;
  ierr = PetscStrcpy(options->meshFilename, "");CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm, "", "L2 Projection Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex2.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "The flag for simplices or tensor cells", "ex2.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-mesh", "Name of the mesh filename if any", "ex2.c", options->meshFilename, options->meshFilename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-faces", "Number of faces per edge if unit square/cube generated", "ex2.c", options->faces, &options->faces, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-k", "Mode number of test", "ex2.c", options->k, &options->k, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-particlesPerCell", "Number of particles per cell", "ex2.c", options->particlesPerCell, &options->particlesPerCell, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-particle_perturbation", "Relative perturbation of particles (0,1)", "ex2.c", options->particleRelDx, &options->particleRelDx, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mesh_perturbation", "Relative perturbation of mesh points (0,1)", "ex2.c", options->meshRelDx, &options->meshRelDx, NULL);CHKERRQ(ierr);
  ii = options->dim;
  ierr = PetscOptionsRealArray("-domain_hi", "Domain size", "ex2.c", options->domain_hi, &ii, NULL);CHKERRQ(ierr);
  ii = options->dim;
  ierr = PetscOptionsRealArray("-domain_lo", "Domain size", "ex2.c", options->domain_lo, &ii, NULL);CHKERRQ(ierr);
  bd = options->boundary[0];
  ierr = PetscOptionsEList("-x_boundary", "The x-boundary", "ex2.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->boundary[0]], &bd, NULL);CHKERRQ(ierr);
  options->boundary[0] = (DMBoundaryType) bd;
  bd = options->boundary[1];
  ierr = PetscOptionsEList("-y_boundary", "The y-boundary", "ex2.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->boundary[1]], &bd, NULL);CHKERRQ(ierr);
  options->boundary[1] = (DMBoundaryType) bd;
  bd = options->boundary[2];
  ierr = PetscOptionsEList("-z_boundary", "The z-boundary", "ex2.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->boundary[2]], &bd, NULL);CHKERRQ(ierr);
  options->boundary[2] = (DMBoundaryType) bd;
  ierr = PetscOptionsString("-function", "Name of test function", "ex2.c", fstring, fstring, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
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
  ierr = PetscOptionsEnd();

  PetscFunctionReturn(0);
}

static PetscErrorCode PerturbVertices(DM dm, AppCtx *user)
{
  PetscRandom    rnd;
  PetscReal      interval = user->meshRelDx;
  Vec            coordinates;
  PetscScalar   *coords;
  PetscReal      hh[3];
  PetscInt       d, cdim, N, p, bs;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  for (d = 0; d < user->dim; ++d) hh[d] = (user->domain_hi[d] - user->domain_lo[d])/user->faces;
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject) dm), &rnd);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rnd, -interval, interval);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rnd);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
  ierr = VecGetLocalSize(coordinates, &N);CHKERRQ(ierr);
  ierr = VecGetBlockSize(coordinates, &bs);CHKERRQ(ierr);
  if (bs != cdim) SETERRQ2(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_SIZ, "Coordinate vector has wrong block size %D != %D", bs, cdim);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  for (p = 0; p < N; p += cdim) {
    PetscScalar *coord = &coords[p], value;

    for (d = 0; d < cdim; ++d) {
      ierr = PetscRandomGetValue(rnd, &value);CHKERRQ(ierr);
      coord[d] = PetscMax(user->domain_lo[d], PetscMin(user->domain_hi[d], coord[d] + value*hh[d]));
    }
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rnd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *user)
{
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscStrcmp(user->meshFilename, "", &flg);CHKERRQ(ierr);
  if (flg) {
    PetscInt faces[3];

    faces[0] = user->faces; faces[1] = user->faces; faces[2] = user->faces;
    ierr = DMPlexCreateBoxMesh(comm, user->dim, user->simplex, faces, user->domain_lo, user->domain_hi, user->boundary, PETSC_TRUE, dm);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateFromFile(comm, user->meshFilename, PETSC_TRUE, dm);CHKERRQ(ierr);
    ierr = DMGetDimension(*dm, &user->dim);CHKERRQ(ierr);
  }
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
  PetscDS        prob;
  PetscFE        fe;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1, user->simplex, NULL, -1, &fe);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe,"fe");CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  /* Setup to form mass matrix */
  ierr = PetscDSSetJacobian(prob, 0, 0, identity, NULL, NULL, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateParticles(DM dm, DM *sw, AppCtx *user)
{
  PetscRandom    rnd, rndp;
  PetscReal      interval = user->particleRelDx;
  PetscScalar    value, *vals;
  PetscReal     *centroid, *coords, *xi0, *v0, *J, *invJ, detJ;
  PetscInt      *cellid;
  PetscInt       Ncell, Np = user->particlesPerCell, p, c, dim, d;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
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
        if (user->simplex && sum > 0.0) for (d = 0; d < dim; ++d) refcoords[d] -= PetscSqrtReal(dim)*sum;
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

      mom[0] += w[idx];
      mom[1] += w[idx] * c[0];
      for (d = 0; d < dim; ++d) mom[2] += w[idx] * c[d]*c[d];
    }
    ierr = PetscFree(pidx);CHKERRQ(ierr);
  }
  ierr = DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **) &w);CHKERRQ(ierr);
  ierr = DMSwarmSortRestoreAccess(sw);CHKERRQ(ierr);
  ierr = MPI_Allreduce(mom, moments, 3, MPIU_REAL, MPI_SUM, PetscObjectComm((PetscObject) sw));CHKERRQ(ierr);
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

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetObjective(prob, 0, &f0_1);CHKERRQ(ierr);
  ierr = DMPlexComputeIntegralFEM(dm, u, &moments[0], user);CHKERRQ(ierr);
  ierr = PetscDSSetObjective(prob, 0, &f0_x);CHKERRQ(ierr);
  ierr = DMPlexComputeIntegralFEM(dm, u, &moments[1], user);CHKERRQ(ierr);
  ierr = PetscDSSetObjective(prob, 0, &f0_r2);CHKERRQ(ierr);
  ierr = DMPlexComputeIntegralFEM(dm, u, &moments[2], user);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestL2Projection(DM dm, DM sw, AppCtx *user)
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
  ierr = DMGetGlobalVector(dm, &fhat);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &rhs);CHKERRQ(ierr);

  //ierr = DMSwarmCreateInterpolationMatrix(sw, dm, &M_p);CHKERRQ(ierr);
  ierr = DMCreateMassMatrix(sw, dm, &M_p);CHKERRQ(ierr);
  ierr = MatViewFromOptions(M_p, NULL, "-M_p_view");CHKERRQ(ierr);
  /* make particle weight vector */
  ierr = DMSwarmCreateGlobalVectorFromField(sw, "w_q", &f);CHKERRQ(ierr);
  /* create matrix RHS vector */
  ierr = MatMult(M_p, f, rhs);CHKERRQ(ierr);
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
  ierr = PetscPrintf(comm, "L2 projection m ([m - m_p]/m) mass: %20.13e (%11.4e), x-momentum: %20.13e (%11.4e), energy: %20.13e (%11.4e).\n", fmoments[0], (fmoments[0] - pmoments[0])/fmoments[0],
                     fmoments[1], (fmoments[1] - pmoments[1])/fmoments[1], fmoments[2], (fmoments[2] - pmoments[2])/fmoments[2]);CHKERRQ(ierr);
  for (m = 0; m < 3; ++m) {
    if (PetscAbsReal((fmoments[m] - pmoments[m])/fmoments[m]) > user->momentTol) SETERRQ3(comm, PETSC_ERR_ARG_WRONG, "Moment %D error too large %g > %g", m, PetscAbsReal((fmoments[m] - pmoments[m])/fmoments[m]), user->momentTol);
  }

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&M);CHKERRQ(ierr);
  ierr = MatDestroy(&M_p);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &fhat);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &rhs);CHKERRQ(ierr);
#if 0
  /* get FE moments */
  {
    PetscScalar momentum, energy, density, tt[0];
    PetscDS     prob;
    Vec         vecs[3];
    ierr = MatMultTranspose(QinterpT, f_q, rhs);CHKERRQ(ierr); /* interpolate particles to grid: Q * w_p */
    ierr = KSPSolve(ksp, rhs, uproj);CHKERRQ(ierr);
    ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
    ierr = PetscDSSetObjective(prob, 0, &f0_1);CHKERRQ(ierr);
    ierr = DMPlexComputeIntegralFEM(dm,uproj,tt,user);CHKERRQ(ierr);
    density = tt[0];
    ierr = PetscDSSetObjective(prob, 0, &f0_momx);CHKERRQ(ierr);
    ierr = DMPlexComputeIntegralFEM(dm,uproj,tt,user);CHKERRQ(ierr);
    momentum = tt[0];
    ierr = PetscDSSetObjective(prob, 0, &f0_energy);CHKERRQ(ierr);
    ierr = DMPlexComputeIntegralFEM(dm,uproj,tt,user);CHKERRQ(ierr);
    energy = tt[0];
    ierr = PetscPrintf(comm, "\t[%D] L2 projection x_m ([x_m-x_p]/x_m) rho: %20.13e (%11.4e), momentum_x: %20.13e (%11.4e), energy: %20.13e (%11.4e).\n",rank,density,(density-den0tot)/density,momentum,(momentum-mom0tot)/momentum,energy,(energy-energy0tot)/energy);CHKERRQ(ierr);
    /* compute coordinate vectos x and moments x' * Q * w_p */
    ierr = computeMomentVectors(dm, vecs, user);CHKERRQ(ierr);
    ierr = VecDot(vecs[0],rhs,&density);CHKERRQ(ierr);
    ierr = VecDot(vecs[1],rhs,&momentum);CHKERRQ(ierr);
    ierr = VecDot(vecs[2],rhs,&energy);CHKERRQ(ierr);
    PetscPrintf(comm, "\t[%D] x' * Q * w_p: x_m ([x_m-x_p]/x_m) rho: %20.13e (%11.4e), momentum_x: %20.13e (%11.4e), energy: %20.13e (%11.4e).\n",rank,density,(density-den0tot)/density,momentum,(momentum-mom0tot)/momentum,energy,(energy-energy0tot)/energy);
    ierr = VecDestroy(&vecs[0]);CHKERRQ(ierr);
    ierr = VecDestroy(&vecs[1]);CHKERRQ(ierr);
    ierr = VecDestroy(&vecs[2]);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

int main (int argc, char * argv[]) {
  MPI_Comm       comm;
  DM             dm, sw;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &dm, &user);CHKERRQ(ierr);
  ierr = CreateFEM(dm, &user);CHKERRQ(ierr);
  ierr = CreateParticles(dm, &sw, &user);CHKERRQ(ierr);
  ierr = TestL2Projection(dm, sw, &user);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&sw);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

  /* if (1) { /\* print array *\/ */
  /*   PetscInt    rStart, cStart, cEnd, cell; */
  /*   PetscScalar *w_p; */
  /*   const PetscReal *coords, *c2; */
  /*   ierr = DMSwarmSortGetAccess(sw);CHKERRQ(ierr); */
  /*   ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr); */
  /*   ierr = VecGetOwnershipRange(f,&rStart,NULL);CHKERRQ(ierr); */
  /*   ierr = DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr); */
  /*   ierr = VecGetArray(f,&w_p);CHKERRQ(ierr); */
  /*   for (cell = cStart, c2 = coords; cell < cEnd; ++cell) { */
  /*     PetscInt *cindices, p; */
  /*     PetscInt  numCIndices; */
  /*     ierr = DMSwarmSortGetPointsPerCell(sw, cell, &numCIndices, &cindices);CHKERRQ(ierr); */
  /*     for (p = 0; p < numCIndices; ++p, c2 += 2) { */
  /*       PetscPrintf(PETSC_COMM_SELF, "[%D]TestL2Projection: %D) real coord[%4D]:%12.5e,%12.5e, w_p[%D]=%12.5e\n",-1,c2-coords,c2-coords,c2[0],c2[1],cindices[p],w_p[cindices[p]]); */
  /*     } */
  /*     ierr = PetscFree(cindices);CHKERRQ(ierr); */
  /*   } */
  /*   ierr = VecRestoreArray(f,&w_p);CHKERRQ(ierr); */
  /*   ierr = DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr); */
  /*   ierr = DMSwarmSortRestoreAccess(sw);CHKERRQ(ierr); */
  /* } */

/*TEST

  test:
    suffix: proj_tri_0
    args: -dim 2 -faces 1 -dm_view -sw_view -petscspace_order 2 -petscfe_default_quadrature_order {{2 3}}

  test:
    suffix: proj_tri_2_faces
    args: -dim 2 -faces 2  -dm_view -sw_view -petscspace_order 2 -petscfe_default_quadrature_order {{2 3}}

  test:
    suffix: proj_quad_0
    args: -dim 2 -simplex 0 -faces 1 -dm_view -sw_view -petscspace_order 2 -petscfe_default_quadrature_order {{2 3}}

  test:
    suffix: proj_quad_2_faces
    args: -dim 2 -simplex 0 -faces 2 -dm_view -sw_view -petscspace_order 2 -petscfe_default_quadrature_order {{2 3}}

  test:
    suffix: proj_tri_5P
    args: -dim 2 -faces 1 -particlesPerCell 5 -dm_view -sw_view -petscspace_order 2 -petscfe_default_quadrature_order {{2 3}} -pc_type lu

  test:
    suffix: proj_quad_5P
    args: -dim 2 -simplex 0 -faces 1 -particlesPerCell 5 -dm_view -sw_view -petscspace_order 2 -petscfe_default_quadrature_order {{2 3}} -pc_type lu

  test:
    suffix: proj_tri_mdx
    args: -dim 2 -faces 1 -mesh_perturbation 1.0e-1 -dm_view -sw_view -petscspace_order 2 -petscfe_default_quadrature_order {{2 3}}

TEST*/
