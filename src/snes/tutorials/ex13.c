static char help[] = "Poisson Problem in 2d and 3d with finite elements.\n\
We solve the Poisson problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\
This example supports automatic convergence estimation\n\
and eventually adaptivity.\n\n\n";

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscconvest.h>

typedef struct {
  /* Domain and mesh definition */
  PetscBool spectral;    /* Look at the spectrum along planes in the solution */
  PetscBool shear;       /* Shear the domain */
  PetscBool adjoint;     /* Solve the adjoint problem */
  PetscBool homogeneous; /* Use homogeneous boudnary conditions */
  PetscBool viewError;   /* Output the solution error */
} AppCtx;

static PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  *u = 0.0;
  return 0;
}

static PetscErrorCode trig_inhomogeneous_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;
  *u = 0.0;
  for (d = 0; d < dim; ++d) *u += PetscSinReal(2.0*PETSC_PI*x[d]);
  return 0;
}

static PetscErrorCode trig_homogeneous_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;
  *u = 1.0;
  for (d = 0; d < dim; ++d) *u *= PetscSinReal(2.0*PETSC_PI*x[d]);
  return 0;
}

/* Compute integral of (residual of solution)*(adjoint solution - projection of adjoint solution) */
static void obj_error_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                        PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar obj[])
{
  obj[0] = a[aOff[0]]*(u[0] - a[aOff[1]]);
}

static void f0_trig_inhomogeneous_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f0[0] += -4.0*PetscSqr(PETSC_PI)*PetscSinReal(2.0*PETSC_PI*x[d]);
}

static void f0_trig_homogeneous_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) {
    PetscScalar v = 1.;
    for (PetscInt e = 0; e < dim; e++) {
      if (e == d) {
        v *= -4.0*PetscSqr(PETSC_PI)*PetscSinReal(2.0*PETSC_PI*x[d]);
      } else {
        v *= PetscSinReal(2.0*PETSC_PI*x[d]);
      }
    }
    f0[0] += v;
  }
}

static void f0_unity_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                       const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                       const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                       PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = 1.0;
}

static void f0_identityaux_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                             const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                             const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                             PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = a[0];
}

static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u_x[d];
}

static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d*dim+d] = 1.0;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  options->shear       = PETSC_FALSE;
  options->spectral    = PETSC_FALSE;
  options->adjoint     = PETSC_FALSE;
  options->homogeneous = PETSC_FALSE;
  options->viewError   = PETSC_FALSE;

  PetscOptionsBegin(comm, "", "Poisson Problem Options", "DMPLEX");
  PetscCall(PetscOptionsBool("-shear", "Shear the domain", "ex13.c", options->shear, &options->shear, NULL));
  PetscCall(PetscOptionsBool("-spectral", "Look at the spectrum along planes of the solution", "ex13.c", options->spectral, &options->spectral, NULL));
  PetscCall(PetscOptionsBool("-adjoint", "Solve the adjoint problem", "ex13.c", options->adjoint, &options->adjoint, NULL));
  PetscCall(PetscOptionsBool("-homogeneous", "Use homogeneous boundary conditions", "ex13.c", options->homogeneous, &options->homogeneous, NULL));
  PetscCall(PetscOptionsBool("-error_view", "Output the solution error", "ex13.c", options->viewError, &options->viewError, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateSpectralPlanes(DM dm, PetscInt numPlanes, const PetscInt planeDir[], const PetscReal planeCoord[], AppCtx *user)
{
  PetscSection       coordSection;
  Vec                coordinates;
  const PetscScalar *coords;
  PetscInt           dim, p, vStart, vEnd, v;

  PetscFunctionBeginUser;
  PetscCall(DMGetCoordinateDim(dm, &dim));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(VecGetArrayRead(coordinates, &coords));
  for (p = 0; p < numPlanes; ++p) {
    DMLabel label;
    char    name[PETSC_MAX_PATH_LEN];

    PetscCall(PetscSNPrintf(name, PETSC_MAX_PATH_LEN, "spectral_plane_%" PetscInt_FMT, p));
    PetscCall(DMCreateLabel(dm, name));
    PetscCall(DMGetLabel(dm, name, &label));
    PetscCall(DMLabelAddStratum(label, 1));
    for (v = vStart; v < vEnd; ++v) {
      PetscInt off;

      PetscCall(PetscSectionGetOffset(coordSection, v, &off));
      if (PetscAbsReal(planeCoord[p] - PetscRealPart(coords[off+planeDir[p]])) < PETSC_SMALL) {
        PetscCall(DMLabelSetValue(label, v, 1));
      }
    }
  }
  PetscCall(VecRestoreArrayRead(coordinates, &coords));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  if (user->shear) PetscCall(DMPlexShearGeometry(*dm, DM_X, NULL));
  PetscCall(DMSetApplicationContext(*dm, user));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  if (user->spectral) {
    PetscInt  planeDir[2]   = {0,  1};
    PetscReal planeCoord[2] = {0., 1.};

    PetscCall(CreateSpectralPlanes(*dm, 2, planeDir, planeCoord, user));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupPrimalProblem(DM dm, AppCtx *user)
{
  PetscDS        ds;
  DMLabel        label;
  const PetscInt id = 1;
  PetscPointFunc f0 = user->homogeneous ? f0_trig_homogeneous_u : f0_trig_inhomogeneous_u;
  PetscErrorCode (*ex)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *) = user->homogeneous ? trig_homogeneous_u : trig_inhomogeneous_u;

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetResidual(ds, 0, f0, f1_u));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu));
  PetscCall(PetscDSSetExactSolution(ds, 0, ex, user));
  PetscCall(DMGetLabel(dm, "marker", &label));
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) ex, NULL, user, NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupAdjointProblem(DM dm, AppCtx *user)
{
  PetscDS        ds;
  DMLabel        label;
  const PetscInt id = 1;

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetResidual(ds, 0, f0_unity_u, f1_u));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu));
  PetscCall(PetscDSSetObjective(ds, 0, obj_error_u));
  PetscCall(DMGetLabel(dm, "marker", &label));
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) zero, NULL, user, NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupErrorProblem(DM dm, AppCtx *user)
{
  PetscDS        prob;

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &prob));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, const char name[], PetscErrorCode (*setup)(DM, AppCtx *), AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe;
  DMPolytopeType ct;
  PetscBool      simplex;
  PetscInt       dim, cStart;
  char           prefix[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;
  /* Create finite element */
  PetscCall(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name));
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, name ? prefix : NULL, -1, &fe));
  PetscCall(PetscObjectSetName((PetscObject) fe, name));
  /* Set discretization and boundary conditions for each mesh */
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject) fe));
  PetscCall(DMCreateDS(dm));
  PetscCall((*setup)(dm, user));
  while (cdm) {
    PetscCall(DMCopyDisc(dm,cdm));
    /* TODO: Check whether the boundary of coarse meshes is marked */
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  PetscCall(PetscFEDestroy(&fe));
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeSpectral(DM dm, Vec u, PetscInt numPlanes, const PetscInt planeDir[], const PetscReal planeCoord[], AppCtx *user)
{
  MPI_Comm           comm;
  PetscSection       coordSection, section;
  Vec                coordinates, uloc;
  const PetscScalar *coords, *array;
  PetscInt           p;
  PetscMPIInt        size, rank;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject) dm, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(DMGetLocalVector(dm, &uloc));
  PetscCall(DMGlobalToLocalBegin(dm, u, INSERT_VALUES, uloc));
  PetscCall(DMGlobalToLocalEnd(dm, u, INSERT_VALUES, uloc));
  PetscCall(DMPlexInsertBoundaryValues(dm, PETSC_TRUE, uloc, 0.0, NULL, NULL, NULL));
  PetscCall(VecViewFromOptions(uloc, NULL, "-sol_view"));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(VecGetArrayRead(uloc, &array));
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(VecGetArrayRead(coordinates, &coords));
  for (p = 0; p < numPlanes; ++p) {
    DMLabel         label;
    char            name[PETSC_MAX_PATH_LEN];
    Mat             F;
    Vec             x, y;
    IS              stratum;
    PetscReal      *ray, *gray;
    PetscScalar    *rvals, *svals, *gsvals;
    PetscInt       *perm, *nperm;
    PetscInt        n, N, i, j, off, offu;
    const PetscInt *points;

    PetscCall(PetscSNPrintf(name, PETSC_MAX_PATH_LEN, "spectral_plane_%" PetscInt_FMT, p));
    PetscCall(DMGetLabel(dm, name, &label));
    PetscCall(DMLabelGetStratumIS(label, 1, &stratum));
    PetscCall(ISGetLocalSize(stratum, &n));
    PetscCall(ISGetIndices(stratum, &points));
    PetscCall(PetscMalloc2(n, &ray, n, &svals));
    for (i = 0; i < n; ++i) {
      PetscCall(PetscSectionGetOffset(coordSection, points[i], &off));
      PetscCall(PetscSectionGetOffset(section, points[i], &offu));
      ray[i]   = PetscRealPart(coords[off+((planeDir[p]+1)%2)]);
      svals[i] = array[offu];
    }
    /* Gather the ray data to proc 0 */
    if (size > 1) {
      PetscMPIInt *cnt, *displs, p;

      PetscCall(PetscCalloc2(size, &cnt, size, &displs));
      PetscCallMPI(MPI_Gather(&n, 1, MPIU_INT, cnt, 1, MPIU_INT, 0, comm));
      for (p = 1; p < size; ++p) displs[p] = displs[p-1] + cnt[p-1];
      N = displs[size-1] + cnt[size-1];
      PetscCall(PetscMalloc2(N, &gray, N, &gsvals));
      PetscCallMPI(MPI_Gatherv(ray, n, MPIU_REAL, gray, cnt, displs, MPIU_REAL, 0, comm));
      PetscCallMPI(MPI_Gatherv(svals, n, MPIU_SCALAR, gsvals, cnt, displs, MPIU_SCALAR, 0, comm));
      PetscCall(PetscFree2(cnt, displs));
    } else {
      N      = n;
      gray   = ray;
      gsvals = svals;
    }
    if (rank == 0) {
      /* Sort point along ray */
      PetscCall(PetscMalloc2(N, &perm, N, &nperm));
      for (i = 0; i < N; ++i) {perm[i] = i;}
      PetscCall(PetscSortRealWithPermutation(N, gray, perm));
      /* Count duplicates and squish mapping */
      nperm[0] = perm[0];
      for (i = 1, j = 1; i < N; ++i) {
        if (PetscAbsReal(gray[perm[i]] - gray[perm[i-1]]) > PETSC_SMALL) nperm[j++] = perm[i];
      }
      /* Create FFT structs */
      PetscCall(MatCreateFFT(PETSC_COMM_SELF, 1, &j, MATFFTW, &F));
      PetscCall(MatCreateVecs(F, &x, &y));
      PetscCall(PetscObjectSetName((PetscObject) y, name));
      PetscCall(VecGetArray(x, &rvals));
      for (i = 0, j = 0; i < N; ++i) {
        if (i > 0 && PetscAbsReal(gray[perm[i]] - gray[perm[i-1]]) < PETSC_SMALL) continue;
        rvals[j] = gsvals[nperm[j]];
        ++j;
      }
      PetscCall(PetscFree2(perm, nperm));
      if (size > 1) PetscCall(PetscFree2(gray, gsvals));
      PetscCall(VecRestoreArray(x, &rvals));
      /* Do FFT along the ray */
      PetscCall(MatMult(F, x, y));
      /* Chop FFT */
      PetscCall(VecChop(y, PETSC_SMALL));
      PetscCall(VecViewFromOptions(x, NULL, "-real_view"));
      PetscCall(VecViewFromOptions(y, NULL, "-fft_view"));
      PetscCall(VecDestroy(&x));
      PetscCall(VecDestroy(&y));
      PetscCall(MatDestroy(&F));
    }
    PetscCall(ISRestoreIndices(stratum, &points));
    PetscCall(ISDestroy(&stratum));
    PetscCall(PetscFree2(ray, svals));
  }
  PetscCall(VecRestoreArrayRead(coordinates, &coords));
  PetscCall(VecRestoreArrayRead(uloc, &array));
  PetscCall(DMRestoreLocalVector(dm, &uloc));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;   /* Problem specification */
  SNES           snes; /* Nonlinear solver */
  Vec            u;    /* Solutions */
  AppCtx         user; /* User-defined work context */

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  /* Primal system */
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(SetupDiscretization(dm, "potential", SetupPrimalProblem, &user));
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(VecSet(u, 0.0));
  PetscCall(PetscObjectSetName((PetscObject) u, "potential"));
  PetscCall(DMPlexSetSNESLocalFEM(dm, &user, &user, &user));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(SNESSolve(snes, NULL, u));
  PetscCall(SNESGetSolution(snes, &u));
  PetscCall(VecViewFromOptions(u, NULL, "-potential_view"));
  if (user.viewError) {
    PetscErrorCode (*sol)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);
    void            *ctx;
    PetscDS          ds;
    PetscReal        error;
    PetscInt         N;

    PetscCall(DMGetDS(dm, &ds));
    PetscCall(PetscDSGetExactSolution(ds, 0, &sol, &ctx));
    PetscCall(VecGetSize(u, &N));
    PetscCall(DMComputeL2Diff(dm, 0.0, &sol, &ctx, u, &error));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "N: %" PetscInt_FMT " L2 error: %g\n", N, (double)error));
  }
  if (user.spectral) {
    PetscInt  planeDir[2]   = {0,  1};
    PetscReal planeCoord[2] = {0., 1.};

    PetscCall(ComputeSpectral(dm, u, 2, planeDir, planeCoord, &user));
  }
  /* Adjoint system */
  if (user.adjoint) {
    DM   dmAdj;
    SNES snesAdj;
    Vec  uAdj;

    PetscCall(SNESCreate(PETSC_COMM_WORLD, &snesAdj));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject) snesAdj, "adjoint_"));
    PetscCall(DMClone(dm, &dmAdj));
    PetscCall(SNESSetDM(snesAdj, dmAdj));
    PetscCall(SetupDiscretization(dmAdj, "adjoint", SetupAdjointProblem, &user));
    PetscCall(DMCreateGlobalVector(dmAdj, &uAdj));
    PetscCall(VecSet(uAdj, 0.0));
    PetscCall(PetscObjectSetName((PetscObject) uAdj, "adjoint"));
    PetscCall(DMPlexSetSNESLocalFEM(dmAdj, &user, &user, &user));
    PetscCall(SNESSetFromOptions(snesAdj));
    PetscCall(SNESSolve(snesAdj, NULL, uAdj));
    PetscCall(SNESGetSolution(snesAdj, &uAdj));
    PetscCall(VecViewFromOptions(uAdj, NULL, "-adjoint_view"));
    /* Error representation */
    {
      DM        dmErr, dmErrAux, dms[2];
      Vec       errorEst, errorL2, uErr, uErrLoc, uAdjLoc, uAdjProj;
      IS       *subis;
      PetscReal errorEstTot, errorL2Norm, errorL2Tot;
      PetscInt  N, i;
      PetscErrorCode (*funcs[1])(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *) = {user.homogeneous ? trig_homogeneous_u : trig_inhomogeneous_u};
      void (*identity[1])(PetscInt, PetscInt, PetscInt,
                          const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                          const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                          PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]) = {f0_identityaux_u};
      void            *ctxs[1] = {0};

      ctxs[0] = &user;
      PetscCall(DMClone(dm, &dmErr));
      PetscCall(SetupDiscretization(dmErr, "error", SetupErrorProblem, &user));
      PetscCall(DMGetGlobalVector(dmErr, &errorEst));
      PetscCall(DMGetGlobalVector(dmErr, &errorL2));
      /*   Compute auxiliary data (solution and projection of adjoint solution) */
      PetscCall(DMGetLocalVector(dmAdj, &uAdjLoc));
      PetscCall(DMGlobalToLocalBegin(dmAdj, uAdj, INSERT_VALUES, uAdjLoc));
      PetscCall(DMGlobalToLocalEnd(dmAdj, uAdj, INSERT_VALUES, uAdjLoc));
      PetscCall(DMGetGlobalVector(dm, &uAdjProj));
      PetscCall(DMSetAuxiliaryVec(dm, NULL, 0, 0, uAdjLoc));
      PetscCall(DMProjectField(dm, 0.0, u, identity, INSERT_VALUES, uAdjProj));
      PetscCall(DMSetAuxiliaryVec(dm, NULL, 0, 0, NULL));
      PetscCall(DMRestoreLocalVector(dmAdj, &uAdjLoc));
      /*   Attach auxiliary data */
      dms[0] = dm; dms[1] = dm;
      PetscCall(DMCreateSuperDM(dms, 2, &subis, &dmErrAux));
      if (0) {
        PetscSection sec;

        PetscCall(DMGetLocalSection(dms[0], &sec));
        PetscCall(PetscSectionView(sec, PETSC_VIEWER_STDOUT_WORLD));
        PetscCall(DMGetLocalSection(dms[1], &sec));
        PetscCall(PetscSectionView(sec, PETSC_VIEWER_STDOUT_WORLD));
        PetscCall(DMGetLocalSection(dmErrAux, &sec));
        PetscCall(PetscSectionView(sec, PETSC_VIEWER_STDOUT_WORLD));
      }
      PetscCall(DMViewFromOptions(dmErrAux, NULL, "-dm_err_view"));
      PetscCall(ISViewFromOptions(subis[0], NULL, "-super_is_view"));
      PetscCall(ISViewFromOptions(subis[1], NULL, "-super_is_view"));
      PetscCall(DMGetGlobalVector(dmErrAux, &uErr));
      PetscCall(VecViewFromOptions(u, NULL, "-map_vec_view"));
      PetscCall(VecViewFromOptions(uAdjProj, NULL, "-map_vec_view"));
      PetscCall(VecViewFromOptions(uErr, NULL, "-map_vec_view"));
      PetscCall(VecISCopy(uErr, subis[0], SCATTER_FORWARD, u));
      PetscCall(VecISCopy(uErr, subis[1], SCATTER_FORWARD, uAdjProj));
      PetscCall(DMRestoreGlobalVector(dm, &uAdjProj));
      for (i = 0; i < 2; ++i) PetscCall(ISDestroy(&subis[i]));
      PetscCall(PetscFree(subis));
      PetscCall(DMGetLocalVector(dmErrAux, &uErrLoc));
      PetscCall(DMGlobalToLocalBegin(dm, uErr, INSERT_VALUES, uErrLoc));
      PetscCall(DMGlobalToLocalEnd(dm, uErr, INSERT_VALUES, uErrLoc));
      PetscCall(DMRestoreGlobalVector(dmErrAux, &uErr));
      PetscCall(DMSetAuxiliaryVec(dmAdj, NULL, 0, 0, uErrLoc));
      /*   Compute cellwise error estimate */
      PetscCall(VecSet(errorEst, 0.0));
      PetscCall(DMPlexComputeCellwiseIntegralFEM(dmAdj, uAdj, errorEst, &user));
      PetscCall(DMSetAuxiliaryVec(dmAdj, NULL, 0, 0, NULL));
      PetscCall(DMRestoreLocalVector(dmErrAux, &uErrLoc));
      PetscCall(DMDestroy(&dmErrAux));
      /*   Plot cellwise error vector */
      PetscCall(VecViewFromOptions(errorEst, NULL, "-error_view"));
      /*   Compute ratio of estimate (sum over cells) with actual L_2 error */
      PetscCall(DMComputeL2Diff(dm, 0.0, funcs, ctxs, u, &errorL2Norm));
      PetscCall(DMPlexComputeL2DiffVec(dm, 0.0, funcs, ctxs, u, errorL2));
      PetscCall(VecViewFromOptions(errorL2, NULL, "-l2_error_view"));
      PetscCall(VecNorm(errorL2,  NORM_INFINITY, &errorL2Tot));
      PetscCall(VecNorm(errorEst, NORM_INFINITY, &errorEstTot));
      PetscCall(VecGetSize(errorEst, &N));
      PetscCall(VecPointwiseDivide(errorEst, errorEst, errorL2));
      PetscCall(PetscObjectSetName((PetscObject) errorEst, "Error ratio"));
      PetscCall(VecViewFromOptions(errorEst, NULL, "-error_ratio_view"));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "N: %" PetscInt_FMT " L2 error: %g Error Ratio: %g/%g = %g\n", N, (double) errorL2Norm, (double) errorEstTot, (double) PetscSqrtReal(errorL2Tot), (double)(errorEstTot/PetscSqrtReal(errorL2Tot))));
      PetscCall(DMRestoreGlobalVector(dmErr, &errorEst));
      PetscCall(DMRestoreGlobalVector(dmErr, &errorL2));
      PetscCall(DMDestroy(&dmErr));
    }
    PetscCall(DMDestroy(&dmAdj));
    PetscCall(VecDestroy(&uAdj));
    PetscCall(SNESDestroy(&snesAdj));
  }
  /* Cleanup */
  PetscCall(VecDestroy(&u));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    # Using -dm_refine 2 -convest_num_refine 3 we get L_2 convergence rate: 1.9
    suffix: 2d_p1_conv
    requires: triangle
    args: -potential_petscspace_degree 1 -snes_convergence_estimate -convest_num_refine 2
  test:
    # Using -dm_refine 2 -convest_num_refine 3 we get L_2 convergence rate: 2.9
    suffix: 2d_p2_conv
    requires: triangle
    args: -potential_petscspace_degree 2 -snes_convergence_estimate -convest_num_refine 2
  test:
    # Using -dm_refine 2 -convest_num_refine 3 we get L_2 convergence rate: 3.9
    suffix: 2d_p3_conv
    requires: triangle
    args: -potential_petscspace_degree 3 -snes_convergence_estimate -convest_num_refine 2
  test:
    # Using -dm_refine 2 -convest_num_refine 3 we get L_2 convergence rate: 1.9
    suffix: 2d_q1_conv
    args: -dm_plex_simplex 0 -potential_petscspace_degree 1 -snes_convergence_estimate -convest_num_refine 2
  test:
    # Using -dm_refine 2 -convest_num_refine 3 we get L_2 convergence rate: 2.9
    suffix: 2d_q2_conv
    args: -dm_plex_simplex 0 -potential_petscspace_degree 2 -snes_convergence_estimate -convest_num_refine 2
  test:
    # Using -dm_refine 2 -convest_num_refine 3 we get L_2 convergence rate: 3.9
    suffix: 2d_q3_conv
    args: -dm_plex_simplex 0 -potential_petscspace_degree 3 -snes_convergence_estimate -convest_num_refine 2
  test:
    # Using -dm_refine 2 -convest_num_refine 3 we get L_2 convergence rate: 1.9
    suffix: 2d_q1_shear_conv
    args: -dm_plex_simplex 0 -shear -potential_petscspace_degree 1 -snes_convergence_estimate -convest_num_refine 2
  test:
    # Using -dm_refine 2 -convest_num_refine 3 we get L_2 convergence rate: 2.9
    suffix: 2d_q2_shear_conv
    args: -dm_plex_simplex 0 -shear -potential_petscspace_degree 2 -snes_convergence_estimate -convest_num_refine 2
  test:
    # Using -dm_refine 2 -convest_num_refine 3 we get L_2 convergence rate: 3.9
    suffix: 2d_q3_shear_conv
    args: -dm_plex_simplex 0 -shear -potential_petscspace_degree 3 -snes_convergence_estimate -convest_num_refine 2
  test:
    # Using -convest_num_refine 3 we get L_2 convergence rate: 1.7
    suffix: 3d_p1_conv
    requires: ctetgen
    args: -dm_plex_dim 3 -dm_refine 1 -potential_petscspace_degree 1 -snes_convergence_estimate -convest_num_refine 1
  test:
    # Using -dm_refine 1 -convest_num_refine 3 we get L_2 convergence rate: 2.8
    suffix: 3d_p2_conv
    requires: ctetgen
    args: -dm_plex_dim 3 -dm_plex_box_faces 2,2,2 -potential_petscspace_degree 2 -snes_convergence_estimate -convest_num_refine 1
  test:
    # Using -dm_refine 1 -convest_num_refine 3 we get L_2 convergence rate: 4.0
    suffix: 3d_p3_conv
    requires: ctetgen
    args: -dm_plex_dim 3 -dm_plex_box_faces 2,2,2 -potential_petscspace_degree 3 -snes_convergence_estimate -convest_num_refine 1
  test:
    # Using -dm_refine 2 -convest_num_refine 3 we get L_2 convergence rate: 1.8
    suffix: 3d_q1_conv
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_refine 1 -potential_petscspace_degree 1 -snes_convergence_estimate -convest_num_refine 1
  test:
    # Using -dm_refine 2 -convest_num_refine 3 we get L_2 convergence rate: 2.8
    suffix: 3d_q2_conv
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -potential_petscspace_degree 2 -snes_convergence_estimate -convest_num_refine 1
  test:
    # Using -dm_refine 1 -convest_num_refine 3 we get L_2 convergence rate: 3.8
    suffix: 3d_q3_conv
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -potential_petscspace_degree 3 -snes_convergence_estimate -convest_num_refine 1
  test:
    suffix: 2d_p1_fas_full
    requires: triangle
    args: -potential_petscspace_degree 1 -dm_refine_hierarchy 5 \
      -snes_max_it 1 -snes_type fas -snes_fas_levels 5 -snes_fas_type full -snes_fas_full_total \
        -fas_coarse_snes_monitor -fas_coarse_snes_max_it 1 -fas_coarse_ksp_atol 1.e-13 \
        -fas_levels_snes_monitor -fas_levels_snes_max_it 1 -fas_levels_snes_type newtonls \
          -fas_levels_pc_type none -fas_levels_ksp_max_it 2 -fas_levels_ksp_converged_maxits -fas_levels_ksp_type chebyshev \
            -fas_levels_esteig_ksp_type cg -fas_levels_ksp_chebyshev_esteig 0,0.25,0,1.1 -fas_levels_esteig_ksp_max_it 10
  test:
    suffix: 2d_p1_fas_full_homogeneous
    requires: triangle
    args: -homogeneous -potential_petscspace_degree 1 -dm_refine_hierarchy 5 \
      -snes_max_it 1 -snes_type fas -snes_fas_levels 5 -snes_fas_type full \
        -fas_coarse_snes_monitor -fas_coarse_snes_max_it 1 -fas_coarse_ksp_atol 1.e-13 \
        -fas_levels_snes_monitor -fas_levels_snes_max_it 1 -fas_levels_snes_type newtonls \
          -fas_levels_pc_type none -fas_levels_ksp_max_it 2 -fas_levels_ksp_converged_maxits -fas_levels_ksp_type chebyshev \
            -fas_levels_esteig_ksp_type cg -fas_levels_ksp_chebyshev_esteig 0,0.25,0,1.1 -fas_levels_esteig_ksp_max_it 10

  test:
    suffix: 2d_p1_scalable
    requires: triangle
    args: -potential_petscspace_degree 1 -dm_refine 3 \
      -ksp_type cg -ksp_rtol 1.e-11 -ksp_norm_type unpreconditioned \
      -pc_type gamg -pc_gamg_esteig_ksp_type cg -pc_gamg_esteig_ksp_max_it 10 \
        -pc_gamg_type agg -pc_gamg_agg_nsmooths 1 \
        -pc_gamg_coarse_eq_limit 1000 \
        -pc_gamg_threshold 0.05 \
        -pc_gamg_threshold_scale .0 \
        -mg_levels_ksp_type chebyshev \
        -mg_levels_ksp_max_it 1 \
        -mg_levels_pc_type jacobi \
      -matptap_via scalable
  test:
    suffix: 2d_p1_gmg_vcycle
    requires: triangle
    args: -potential_petscspace_degree 1 -dm_plex_box_faces 2,2 -dm_refine_hierarchy 3 \
          -ksp_rtol 5e-10 -pc_type mg \
            -mg_levels_ksp_max_it 1 \
            -mg_levels_esteig_ksp_type cg \
            -mg_levels_esteig_ksp_max_it 10 \
            -mg_levels_ksp_chebyshev_esteig 0,0.1,0,1.1 \
            -mg_levels_pc_type jacobi
  test:
    suffix: 2d_p1_gmg_fcycle
    requires: triangle
    args: -potential_petscspace_degree 1 -dm_plex_box_faces 2,2 -dm_refine_hierarchy 3 \
          -ksp_rtol 5e-10 -pc_type mg -pc_mg_type full \
            -mg_levels_ksp_max_it 2 \
            -mg_levels_esteig_ksp_type cg \
            -mg_levels_esteig_ksp_max_it 10 \
            -mg_levels_ksp_chebyshev_esteig 0,0.1,0,1.1 \
            -mg_levels_pc_type jacobi
  test:
    suffix: 2d_p1_gmg_vcycle_adapt
    requires: triangle
    args: -petscpartitioner_type simple -potential_petscspace_degree 1 -dm_plex_box_faces 2,2 -dm_refine_hierarchy 3 \
          -ksp_rtol 5e-10 -pc_type mg -pc_mg_galerkin -pc_mg_adapt_interp_coarse_space harmonic -pc_mg_adapt_interp_n 8 \
            -mg_levels_ksp_max_it 1 \
            -mg_levels_esteig_ksp_type cg \
            -mg_levels_esteig_ksp_max_it 10 \
            -mg_levels_ksp_chebyshev_esteig 0,0.1,0,1.1 \
            -mg_levels_pc_type jacobi
  test:
    suffix: 2d_p1_spectral_0
    requires: triangle fftw !complex
    args: -dm_plex_box_faces 1,1 -potential_petscspace_degree 1 -dm_refine 6 -spectral -fft_view
  test:
    suffix: 2d_p1_spectral_1
    requires: triangle fftw !complex
    nsize: 2
    args: -dm_plex_box_faces 4,4 -potential_petscspace_degree 1 -spectral -fft_view
  test:
    suffix: 2d_p1_adj_0
    requires: triangle
    args: -potential_petscspace_degree 1 -dm_refine 1 -adjoint -adjoint_petscspace_degree 1 -error_petscspace_degree 0
  test:
    nsize: 2
    requires: !sycl kokkos_kernels
    suffix: kokkos
    args: -dm_plex_dim 3 -dm_plex_box_faces 2,3,6 -petscpartitioner_type simple -dm_plex_simplex 0 -potential_petscspace_degree 1 \
         -dm_refine 0 -ksp_type cg -ksp_rtol 1.e-11 -ksp_norm_type unpreconditioned -pc_type gamg -pc_gamg_coarse_eq_limit 1000 -pc_gamg_threshold 0.0 \
         -pc_gamg_threshold_scale .5 -mg_levels_ksp_type chebyshev -mg_levels_ksp_max_it 2 -pc_gamg_esteig_ksp_type cg -pc_gamg_esteig_ksp_max_it 10 \
         -ksp_monitor -snes_monitor -dm_view -dm_mat_type aijkokkos -dm_vec_type kokkos

TEST*/
