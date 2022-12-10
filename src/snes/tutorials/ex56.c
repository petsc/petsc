/* Portions of this code are under:
   Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
*/
static char help[] = "3D tensor hexahedra & 3D Laplacian displacement finite element formulation\n\
of linear elasticity.  E=1.0, nu=1/3.\n\
Unit cube domain with Dirichlet boundary\n\n";

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscdmforest.h>

static PetscReal s_soft_alpha = 0.01;
static PetscReal s_mu         = 0.4;
static PetscReal s_lambda     = 0.4;

static void f0_bd_u_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = 1;     /* x direction pull */
  f0[1] = -x[2]; /* add a twist around x-axis */
  f0[2] = x[1];
}

static void f1_bd_u(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscInt Ncomp = dim;
  PetscInt       comp, d;
  for (comp = 0; comp < Ncomp; ++comp) {
    for (d = 0; d < dim; ++d) f1[comp * dim + d] = 0.0;
  }
}

/* gradU[comp*dim+d] = {u_x, u_y} or {u_x, u_y, u_z} */
static void f1_u_3d_alpha(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscReal trace, mu = s_mu, lambda = s_lambda, rad;
  PetscInt  i, j;
  for (i = 0, rad = 0.; i < dim; i++) {
    PetscReal t = x[i];
    rad += t * t;
  }
  rad = PetscSqrtReal(rad);
  if (rad > 0.25) {
    mu *= s_soft_alpha;
    lambda *= s_soft_alpha; /* we could keep the bulk the same like rubberish */
  }
  for (i = 0, trace = 0; i < dim; ++i) trace += PetscRealPart(u_x[i * dim + i]);
  for (i = 0; i < dim; ++i) {
    for (j = 0; j < dim; ++j) f1[i * dim + j] = mu * (u_x[i * dim + j] + u_x[j * dim + i]);
    f1[i * dim + i] += lambda * trace;
  }
}

/* gradU[comp*dim+d] = {u_x, u_y} or {u_x, u_y, u_z} */
static void f1_u_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscReal trace, mu = s_mu, lambda = s_lambda;
  PetscInt  i, j;
  for (i = 0, trace = 0; i < dim; ++i) trace += PetscRealPart(u_x[i * dim + i]);
  for (i = 0; i < dim; ++i) {
    for (j = 0; j < dim; ++j) f1[i * dim + j] = mu * (u_x[i * dim + j] + u_x[j * dim + i]);
    f1[i * dim + i] += lambda * trace;
  }
}

static void f1_u_lap(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u_x[d];
}

/* 3D elasticity */
#define IDX(ii, jj, kk, ll) (27 * ii + 9 * jj + 3 * kk + ll)

void g3_uu_3d_private(PetscScalar g3[], const PetscReal mu, const PetscReal lambda)
{
  if (1) {
    g3[0] += lambda;
    g3[0] += mu;
    g3[0] += mu;
    g3[4] += lambda;
    g3[8] += lambda;
    g3[10] += mu;
    g3[12] += mu;
    g3[20] += mu;
    g3[24] += mu;
    g3[28] += mu;
    g3[30] += mu;
    g3[36] += lambda;
    g3[40] += lambda;
    g3[40] += mu;
    g3[40] += mu;
    g3[44] += lambda;
    g3[50] += mu;
    g3[52] += mu;
    g3[56] += mu;
    g3[60] += mu;
    g3[68] += mu;
    g3[70] += mu;
    g3[72] += lambda;
    g3[76] += lambda;
    g3[80] += lambda;
    g3[80] += mu;
    g3[80] += mu;
  } else {
    int        i, j, k, l;
    static int cc = -1;
    cc++;
    for (i = 0; i < 3; ++i) {
      for (j = 0; j < 3; ++j) {
        for (k = 0; k < 3; ++k) {
          for (l = 0; l < 3; ++l) {
            if (k == l && i == j) g3[IDX(i, j, k, l)] += lambda;
            if (i == k && j == l) g3[IDX(i, j, k, l)] += mu;
            if (i == l && j == k) g3[IDX(i, j, k, l)] += mu;
            if (k == l && i == j && !cc) (void)PetscPrintf(PETSC_COMM_WORLD, "g3[%d] += lambda;\n", IDX(i, j, k, l));
            if (i == k && j == l && !cc) (void)PetscPrintf(PETSC_COMM_WORLD, "g3[%d] += mu;\n", IDX(i, j, k, l));
            if (i == l && j == k && !cc) (void)PetscPrintf(PETSC_COMM_WORLD, "g3[%d] += mu;\n", IDX(i, j, k, l));
          }
        }
      }
    }
  }
}

static void g3_uu_3d_alpha(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscReal mu = s_mu, lambda = s_lambda, rad;
  PetscInt  i;
  for (i = 0, rad = 0.; i < dim; i++) {
    PetscReal t = x[i];
    rad += t * t;
  }
  rad = PetscSqrtReal(rad);
  if (rad > 0.25) {
    mu *= s_soft_alpha;
    lambda *= s_soft_alpha; /* we could keep the bulk the same like rubberish */
  }
  g3_uu_3d_private(g3, mu, lambda);
}

static void g3_uu_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  g3_uu_3d_private(g3, s_mu, s_lambda);
}

static void g3_lap(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d * dim + d] = 1.0;
}

static void g3_lap_alpha(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscReal lambda = 1, rad;
  PetscInt  i;
  for (i = 0, rad = 0.; i < dim; i++) {
    PetscReal t = x[i];
    rad += t * t;
  }
  rad = PetscSqrtReal(rad);
  if (rad > 0.25) { lambda *= s_soft_alpha; /* we could keep the bulk the same like rubberish */ }
  for (int d = 0; d < dim; ++d) g3[d * dim + d] = lambda;
}

static void f0_u(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscInt Ncomp = dim;
  PetscInt       comp;

  for (comp = 0; comp < Ncomp; ++comp) f0[comp] = 0.0;
}

/* PI_i (x_i^4 - x_i^2) */
static void f0_u_x4(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  for (int comp = 0; comp < Nf; ++comp) {
    f0[comp] = 1e5;
    for (int i = 0; i < dim; ++i) { f0[comp] *= /* (comp+1)* */ (x[i] * x[i] * x[i] * x[i] - x[i] * x[i]); /* assumes (0,1]^D domain */ }
  }
}

PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  const PetscInt Ncomp = dim;
  PetscInt       comp;

  for (comp = 0; comp < Ncomp; ++comp) u[comp] = 0;
  return PETSC_SUCCESS;
}

int main(int argc, char **args)
{
  Mat         Amat;
  SNES        snes;
  KSP         ksp;
  MPI_Comm    comm;
  PetscMPIInt rank;
#if defined(PETSC_USE_LOG)
  PetscLogStage stage[17];
#endif
  PetscBool test_nonzero_cols = PETSC_FALSE, use_nearnullspace = PETSC_TRUE, attach_nearnullspace = PETSC_FALSE;
  Vec       xx, bb;
  PetscInt  iter, i, N, dim = 3, max_conv_its, sizes[7], run_type = 1, Ncomp = dim;
  DM        dm;
  PetscBool flg;
  PetscReal Lx, mdisp[10], err[10];
  PetscFunctionBeginUser;
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  /* options */
  PetscOptionsBegin(comm, NULL, "3D bilinear Q1 elasticity options", "");
  {
    Lx           = 1.; /* or ne for rod */
    max_conv_its = 3;
    PetscCall(PetscOptionsInt("-max_conv_its", "Number of iterations in convergence study", "", max_conv_its, &max_conv_its, NULL));
    PetscCheck(max_conv_its > 0 && max_conv_its < 8, PETSC_COMM_WORLD, PETSC_ERR_USER, "Bad number of iterations for convergence test (%" PetscInt_FMT ")", max_conv_its);
    PetscCall(PetscOptionsReal("-lx", "Length of domain", "", Lx, &Lx, NULL));
    PetscCall(PetscOptionsReal("-alpha", "material coefficient inside circle", "", s_soft_alpha, &s_soft_alpha, NULL));
    PetscCall(PetscOptionsBool("-test_nonzero_cols", "nonzero test", "", test_nonzero_cols, &test_nonzero_cols, NULL));
    PetscCall(PetscOptionsBool("-use_mat_nearnullspace", "MatNearNullSpace API test", "", use_nearnullspace, &use_nearnullspace, NULL));
    PetscCall(PetscOptionsBool("-attach_mat_nearnullspace", "MatNearNullSpace API test (via MatSetNearNullSpace)", "", attach_nearnullspace, &attach_nearnullspace, NULL));
    PetscCall(PetscOptionsInt("-run_type", "0: twisting load on cantalever, 1: Elasticty convergence test on cube, 2: Laplacian, 3: soft core Laplacian", "", run_type, &run_type, NULL));
  }
  PetscOptionsEnd();
  PetscCall(PetscLogStageRegister("Mesh Setup", &stage[16]));
  for (iter = 0; iter < max_conv_its; iter++) {
    char str[] = "Solve 0";
    str[6] += iter;
    PetscCall(PetscLogStageRegister(str, &stage[iter]));
  }
  /* create DM, Plex calls DMSetup */
  PetscCall(PetscLogStagePush(stage[16]));
  PetscCall(DMCreate(comm, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(PetscObjectSetName((PetscObject)dm, "Mesh"));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMPlexDistributeSetDefault(dm, PETSC_FALSE));
  PetscCall(DMGetDimension(dm, &dim));
  {
    DMLabel label;
    IS      is;
    PetscCall(DMCreateLabel(dm, "boundary"));
    PetscCall(DMGetLabel(dm, "boundary", &label));
    PetscCall(DMPlexMarkBoundaryFaces(dm, 1, label));
    if (run_type == 0) {
      PetscCall(DMGetStratumIS(dm, "boundary", 1, &is));
      PetscCall(DMCreateLabel(dm, "Faces"));
      if (is) {
        PetscInt        d, f, Nf;
        const PetscInt *faces;
        PetscInt        csize;
        PetscSection    cs;
        Vec             coordinates;
        DM              cdm;
        PetscCall(ISGetLocalSize(is, &Nf));
        PetscCall(ISGetIndices(is, &faces));
        PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
        PetscCall(DMGetCoordinateDM(dm, &cdm));
        PetscCall(DMGetLocalSection(cdm, &cs));
        /* Check for each boundary face if any component of its centroid is either 0.0 or 1.0 */
        for (f = 0; f < Nf; ++f) {
          PetscReal    faceCoord;
          PetscInt     b, v;
          PetscScalar *coords = NULL;
          PetscInt     Nv;
          PetscCall(DMPlexVecGetClosure(cdm, cs, coordinates, faces[f], &csize, &coords));
          Nv = csize / dim; /* Calculate mean coordinate vector */
          for (d = 0; d < dim; ++d) {
            faceCoord = 0.0;
            for (v = 0; v < Nv; ++v) faceCoord += PetscRealPart(coords[v * dim + d]);
            faceCoord /= Nv;
            for (b = 0; b < 2; ++b) {
              if (PetscAbs(faceCoord - b) < PETSC_SMALL) { /* domain have not been set yet, still [0,1]^3 */
                PetscCall(DMSetLabelValue(dm, "Faces", faces[f], d * 2 + b + 1));
              }
            }
          }
          PetscCall(DMPlexVecRestoreClosure(cdm, cs, coordinates, faces[f], &csize, &coords));
        }
        PetscCall(ISRestoreIndices(is, &faces));
      }
      PetscCall(ISDestroy(&is));
      PetscCall(DMGetLabel(dm, "Faces", &label));
      PetscCall(DMPlexLabelComplete(dm, label));
    }
  }
  PetscCall(PetscLogStagePop());
  for (iter = 0; iter < max_conv_its; iter++) {
    PetscCall(PetscLogStagePush(stage[16]));
    /* snes */
    PetscCall(SNESCreate(comm, &snes));
    PetscCall(SNESSetDM(snes, dm));
    PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
    /* fem */
    {
      const PetscInt components[] = {0, 1, 2};
      const PetscInt Nfid = 1, Npid = 1;
      PetscInt       fid[] = {1}; /* The fixed faces (x=0) */
      const PetscInt pid[] = {2}; /* The faces with loading (x=L_x) */
      PetscFE        fe;
      PetscDS        prob;
      DMLabel        label;

      if (run_type == 2 || run_type == 3) Ncomp = 1;
      else Ncomp = dim;
      PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, Ncomp, PETSC_FALSE, NULL, PETSC_DECIDE, &fe));
      PetscCall(PetscObjectSetName((PetscObject)fe, "deformation"));
      /* FEM prob */
      PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
      PetscCall(DMCreateDS(dm));
      PetscCall(DMGetDS(dm, &prob));
      /* setup problem */
      if (run_type == 1) { // elast
        PetscCall(PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu_3d));
        PetscCall(PetscDSSetResidual(prob, 0, f0_u_x4, f1_u_3d));
      } else if (run_type == 0) { //twisted not maintained
        PetscWeakForm wf;
        PetscInt      bd, i;
        PetscCall(PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu_3d_alpha));
        PetscCall(PetscDSSetResidual(prob, 0, f0_u, f1_u_3d_alpha));
        PetscCall(DMGetLabel(dm, "Faces", &label));
        PetscCall(DMAddBoundary(dm, DM_BC_NATURAL, "traction", label, Npid, pid, 0, Ncomp, components, NULL, NULL, NULL, &bd));
        PetscCall(PetscDSGetBoundary(prob, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
        for (i = 0; i < Npid; ++i) PetscCall(PetscWeakFormSetIndexBdResidual(wf, label, pid[i], 0, 0, 0, f0_bd_u_3d, 0, f1_bd_u));
      } else if (run_type == 2) { // Laplacian
        PetscCall(PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_lap));
        PetscCall(PetscDSSetResidual(prob, 0, f0_u_x4, f1_u_lap));
      } else if (run_type == 3) { // soft core Laplacian
        PetscCall(PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_lap_alpha));
        PetscCall(PetscDSSetResidual(prob, 0, f0_u_x4, f1_u_lap));
      }
      /* bcs */
      if (run_type != 0) {
        PetscInt id = 1;
        PetscCall(DMGetLabel(dm, "boundary", &label));
        PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void))zero, NULL, NULL, NULL));
      } else {
        PetscCall(DMGetLabel(dm, "Faces", &label));
        PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "fixed", label, Nfid, fid, 0, Ncomp, components, (void (*)(void))zero, NULL, NULL, NULL));
      }
      PetscCall(PetscFEDestroy(&fe));
    }
    /* vecs & mat */
    PetscCall(DMCreateGlobalVector(dm, &xx));
    PetscCall(VecDuplicate(xx, &bb));
    PetscCall(PetscObjectSetName((PetscObject)bb, "b"));
    PetscCall(PetscObjectSetName((PetscObject)xx, "u"));
    PetscCall(DMCreateMatrix(dm, &Amat));
    PetscCall(MatSetOption(Amat, MAT_SYMMETRIC, PETSC_TRUE));        /* Some matrix kernels can take advantage of symmetry if we set this. */
    PetscCall(MatSetOption(Amat, MAT_SYMMETRY_ETERNAL, PETSC_TRUE)); /* Inform PETSc that Amat is always symmetric, so info set above isn't lost. */
    PetscCall(MatSetBlockSize(Amat, Ncomp));
    PetscCall(MatSetOption(Amat, MAT_SPD, PETSC_TRUE));
    PetscCall(MatSetOption(Amat, MAT_SPD_ETERNAL, PETSC_TRUE));
    PetscCall(VecGetSize(bb, &N));
    sizes[iter] = N;
    PetscCall(PetscInfo(snes, "%" PetscInt_FMT " global equations, %" PetscInt_FMT " vertices\n", N, N / dim));
    if ((use_nearnullspace || attach_nearnullspace) && N / dim > 1 && Ncomp > 1) {
      /* Set up the near null space (a.k.a. rigid body modes) that will be used by the multigrid preconditioner */
      DM           subdm;
      MatNullSpace nearNullSpace;
      PetscInt     fields = 0;
      PetscObject  deformation;
      PetscCall(DMCreateSubDM(dm, 1, &fields, NULL, &subdm));
      PetscCall(DMPlexCreateRigidBody(subdm, 0, &nearNullSpace));
      PetscCall(DMGetField(dm, 0, NULL, &deformation));
      PetscCall(PetscObjectCompose(deformation, "nearnullspace", (PetscObject)nearNullSpace));
      PetscCall(DMDestroy(&subdm));
      if (attach_nearnullspace) PetscCall(MatSetNearNullSpace(Amat, nearNullSpace));
      PetscCall(MatNullSpaceDestroy(&nearNullSpace)); /* created by DM and destroyed by Mat */
    }
    PetscCall(DMPlexSetSNESLocalFEM(dm, NULL, NULL, NULL));
    PetscCall(SNESSetJacobian(snes, Amat, Amat, NULL, NULL));
    PetscCall(SNESSetFromOptions(snes));
    PetscCall(DMSetUp(dm));
    PetscCall(PetscLogStagePop());
    PetscCall(PetscLogStagePush(stage[16]));
    /* ksp */
    PetscCall(SNESGetKSP(snes, &ksp));
    PetscCall(KSPSetComputeSingularValues(ksp, PETSC_TRUE));
    /* test BCs */
    PetscCall(VecZeroEntries(xx));
    if (test_nonzero_cols) {
      if (rank == 0) PetscCall(VecSetValue(xx, 0, 1.0, INSERT_VALUES));
      PetscCall(VecAssemblyBegin(xx));
      PetscCall(VecAssemblyEnd(xx));
    }
    PetscCall(VecZeroEntries(bb));
    PetscCall(VecGetSize(bb, &i));
    sizes[iter] = i;
    PetscCall(PetscInfo(snes, "%" PetscInt_FMT " equations in vector, %" PetscInt_FMT " vertices\n", i, i / dim));
    PetscCall(PetscLogStagePop());
    /* solve */
    PetscCall(SNESComputeJacobian(snes, xx, Amat, Amat));
    PetscCall(MatViewFromOptions(Amat, NULL, "-my_mat_view"));
    PetscCall(PetscLogStagePush(stage[iter]));
    PetscCall(SNESSolve(snes, bb, xx));
    PetscCall(PetscLogStagePop());
    PetscCall(VecNorm(xx, NORM_INFINITY, &mdisp[iter]));
    {
      PetscViewer       viewer = NULL;
      PetscViewerFormat fmt;
      PetscCall(PetscOptionsGetViewer(comm, NULL, "", "-vec_view", &viewer, &fmt, &flg));
      if (flg) {
        PetscCall(PetscViewerPushFormat(viewer, fmt));
        PetscCall(VecView(xx, viewer));
        PetscCall(VecView(bb, viewer));
        PetscCall(PetscViewerPopFormat(viewer));
      }
      PetscCall(PetscViewerDestroy(&viewer));
    }
    /* Free work space */
    PetscCall(SNESDestroy(&snes));
    PetscCall(VecDestroy(&xx));
    PetscCall(VecDestroy(&bb));
    PetscCall(MatDestroy(&Amat));
    if (iter + 1 < max_conv_its) {
      DM newdm;
      PetscCall(DMViewFromOptions(dm, NULL, "-my_dm_view"));
      PetscCall(DMRefine(dm, comm, &newdm));
      if (rank == -1) {
        PetscDS prob;
        PetscCall(DMGetDS(dm, &prob));
        PetscCall(PetscDSViewFromOptions(prob, NULL, "-ds_view"));
        PetscCall(DMGetDS(newdm, &prob));
        PetscCall(PetscDSViewFromOptions(prob, NULL, "-ds_view"));
      }
      PetscCall(DMDestroy(&dm));
      dm = newdm;
      PetscCall(PetscObjectSetName((PetscObject)dm, "Mesh"));
      PetscCall(DMViewFromOptions(dm, NULL, "-my_dm_view"));
      PetscCall(DMSetFromOptions(dm));
    }
  }
  PetscCall(DMDestroy(&dm));
  if (run_type == 1) err[0] = 5.97537599375e+01 - mdisp[0]; /* error with what I think is the exact solution */
  else if (run_type == 0) err[0] = 0;
  else if (run_type == 2) err[0] = 3.527795e+01 - mdisp[0];
  else err[0] = 0;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[%d] %d) N=%12" PetscInt_FMT ", max displ=%9.7e, error=%4.3e\n", rank, 0, sizes[0], (double)mdisp[0], (double)err[0]));
  for (iter = 1; iter < max_conv_its; iter++) {
    if (run_type == 1) err[iter] = 5.97537599375e+01 - mdisp[iter];
    else if (run_type == 0) err[iter] = 0;
    else if (run_type == 2) err[iter] = 3.527795e+01 - mdisp[iter];
    else err[iter] = 0;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[%d] %" PetscInt_FMT ") N=%12" PetscInt_FMT ", max displ=%9.7e, disp diff=%9.2e, error=%4.3e, rate=%3.2g\n", rank, iter, sizes[iter], (double)mdisp[iter], (double)(mdisp[iter] - mdisp[iter - 1]), (double)err[iter], (double)(PetscLogReal(PetscAbs(err[iter - 1] / err[iter])) / PetscLogReal(2.))));
  }

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    nsize: 4
    requires: !single
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 1,1,1 -dm_plex_box_faces 2,2,1 -petscpartitioner_simple_process_grid 2,2,1 -max_conv_its 2 -petscspace_degree 2 -snes_max_it 1 -ksp_max_it 100 -ksp_type cg -ksp_rtol 1.e-10 -ksp_norm_type unpreconditioned -pc_type gamg -pc_gamg_coarse_eq_limit 10 -pc_gamg_reuse_interpolation true -pc_gamg_aggressive_coarsening 1 -pc_gamg_threshold 0.001 -ksp_converged_reason  -use_mat_nearnullspace true -mg_levels_ksp_max_it 2 -mg_levels_ksp_type chebyshev -mg_levels_ksp_chebyshev_esteig 0,0.2,0,1.1 -mg_levels_pc_type jacobi -petscpartitioner_type simple -my_dm_view -snes_lag_jacobian -2 -snes_type ksponly
    timeoutfactor: 2
    test:
      suffix: 0
      args: -run_type 1
    test:
      suffix: 1
      args: -run_type 2

  # HYPRE PtAP broken with complex numbers
  test:
    suffix: hypre
    requires: hypre !single !complex !defined(PETSC_HAVE_HYPRE_DEVICE)
    nsize: 4
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 1,1,1 -run_type 1 -dm_plex_box_faces 2,2,1 -petscpartitioner_simple_process_grid 2,2,1 -max_conv_its 2 -lx 1. -alpha .01 -petscspace_degree 2 -ksp_type cg -ksp_monitor_short -ksp_rtol 1.e-8 -pc_type hypre -pc_hypre_type boomeramg -pc_hypre_boomeramg_no_CF true -pc_hypre_boomeramg_agg_nl 1 -pc_hypre_boomeramg_coarsen_type HMIS -pc_hypre_boomeramg_interp_type ext+i -ksp_converged_reason -use_mat_nearnullspace true -petscpartitioner_type simple

  test:
    suffix: ml
    requires: ml !single
    nsize: 4
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 1,1,1 -run_type 1 -dm_plex_box_faces 2,2,1 -petscpartitioner_simple_process_grid 2,2,1 -max_conv_its 2 -lx 1. -alpha .01 -petscspace_degree 2 -ksp_type cg -ksp_monitor_short -ksp_converged_reason -ksp_rtol 1.e-8 -pc_type ml -mg_levels_ksp_type chebyshev -mg_levels_ksp_max_it 3 -mg_levels_ksp_chebyshev_esteig 0,0.05,0,1.05 -mg_levels_pc_type sor -petscpartitioner_type simple -use_mat_nearnullspace

  test:
    suffix: hpddm
    requires: hpddm slepc !single defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
    nsize: 4
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 1,1,1 -run_type 1 -dm_plex_box_faces 2,2,1 -petscpartitioner_simple_process_grid 2,2,1 -max_conv_its 2 -lx 1. -alpha .01 -petscspace_degree 2 -ksp_type fgmres -ksp_monitor_short -ksp_converged_reason -ksp_rtol 1.e-8 -pc_type hpddm -petscpartitioner_type simple -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_levels_1_eps_nev 6 -pc_hpddm_coarse_p 1 -pc_hpddm_coarse_pc_type svd

  test:
    suffix: repart
    nsize: 4
    requires: parmetis !single
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 1,1,1 -run_type 1 -dm_plex_box_faces 2,2,1 -petscpartitioner_simple_process_grid 2,2,1 -max_conv_its 2 -petscspace_degree 2 -snes_max_it 4 -ksp_max_it 100 -ksp_type cg -ksp_rtol 1.e-2 -ksp_norm_type unpreconditioned -snes_rtol 1.e-3 -pc_type gamg -pc_gamg_esteig_ksp_max_it 10 -pc_gamg_type agg -pc_gamg_agg_nsmooths 1 -pc_gamg_aggressive_coarsening 1 -pc_gamg_threshold 0.05 -pc_gamg_threshold_scale .0 -use_mat_nearnullspace true -mg_levels_ksp_max_it 2 -mg_levels_ksp_type chebyshev -mg_levels_ksp_chebyshev_esteig 0,0.05,0,1.05 -mg_levels_pc_type jacobi -pc_gamg_mat_partitioning_type parmetis -pc_gamg_repartition true  -pc_gamg_process_eq_limit 20 -pc_gamg_coarse_eq_limit 10 -ksp_converged_reason  -pc_gamg_reuse_interpolation true -petscpartitioner_type simple

  test:
    suffix: bddc
    nsize: 4
    requires: !single
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 1,1,1 -run_type 1 -dm_plex_box_faces 2,2,1 -petscpartitioner_simple_process_grid 2,2,1 -max_conv_its 2 -lx 1. -alpha .01 -petscspace_degree 2 -ksp_type cg -ksp_monitor_short -ksp_rtol 1.e-8 -ksp_converged_reason -petscpartitioner_type simple -dm_mat_type is -matis_localmat_type {{sbaij baij aij}} -pc_type bddc

  testset:
    nsize: 4
    requires: !single
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 1,1,1 -run_type 1 -dm_plex_box_faces 2,2,1 -petscpartitioner_simple_process_grid 2,2,1 -max_conv_its 2 -lx 1. -alpha .01 -petscspace_degree 2 -ksp_type cg -ksp_monitor_short -ksp_rtol 1.e-10 -ksp_converged_reason -petscpartitioner_type simple -dm_mat_type is -matis_localmat_type aij -pc_type bddc -attach_mat_nearnullspace {{0 1}separate output}
    test:
      suffix: bddc_approx_gamg
      args: -pc_bddc_switch_static -prefix_push pc_bddc_dirichlet_ -approximate -pc_type gamg -pc_gamg_esteig_ksp_max_it 10 -pc_gamg_type agg -pc_gamg_agg_nsmooths 1 -pc_gamg_reuse_interpolation true -pc_gamg_aggressive_coarsening 1 -pc_gamg_threshold 0.05 -pc_gamg_threshold_scale .0 -mg_levels_ksp_max_it 1 -mg_levels_ksp_type chebyshev -prefix_pop -prefix_push pc_bddc_neumann_ -approximate -pc_type gamg -pc_gamg_esteig_ksp_max_it 10 -pc_gamg_type agg -pc_gamg_agg_nsmooths 1 -pc_gamg_coarse_eq_limit 10 -pc_gamg_reuse_interpolation true -pc_gamg_aggressive_coarsening 1 -pc_gamg_threshold 0.05 -pc_gamg_threshold_scale .0 -mg_levels_ksp_max_it 1 -mg_levels_ksp_type chebyshev -prefix_pop
    # HYPRE PtAP broken with complex numbers
    test:
      requires: hypre !complex !defined(PETSC_HAVE_HYPRE_DEVICE)
      suffix: bddc_approx_hypre
      args: -pc_bddc_switch_static -prefix_push pc_bddc_dirichlet_ -pc_type hypre -pc_hypre_boomeramg_no_CF true -pc_hypre_boomeramg_strong_threshold 0.75 -pc_hypre_boomeramg_agg_nl 1 -pc_hypre_boomeramg_coarsen_type HMIS -pc_hypre_boomeramg_interp_type ext+i -prefix_pop -prefix_push pc_bddc_neumann_ -pc_type hypre -pc_hypre_boomeramg_no_CF true -pc_hypre_boomeramg_strong_threshold 0.75 -pc_hypre_boomeramg_agg_nl 1 -pc_hypre_boomeramg_coarsen_type HMIS -pc_hypre_boomeramg_interp_type ext+i -prefix_pop
    test:
      requires: ml
      suffix: bddc_approx_ml
      args: -pc_bddc_switch_static -prefix_push pc_bddc_dirichlet_ -approximate -pc_type ml -mg_levels_ksp_max_it 1 -mg_levels_ksp_type chebyshev -prefix_pop -prefix_push pc_bddc_neumann_ -approximate -pc_type ml -mg_levels_ksp_max_it 1 -mg_levels_ksp_type chebyshev -prefix_pop

  test:
    suffix: fetidp
    nsize: 4
    requires: !single
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 1,1,1 -run_type 1 -dm_plex_box_faces 2,2,1 -petscpartitioner_simple_process_grid 2,2,1 -max_conv_its 2 -lx 1. -alpha .01 -petscspace_degree 2 -ksp_type fetidp -fetidp_ksp_type cg -ksp_monitor_short -ksp_rtol 1.e-8 -ksp_converged_reason -petscpartitioner_type simple -dm_mat_type is -matis_localmat_type {{sbaij baij aij}}

  test:
    suffix: bddc_elast
    nsize: 4
    requires: !single
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 1,1,1 -run_type 1 -dm_plex_box_faces 2,2,1 -petscpartitioner_simple_process_grid 2,2,1 -max_conv_its 2 -lx 1. -alpha .01 -petscspace_degree 2 -ksp_type cg -ksp_monitor_short -ksp_rtol 1.e-8 -ksp_converged_reason -petscpartitioner_type simple -dm_mat_type is -matis_localmat_type sbaij -pc_type bddc -pc_bddc_monolithic -attach_mat_nearnullspace

  test:
    suffix: fetidp_elast
    nsize: 4
    requires: !single
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 1,1,1 -run_type 1 -dm_plex_box_faces 2,2,1 -petscpartitioner_simple_process_grid 2,2,1 -max_conv_its 2 -lx 1. -alpha .01 -petscspace_degree 2 -ksp_type fetidp -fetidp_ksp_type cg -ksp_monitor_short -ksp_rtol 1.e-8 -ksp_converged_reason -petscpartitioner_type simple -dm_mat_type is -matis_localmat_type sbaij -fetidp_bddc_pc_bddc_monolithic -attach_mat_nearnullspace

  test:
    suffix: gdsw
    nsize: 4
    requires: !single
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 1,1,1 -run_type 1 -dm_plex_box_faces 2,2,1 -petscpartitioner_simple_process_grid 2,2,1 -max_conv_its 2 -lx 1. -alpha .01 -petscspace_degree 2 -ksp_type cg -ksp_monitor_short -ksp_rtol 1.e-8 -ksp_converged_reason -petscpartitioner_type simple -dm_mat_type is -attach_mat_nearnullspace \
          -pc_type mg -pc_mg_galerkin -pc_mg_adapt_interp_coarse_space gdsw -pc_mg_levels 2 -mg_levels_pc_type bjacobi -mg_levels_sub_pc_type icc

  testset:
    nsize: 4
    requires: !single
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 1,1,1 -run_type 1 -dm_plex_box_faces 2,2,1 -petscpartitioner_simple_process_grid 2,2,1 -max_conv_its 2 -petscspace_degree 2 -snes_max_it 2 -ksp_max_it 100 -ksp_type cg -ksp_rtol 1.e-10 -ksp_norm_type unpreconditioned -snes_rtol 1.e-10 -pc_type gamg -pc_gamg_esteig_ksp_max_it 10 -pc_gamg_type agg -pc_gamg_agg_nsmooths 1 -pc_gamg_coarse_eq_limit 10 -pc_gamg_reuse_interpolation true -pc_gamg_aggressive_coarsening 1 -pc_gamg_threshold 0.05 -pc_gamg_threshold_scale .0 -use_mat_nearnullspace true -mg_levels_ksp_max_it 2 -mg_levels_ksp_type chebyshev -mg_levels_ksp_chebyshev_esteig 0,0.05,0,1.05 -mg_levels_pc_type jacobi -ksp_monitor_short -ksp_converged_reason  -snes_monitor_short -dm_view -petscpartitioner_type simple -pc_gamg_process_eq_limit 20
    output_file: output/ex56_cuda.out

    test:
      suffix: cuda
      requires: cuda
      args: -dm_mat_type aijcusparse -dm_vec_type cuda

    test:
      suffix: hip
      requires: hip
      args: -dm_mat_type aijhipsparse -dm_vec_type hip

    test:
      suffix: viennacl
      requires: viennacl
      args: -dm_mat_type aijviennacl -dm_vec_type viennacl

    test:
      suffix: kokkos
      requires: kokkos_kernels
      args: -dm_mat_type aijkokkos -dm_vec_type kokkos
  # Don't run AIJMKL caes with complex scalars because of convergence issues.
  # Note that we need to test both single and multiple MPI rank cases, because these use different sparse MKL routines to implement the PtAP operation.
  test:
    suffix: seqaijmkl
    nsize: 1
    requires: defined(PETSC_HAVE_MKL_SPARSE_OPTIMIZE) !single !complex
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 1,1,1 -run_type 1 -dm_plex_box_faces 2,2,1 -petscpartitioner_simple_process_grid 2,2,1 -max_conv_its 2 -petscspace_degree 2 -snes_max_it 2 -ksp_max_it 100 -ksp_type cg -ksp_rtol 1.e-11 -ksp_norm_type unpreconditioned -snes_rtol 1.e-10 -pc_type gamg -pc_gamg_type agg -pc_gamg_agg_nsmooths 1 -pc_gamg_coarse_eq_limit 1000 -pc_gamg_reuse_interpolation true -pc_gamg_aggressive_coarsening 1 -pc_gamg_threshold 0.05 -pc_gamg_threshold_scale .0 -ksp_converged_reason -snes_monitor_short -ksp_monitor_short -use_mat_nearnullspace true -mg_levels_ksp_max_it 1 -mg_levels_ksp_type chebyshev -pc_gamg_esteig_ksp_type cg -pc_gamg_esteig_ksp_max_it 10 -mg_levels_ksp_chebyshev_esteig 0,0.05,0,1.1 -mg_levels_pc_type jacobi -petscpartitioner_type simple -mat_block_size 3 -dm_view -mat_seqaij_type seqaijmkl
    timeoutfactor: 2

  test:
    suffix: mpiaijmkl
    nsize: 4
    requires: defined(PETSC_HAVE_MKL_SPARSE_OPTIMIZE) !single !complex
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 1,1,1 -run_type 1 -dm_plex_box_faces 2,2,1 -petscpartitioner_simple_process_grid 2,2,1 -max_conv_its 2 -petscspace_degree 2 -snes_max_it 2 -ksp_max_it 100 -ksp_type cg -ksp_rtol 1.e-11 -ksp_norm_type unpreconditioned -snes_rtol 1.e-10 -pc_type gamg -pc_gamg_type agg -pc_gamg_agg_nsmooths 1 -pc_gamg_coarse_eq_limit 1000 -pc_gamg_reuse_interpolation true -pc_gamg_aggressive_coarsening 1 -pc_gamg_threshold 0.05 -pc_gamg_threshold_scale .0 -ksp_converged_reason -snes_monitor_short -ksp_monitor_short -use_mat_nearnullspace true -mg_levels_ksp_max_it 1 -mg_levels_ksp_type chebyshev -pc_gamg_esteig_ksp_type cg -pc_gamg_esteig_ksp_max_it 10 -mg_levels_ksp_chebyshev_esteig 0,0.05,0,1.1 -mg_levels_pc_type jacobi -petscpartitioner_type simple -mat_block_size 3 -dm_view -mat_seqaij_type seqaijmkl
    timeoutfactor: 2

TEST*/
