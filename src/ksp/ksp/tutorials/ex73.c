
/*
This example was derived from src/ksp/ksp/tutorials ex29.c

Inhomogeneous Laplacian in 2D. Modeled by the partial differential equation

   -div \rho grad u = f,  0 < x,y < 1,

with forcing function

   f = e^{-x^2/\nu} e^{-y^2/\nu}

with Dirichlet boundary conditions

   u = f(x,y) for x = 0, x = 1, y = 0, y = 1

or pure Neumman boundary conditions.
*/

static char help[] = "Solves 2D inhomogeneous Laplacian. Demonstrates using PCTelescopeSetCoarseDM functionality of PCTelescope via a DMShell\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmshell.h>
#include <petscksp.h>

PetscErrorCode ComputeMatrix_ShellDA(KSP, Mat, Mat, void *);
PetscErrorCode ComputeMatrix_DMDA(DM, Mat, Mat, void *);
PetscErrorCode ComputeRHS_DMDA(DM, Vec, void *);
PetscErrorCode DMShellCreate_ShellDA(DM, DM *);
PetscErrorCode DMFieldScatter_ShellDA(DM, Vec, ScatterMode, DM, Vec);
PetscErrorCode DMStateScatter_ShellDA(DM, ScatterMode, DM);

typedef enum {
  DIRICHLET,
  NEUMANN
} BCType;

typedef struct {
  PetscReal rho;
  PetscReal nu;
  BCType    bcType;
  MPI_Comm  comm;
} UserContext;

PetscErrorCode UserContextCreate(MPI_Comm comm, UserContext **ctx)
{
  UserContext *user;
  const char  *bcTypes[2] = {"dirichlet", "neumann"};
  PetscInt     bc;

  PetscFunctionBeginUser;
  PetscCall(PetscCalloc1(1, &user));
  user->comm = comm;
  PetscOptionsBegin(comm, "", "Options for the inhomogeneous Poisson equation", "DMqq");
  user->rho = 1.0;
  PetscCall(PetscOptionsReal("-rho", "The conductivity", "ex29.c", user->rho, &user->rho, NULL));
  user->nu = 0.1;
  PetscCall(PetscOptionsReal("-nu", "The width of the Gaussian source", "ex29.c", user->nu, &user->nu, NULL));
  bc = (PetscInt)DIRICHLET;
  PetscCall(PetscOptionsEList("-bc_type", "Type of boundary condition", "ex29.c", bcTypes, 2, bcTypes[0], &bc, NULL));
  user->bcType = (BCType)bc;
  PetscOptionsEnd();
  *ctx = user;
  PetscFunctionReturn(0);
}

PetscErrorCode CommCoarsen(MPI_Comm comm, PetscInt number, PetscSubcomm *p)
{
  PetscSubcomm psubcomm;
  PetscFunctionBeginUser;
  PetscCall(PetscSubcommCreate(comm, &psubcomm));
  PetscCall(PetscSubcommSetNumber(psubcomm, number));
  PetscCall(PetscSubcommSetType(psubcomm, PETSC_SUBCOMM_INTERLACED));
  *p = psubcomm;
  PetscFunctionReturn(0);
}

PetscErrorCode CommHierarchyCreate(MPI_Comm comm, PetscInt n, PetscInt number[], PetscSubcomm pscommlist[])
{
  PetscInt  k;
  PetscBool view_hierarchy = PETSC_FALSE;

  PetscFunctionBeginUser;
  for (k = 0; k < n; k++) pscommlist[k] = NULL;

  if (n < 1) PetscFunctionReturn(0);

  PetscCall(CommCoarsen(comm, number[n - 1], &pscommlist[n - 1]));
  for (k = n - 2; k >= 0; k--) {
    MPI_Comm comm_k = PetscSubcommChild(pscommlist[k + 1]);
    if (pscommlist[k + 1]->color == 0) PetscCall(CommCoarsen(comm_k, number[k], &pscommlist[k]));
  }

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-view_hierarchy", &view_hierarchy, NULL));
  if (view_hierarchy) {
    PetscMPIInt size;

    PetscCallMPI(MPI_Comm_size(comm, &size));
    PetscCall(PetscPrintf(comm, "level[%" PetscInt_FMT "] size %d\n", n, (int)size));
    for (k = n - 1; k >= 0; k--) {
      if (pscommlist[k]) {
        MPI_Comm comm_k = PetscSubcommChild(pscommlist[k]);

        if (pscommlist[k]->color == 0) {
          PetscCallMPI(MPI_Comm_size(comm_k, &size));
          PetscCall(PetscPrintf(comm_k, "level[%" PetscInt_FMT "] size %d\n", k, (int)size));
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

/* taken from src/ksp/pc/impls/telescope/telescope_dmda.c */
static PetscErrorCode _DMDADetermineRankFromGlobalIJ_2d(PetscInt i, PetscInt j, PetscInt Mp, PetscInt Np, PetscInt start_i[], PetscInt start_j[], PetscInt span_i[], PetscInt span_j[], PetscMPIInt *_pi, PetscMPIInt *_pj, PetscMPIInt *rank_re)
{
  PetscInt pi, pj, n;

  PetscFunctionBeginUser;
  *rank_re = -1;
  pi = pj = -1;
  if (_pi) {
    for (n = 0; n < Mp; n++) {
      if ((i >= start_i[n]) && (i < start_i[n] + span_i[n])) {
        pi = n;
        break;
      }
    }
    PetscCheck(pi != -1, PETSC_COMM_SELF, PETSC_ERR_USER, "[dmda-ij] pi cannot be determined : range %" PetscInt_FMT ", val %" PetscInt_FMT, Mp, i);
    *_pi = (PetscMPIInt)pi;
  }

  if (_pj) {
    for (n = 0; n < Np; n++) {
      if ((j >= start_j[n]) && (j < start_j[n] + span_j[n])) {
        pj = n;
        break;
      }
    }
    PetscCheck(pj != -1, PETSC_COMM_SELF, PETSC_ERR_USER, "[dmda-ij] pj cannot be determined : range %" PetscInt_FMT ", val %" PetscInt_FMT, Np, j);
    *_pj = (PetscMPIInt)pj;
  }

  *rank_re = (PetscMPIInt)(pi + pj * Mp);
  PetscFunctionReturn(0);
}

/* taken from src/ksp/pc/impls/telescope/telescope_dmda.c */
static PetscErrorCode _DMDADetermineGlobalS0_2d(PetscMPIInt rank_re, PetscInt Mp_re, PetscInt Np_re, PetscInt range_i_re[], PetscInt range_j_re[], PetscInt *s0)
{
  PetscInt    i, j, start_IJ = 0;
  PetscMPIInt rank_ij;

  PetscFunctionBeginUser;
  *s0 = -1;
  for (j = 0; j < Np_re; j++) {
    for (i = 0; i < Mp_re; i++) {
      rank_ij = (PetscMPIInt)(i + j * Mp_re);
      if (rank_ij < rank_re) start_IJ += range_i_re[i] * range_j_re[j];
    }
  }
  *s0 = start_IJ;
  PetscFunctionReturn(0);
}

/* adapted from src/ksp/pc/impls/telescope/telescope_dmda.c */
static PetscErrorCode DMDACreatePermutation_2d(DM dmrepart, DM dmf, Mat *mat)
{
  PetscInt        k, sum, Mp_re = 0, Np_re = 0;
  PetscInt        nx, ny, sr, er, Mr, ndof;
  PetscInt        i, j, location, startI[2], endI[2], lenI[2];
  const PetscInt *_range_i_re = NULL, *_range_j_re = NULL;
  PetscInt       *range_i_re, *range_j_re;
  PetscInt       *start_i_re, *start_j_re;
  MPI_Comm        comm;
  Vec             V;
  Mat             Pscalar;

  PetscFunctionBeginUser;
  PetscCall(PetscInfo(dmf, "setting up the permutation matrix (DMDA-2D)\n"));
  PetscCall(PetscObjectGetComm((PetscObject)dmf, &comm));

  _range_i_re = _range_j_re = NULL;
  /* Create DMDA on the child communicator */
  if (dmrepart) {
    PetscCall(DMDAGetInfo(dmrepart, NULL, NULL, NULL, NULL, &Mp_re, &Np_re, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(DMDAGetOwnershipRanges(dmrepart, &_range_i_re, &_range_j_re, NULL));
  }

  /* note - assume rank 0 always participates */
  PetscCallMPI(MPI_Bcast(&Mp_re, 1, MPIU_INT, 0, comm));
  PetscCallMPI(MPI_Bcast(&Np_re, 1, MPIU_INT, 0, comm));

  PetscCall(PetscCalloc1(Mp_re, &range_i_re));
  PetscCall(PetscCalloc1(Np_re, &range_j_re));

  if (_range_i_re) PetscCall(PetscArraycpy(range_i_re, _range_i_re, Mp_re));
  if (_range_j_re) PetscCall(PetscArraycpy(range_j_re, _range_j_re, Np_re));

  PetscCallMPI(MPI_Bcast(range_i_re, Mp_re, MPIU_INT, 0, comm));
  PetscCallMPI(MPI_Bcast(range_j_re, Np_re, MPIU_INT, 0, comm));

  PetscCall(PetscMalloc1(Mp_re, &start_i_re));
  PetscCall(PetscMalloc1(Np_re, &start_j_re));

  sum = 0;
  for (k = 0; k < Mp_re; k++) {
    start_i_re[k] = sum;
    sum += range_i_re[k];
  }

  sum = 0;
  for (k = 0; k < Np_re; k++) {
    start_j_re[k] = sum;
    sum += range_j_re[k];
  }

  /* Create permutation */
  PetscCall(DMDAGetInfo(dmf, NULL, &nx, &ny, NULL, NULL, NULL, NULL, &ndof, NULL, NULL, NULL, NULL, NULL));
  PetscCall(DMGetGlobalVector(dmf, &V));
  PetscCall(VecGetSize(V, &Mr));
  PetscCall(VecGetOwnershipRange(V, &sr, &er));
  PetscCall(DMRestoreGlobalVector(dmf, &V));
  sr = sr / ndof;
  er = er / ndof;
  Mr = Mr / ndof;

  PetscCall(MatCreate(comm, &Pscalar));
  PetscCall(MatSetSizes(Pscalar, (er - sr), (er - sr), Mr, Mr));
  PetscCall(MatSetType(Pscalar, MATAIJ));
  PetscCall(MatSeqAIJSetPreallocation(Pscalar, 1, NULL));
  PetscCall(MatMPIAIJSetPreallocation(Pscalar, 1, NULL, 1, NULL));

  PetscCall(DMDAGetCorners(dmf, NULL, NULL, NULL, &lenI[0], &lenI[1], NULL));
  PetscCall(DMDAGetCorners(dmf, &startI[0], &startI[1], NULL, &endI[0], &endI[1], NULL));
  endI[0] += startI[0];
  endI[1] += startI[1];

  for (j = startI[1]; j < endI[1]; j++) {
    for (i = startI[0]; i < endI[0]; i++) {
      PetscMPIInt rank_ijk_re, rank_reI[] = {0, 0};
      PetscInt    s0_re;
      PetscInt    ii, jj, local_ijk_re, mapped_ijk;
      PetscInt    lenI_re[] = {0, 0};

      location = (i - startI[0]) + (j - startI[1]) * lenI[0];
      PetscCall(_DMDADetermineRankFromGlobalIJ_2d(i, j, Mp_re, Np_re, start_i_re, start_j_re, range_i_re, range_j_re, &rank_reI[0], &rank_reI[1], &rank_ijk_re));

      PetscCall(_DMDADetermineGlobalS0_2d(rank_ijk_re, Mp_re, Np_re, range_i_re, range_j_re, &s0_re));

      ii = i - start_i_re[rank_reI[0]];
      PetscCheck(ii >= 0, PETSC_COMM_SELF, PETSC_ERR_USER, "[dmdarepart-perm2d] index error ii");
      jj = j - start_j_re[rank_reI[1]];
      PetscCheck(jj >= 0, PETSC_COMM_SELF, PETSC_ERR_USER, "[dmdarepart-perm2d] index error jj");

      lenI_re[0]   = range_i_re[rank_reI[0]];
      lenI_re[1]   = range_j_re[rank_reI[1]];
      local_ijk_re = ii + jj * lenI_re[0];
      mapped_ijk   = s0_re + local_ijk_re;
      PetscCall(MatSetValue(Pscalar, sr + location, mapped_ijk, 1.0, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(Pscalar, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Pscalar, MAT_FINAL_ASSEMBLY));

  *mat = Pscalar;

  PetscCall(PetscFree(range_i_re));
  PetscCall(PetscFree(range_j_re));
  PetscCall(PetscFree(start_i_re));
  PetscCall(PetscFree(start_j_re));
  PetscFunctionReturn(0);
}

/* adapted from src/ksp/pc/impls/telescope/telescope_dmda.c */
static PetscErrorCode PCTelescopeSetUp_dmda_scatters(DM dmf, DM dmc)
{
  Vec        xred, yred, xtmp, x, xp;
  VecScatter scatter;
  IS         isin;
  PetscInt   m, bs, st, ed;
  MPI_Comm   comm;
  VecType    vectype;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)dmf, &comm));
  PetscCall(DMCreateGlobalVector(dmf, &x));
  PetscCall(VecGetBlockSize(x, &bs));
  PetscCall(VecGetType(x, &vectype));

  /* cannot use VecDuplicate as xp is already composed with dmf */
  /*PetscCall(VecDuplicate(x,&xp));*/
  {
    PetscInt m, M;

    PetscCall(VecGetSize(x, &M));
    PetscCall(VecGetLocalSize(x, &m));
    PetscCall(VecCreate(comm, &xp));
    PetscCall(VecSetSizes(xp, m, M));
    PetscCall(VecSetBlockSize(xp, bs));
    PetscCall(VecSetType(xp, vectype));
  }

  m    = 0;
  xred = NULL;
  yred = NULL;
  if (dmc) {
    PetscCall(DMCreateGlobalVector(dmc, &xred));
    PetscCall(VecDuplicate(xred, &yred));
    PetscCall(VecGetOwnershipRange(xred, &st, &ed));
    PetscCall(ISCreateStride(comm, ed - st, st, 1, &isin));
    PetscCall(VecGetLocalSize(xred, &m));
  } else {
    PetscCall(VecGetOwnershipRange(x, &st, &ed));
    PetscCall(ISCreateStride(comm, 0, st, 1, &isin));
  }
  PetscCall(ISSetBlockSize(isin, bs));
  PetscCall(VecCreate(comm, &xtmp));
  PetscCall(VecSetSizes(xtmp, m, PETSC_DECIDE));
  PetscCall(VecSetBlockSize(xtmp, bs));
  PetscCall(VecSetType(xtmp, vectype));
  PetscCall(VecScatterCreate(x, isin, xtmp, NULL, &scatter));

  PetscCall(PetscObjectCompose((PetscObject)dmf, "isin", (PetscObject)isin));
  PetscCall(PetscObjectCompose((PetscObject)dmf, "scatter", (PetscObject)scatter));
  PetscCall(PetscObjectCompose((PetscObject)dmf, "xtmp", (PetscObject)xtmp));
  PetscCall(PetscObjectCompose((PetscObject)dmf, "xp", (PetscObject)xp));

  PetscCall(VecDestroy(&xred));
  PetscCall(VecDestroy(&yred));
  PetscCall(VecDestroy(&x));
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateMatrix_ShellDA(DM dm, Mat *A)
{
  DM           da;
  MPI_Comm     comm;
  PetscMPIInt  size;
  UserContext *ctx = NULL;
  PetscInt     M, N;

  PetscFunctionBeginUser;
  PetscCall(DMShellGetContext(dm, &da));
  PetscCall(PetscObjectGetComm((PetscObject)da, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(DMCreateMatrix(da, A));
  PetscCall(MatGetSize(*A, &M, &N));
  PetscCall(PetscPrintf(comm, "[size %" PetscInt_FMT "] DMCreateMatrix_ShellDA (%" PetscInt_FMT " x %" PetscInt_FMT ")\n", (PetscInt)size, M, N));

  PetscCall(DMGetApplicationContext(dm, &ctx));
  if (ctx->bcType == NEUMANN) {
    MatNullSpace nullspace = NULL;
    PetscCall(PetscPrintf(comm, "[size %" PetscInt_FMT "] DMCreateMatrix_ShellDA: using neumann bcs\n", (PetscInt)size));

    PetscCall(MatGetNullSpace(*A, &nullspace));
    if (!nullspace) {
      PetscCall(PetscPrintf(comm, "[size %" PetscInt_FMT "] DMCreateMatrix_ShellDA: operator does not have nullspace - attaching\n", (PetscInt)size));
      PetscCall(MatNullSpaceCreate(comm, PETSC_TRUE, 0, 0, &nullspace));
      PetscCall(MatSetNullSpace(*A, nullspace));
      PetscCall(MatNullSpaceDestroy(&nullspace));
    } else {
      PetscCall(PetscPrintf(comm, "[size %" PetscInt_FMT "] DMCreateMatrix_ShellDA: operator already has a nullspace\n", (PetscInt)size));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateGlobalVector_ShellDA(DM dm, Vec *x)
{
  DM da;
  PetscFunctionBeginUser;
  PetscCall(DMShellGetContext(dm, &da));
  PetscCall(DMCreateGlobalVector(da, x));
  PetscCall(VecSetDM(*x, dm));
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateLocalVector_ShellDA(DM dm, Vec *x)
{
  DM da;
  PetscFunctionBeginUser;
  PetscCall(DMShellGetContext(dm, &da));
  PetscCall(DMCreateLocalVector(da, x));
  PetscCall(VecSetDM(*x, dm));
  PetscFunctionReturn(0);
}

PetscErrorCode DMCoarsen_ShellDA(DM dm, MPI_Comm comm, DM *dmc)
{
  PetscFunctionBeginUser;
  *dmc = NULL;
  PetscCall(DMGetCoarseDM(dm, dmc));
  if (!*dmc) {
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "The coarse DM should never be NULL. The DM hierarchy should have already been defined");
  } else {
    PetscCall(PetscObjectReference((PetscObject)(*dmc)));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateInterpolation_ShellDA(DM dm1, DM dm2, Mat *mat, Vec *vec)
{
  DM da1, da2;
  PetscFunctionBeginUser;
  PetscCall(DMShellGetContext(dm1, &da1));
  PetscCall(DMShellGetContext(dm2, &da2));
  PetscCall(DMCreateInterpolation(da1, da2, mat, vec));
  PetscFunctionReturn(0);
}

PetscErrorCode DMShellDASetUp_TelescopeDMScatter(DM dmf_shell, DM dmc_shell)
{
  Mat P   = NULL;
  DM  dmf = NULL, dmc = NULL;

  PetscFunctionBeginUser;
  PetscCall(DMShellGetContext(dmf_shell, &dmf));
  if (dmc_shell) PetscCall(DMShellGetContext(dmc_shell, &dmc));
  PetscCall(DMDACreatePermutation_2d(dmc, dmf, &P));
  PetscCall(PetscObjectCompose((PetscObject)dmf, "P", (PetscObject)P));
  PetscCall(PCTelescopeSetUp_dmda_scatters(dmf, dmc));
  PetscCall(PetscObjectComposeFunction((PetscObject)dmf_shell, "PCTelescopeFieldScatter", DMFieldScatter_ShellDA));
  PetscCall(PetscObjectComposeFunction((PetscObject)dmf_shell, "PCTelescopeStateScatter", DMStateScatter_ShellDA));
  PetscFunctionReturn(0);
}

PetscErrorCode DMShellDAFieldScatter_Forward(DM dmf, Vec x, DM dmc, Vec xc)
{
  Mat                P  = NULL;
  Vec                xp = NULL, xtmp = NULL;
  VecScatter         scatter = NULL;
  const PetscScalar *x_array;
  PetscInt           i, st, ed;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectQuery((PetscObject)dmf, "P", (PetscObject *)&P));
  PetscCall(PetscObjectQuery((PetscObject)dmf, "xp", (PetscObject *)&xp));
  PetscCall(PetscObjectQuery((PetscObject)dmf, "scatter", (PetscObject *)&scatter));
  PetscCall(PetscObjectQuery((PetscObject)dmf, "xtmp", (PetscObject *)&xtmp));
  PetscCheck(P, PETSC_COMM_SELF, PETSC_ERR_USER, "Require a permutation matrix (\"P\")to be composed with the parent (fine) DM");
  PetscCheck(xp, PETSC_COMM_SELF, PETSC_ERR_USER, "Require \"xp\" to be composed with the parent (fine) DM");
  PetscCheck(scatter, PETSC_COMM_SELF, PETSC_ERR_USER, "Require \"scatter\" to be composed with the parent (fine) DM");
  PetscCheck(xtmp, PETSC_COMM_SELF, PETSC_ERR_USER, "Require \"xtmp\" to be composed with the parent (fine) DM");

  PetscCall(MatMultTranspose(P, x, xp));

  /* pull in vector x->xtmp */
  PetscCall(VecScatterBegin(scatter, xp, xtmp, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter, xp, xtmp, INSERT_VALUES, SCATTER_FORWARD));

  /* copy vector entries into xred */
  PetscCall(VecGetArrayRead(xtmp, &x_array));
  if (xc) {
    PetscScalar *LA_xred;
    PetscCall(VecGetOwnershipRange(xc, &st, &ed));

    PetscCall(VecGetArray(xc, &LA_xred));
    for (i = 0; i < ed - st; i++) LA_xred[i] = x_array[i];
    PetscCall(VecRestoreArray(xc, &LA_xred));
  }
  PetscCall(VecRestoreArrayRead(xtmp, &x_array));
  PetscFunctionReturn(0);
}

PetscErrorCode DMShellDAFieldScatter_Reverse(DM dmf, Vec y, DM dmc, Vec yc)
{
  Mat          P  = NULL;
  Vec          xp = NULL, xtmp = NULL;
  VecScatter   scatter = NULL;
  PetscScalar *array;
  PetscInt     i, st, ed;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectQuery((PetscObject)dmf, "P", (PetscObject *)&P));
  PetscCall(PetscObjectQuery((PetscObject)dmf, "xp", (PetscObject *)&xp));
  PetscCall(PetscObjectQuery((PetscObject)dmf, "scatter", (PetscObject *)&scatter));
  PetscCall(PetscObjectQuery((PetscObject)dmf, "xtmp", (PetscObject *)&xtmp));

  PetscCheck(P, PETSC_COMM_SELF, PETSC_ERR_USER, "Require a permutation matrix (\"P\")to be composed with the parent (fine) DM");
  PetscCheck(xp, PETSC_COMM_SELF, PETSC_ERR_USER, "Require \"xp\" to be composed with the parent (fine) DM");
  PetscCheck(scatter, PETSC_COMM_SELF, PETSC_ERR_USER, "Require \"scatter\" to be composed with the parent (fine) DM");
  PetscCheck(xtmp, PETSC_COMM_SELF, PETSC_ERR_USER, "Require \"xtmp\" to be composed with the parent (fine) DM");

  /* return vector */
  PetscCall(VecGetArray(xtmp, &array));
  if (yc) {
    const PetscScalar *LA_yred;
    PetscCall(VecGetOwnershipRange(yc, &st, &ed));
    PetscCall(VecGetArrayRead(yc, &LA_yred));
    for (i = 0; i < ed - st; i++) array[i] = LA_yred[i];
    PetscCall(VecRestoreArrayRead(yc, &LA_yred));
  }
  PetscCall(VecRestoreArray(xtmp, &array));
  PetscCall(VecScatterBegin(scatter, xtmp, xp, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(scatter, xtmp, xp, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(MatMult(P, xp, y));
  PetscFunctionReturn(0);
}

PetscErrorCode DMFieldScatter_ShellDA(DM dmf_shell, Vec x, ScatterMode mode, DM dmc_shell, Vec xc)
{
  DM dmf = NULL, dmc = NULL;

  PetscFunctionBeginUser;
  PetscCall(DMShellGetContext(dmf_shell, &dmf));
  if (dmc_shell) PetscCall(DMShellGetContext(dmc_shell, &dmc));
  if (mode == SCATTER_FORWARD) {
    PetscCall(DMShellDAFieldScatter_Forward(dmf, x, dmc, xc));
  } else if (mode == SCATTER_REVERSE) {
    PetscCall(DMShellDAFieldScatter_Reverse(dmf, x, dmc, xc));
  } else SETERRQ(PetscObjectComm((PetscObject)dmf_shell), PETSC_ERR_SUP, "Only mode = SCATTER_FORWARD, SCATTER_REVERSE supported");
  PetscFunctionReturn(0);
}

PetscErrorCode DMStateScatter_ShellDA(DM dmf_shell, ScatterMode mode, DM dmc_shell)
{
  PetscMPIInt size_f = 0, size_c = 0;
  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dmf_shell), &size_f));
  if (dmc_shell) PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dmc_shell), &size_c));
  if (mode == SCATTER_FORWARD) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dmf_shell), "User supplied state scatter (fine [size %d]-> coarse [size %d])\n", (int)size_f, (int)size_c));
  } else if (mode == SCATTER_REVERSE) {
  } else SETERRQ(PetscObjectComm((PetscObject)dmf_shell), PETSC_ERR_SUP, "Only mode = SCATTER_FORWARD, SCATTER_REVERSE supported");
  PetscFunctionReturn(0);
}

PetscErrorCode DMShellCreate_ShellDA(DM da, DM *dms)
{
  PetscFunctionBeginUser;
  if (da) {
    PetscCall(DMShellCreate(PetscObjectComm((PetscObject)da), dms));
    PetscCall(DMShellSetContext(*dms, da));
    PetscCall(DMShellSetCreateGlobalVector(*dms, DMCreateGlobalVector_ShellDA));
    PetscCall(DMShellSetCreateLocalVector(*dms, DMCreateLocalVector_ShellDA));
    PetscCall(DMShellSetCreateMatrix(*dms, DMCreateMatrix_ShellDA));
    PetscCall(DMShellSetCoarsen(*dms, DMCoarsen_ShellDA));
    PetscCall(DMShellSetCreateInterpolation(*dms, DMCreateInterpolation_ShellDA));
  } else {
    *dms = NULL;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMDestroyShellDMDA(DM *_dm)
{
  DM dm, da = NULL;

  PetscFunctionBeginUser;
  if (!_dm) PetscFunctionReturn(0);
  dm = *_dm;
  if (!dm) PetscFunctionReturn(0);

  PetscCall(DMShellGetContext(dm, &da));
  if (da) {
    Vec        vec;
    VecScatter scatter = NULL;
    IS         is      = NULL;
    Mat        P       = NULL;

    PetscCall(PetscObjectQuery((PetscObject)da, "P", (PetscObject *)&P));
    PetscCall(MatDestroy(&P));

    vec = NULL;
    PetscCall(PetscObjectQuery((PetscObject)da, "xp", (PetscObject *)&vec));
    PetscCall(VecDestroy(&vec));

    PetscCall(PetscObjectQuery((PetscObject)da, "scatter", (PetscObject *)&scatter));
    PetscCall(VecScatterDestroy(&scatter));

    vec = NULL;
    PetscCall(PetscObjectQuery((PetscObject)da, "xtmp", (PetscObject *)&vec));
    PetscCall(VecDestroy(&vec));

    PetscCall(PetscObjectQuery((PetscObject)da, "isin", (PetscObject *)&is));
    PetscCall(ISDestroy(&is));

    PetscCall(DMDestroy(&da));
  }
  PetscCall(PetscObjectComposeFunction((PetscObject)dm, "PCTelescopeFieldScatter", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)dm, "PCTelescopeStateScatter", NULL));
  PetscCall(DMDestroy(&dm));
  *_dm = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode HierarchyCreate_Basic(DM *dm_f, DM *dm_c, UserContext *ctx)
{
  DM          dm, dmc, dm_shell, dmc_shell;
  PetscMPIInt rank;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 17, 17, PETSC_DECIDE, PETSC_DECIDE, 1, 1, 0, 0, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMDASetUniformCoordinates(dm, 0, 1, 0, 1, 0, 0));
  PetscCall(DMDASetFieldName(dm, 0, "Pressure"));
  PetscCall(DMShellCreate_ShellDA(dm, &dm_shell));
  PetscCall(DMSetApplicationContext(dm_shell, ctx));

  dmc       = NULL;
  dmc_shell = NULL;
  if (rank == 0) {
    PetscCall(DMDACreate2d(PETSC_COMM_SELF, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 17, 17, PETSC_DECIDE, PETSC_DECIDE, 1, 1, 0, 0, &dmc));
    PetscCall(DMSetFromOptions(dmc));
    PetscCall(DMSetUp(dmc));
    PetscCall(DMDASetUniformCoordinates(dmc, 0, 1, 0, 1, 0, 0));
    PetscCall(DMDASetFieldName(dmc, 0, "Pressure"));
    PetscCall(DMShellCreate_ShellDA(dmc, &dmc_shell));
    PetscCall(DMSetApplicationContext(dmc_shell, ctx));
  }

  PetscCall(DMSetCoarseDM(dm_shell, dmc_shell));
  PetscCall(DMShellDASetUp_TelescopeDMScatter(dm_shell, dmc_shell));

  *dm_f = dm_shell;
  *dm_c = dmc_shell;
  PetscFunctionReturn(0);
}

PetscErrorCode HierarchyCreate(PetscInt *_nd, PetscInt *_nref, MPI_Comm **_cl, DM **_dl)
{
  PetscInt      d, k, ndecomps, ncoarsen, found, nx;
  PetscInt      levelrefs, *number;
  PetscSubcomm *pscommlist;
  MPI_Comm     *commlist;
  DM           *dalist, *dmlist;
  PetscBool     set;

  PetscFunctionBeginUser;
  ndecomps = 1;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ndecomps", &ndecomps, NULL));
  ncoarsen = ndecomps - 1;
  PetscCheck(ncoarsen >= 0, PETSC_COMM_WORLD, PETSC_ERR_USER, "-ndecomps must be >= 1");

  levelrefs = 2;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-level_nrefs", &levelrefs, NULL));
  PetscCall(PetscMalloc1(ncoarsen + 1, &number));
  for (k = 0; k < ncoarsen + 1; k++) number[k] = 2;
  found = ncoarsen;
  set   = PETSC_FALSE;
  PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-level_comm_red_factor", number, &found, &set));
  if (set) PetscCheck(found == ncoarsen, PETSC_COMM_WORLD, PETSC_ERR_USER, "Expected %" PetscInt_FMT " values for -level_comm_red_factor. Found %" PetscInt_FMT, ncoarsen, found);

  PetscCall(PetscMalloc1(ncoarsen + 1, &pscommlist));
  for (k = 0; k < ncoarsen + 1; k++) pscommlist[k] = NULL;

  PetscCall(PetscMalloc1(ndecomps, &commlist));
  for (k = 0; k < ndecomps; k++) commlist[k] = MPI_COMM_NULL;
  PetscCall(PetscMalloc1(ndecomps * levelrefs, &dalist));
  PetscCall(PetscMalloc1(ndecomps * levelrefs, &dmlist));
  for (k = 0; k < ndecomps * levelrefs; k++) {
    dalist[k] = NULL;
    dmlist[k] = NULL;
  }

  PetscCall(CommHierarchyCreate(PETSC_COMM_WORLD, ncoarsen, number, pscommlist));
  for (k = 0; k < ncoarsen; k++) {
    if (pscommlist[k]) {
      MPI_Comm comm_k = PetscSubcommChild(pscommlist[k]);
      if (pscommlist[k]->color == 0) PetscCall(PetscCommDuplicate(comm_k, &commlist[k], NULL));
    }
  }
  PetscCall(PetscCommDuplicate(PETSC_COMM_WORLD, &commlist[ndecomps - 1], NULL));

  for (k = 0; k < ncoarsen; k++) {
    if (pscommlist[k]) PetscCall(PetscSubcommDestroy(&pscommlist[k]));
  }

  nx = 17;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &nx, NULL));
  for (d = 0; d < ndecomps; d++) {
    DM   dmroot = NULL;
    char name[PETSC_MAX_PATH_LEN];

    if (commlist[d] != MPI_COMM_NULL) {
      PetscCall(DMDACreate2d(commlist[d], DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, nx, nx, PETSC_DECIDE, PETSC_DECIDE, 1, 1, 0, 0, &dmroot));
      PetscCall(DMSetUp(dmroot));
      PetscCall(DMDASetUniformCoordinates(dmroot, 0, 1, 0, 1, 0, 0));
      PetscCall(DMDASetFieldName(dmroot, 0, "Pressure"));
      PetscCall(PetscSNPrintf(name, PETSC_MAX_PATH_LEN - 1, "root-decomp-%" PetscInt_FMT, d));
      PetscCall(PetscObjectSetName((PetscObject)dmroot, name));
      /*PetscCall(DMView(dmroot,PETSC_VIEWER_STDOUT_(commlist[d])));*/
    }

    dalist[d * levelrefs + 0] = dmroot;
    for (k = 1; k < levelrefs; k++) {
      DM dmref = NULL;

      if (commlist[d] != MPI_COMM_NULL) {
        PetscCall(DMRefine(dalist[d * levelrefs + (k - 1)], MPI_COMM_NULL, &dmref));
        PetscCall(PetscSNPrintf(name, PETSC_MAX_PATH_LEN - 1, "ref%" PetscInt_FMT "-decomp-%" PetscInt_FMT, k, d));
        PetscCall(PetscObjectSetName((PetscObject)dmref, name));
        PetscCall(DMDAGetInfo(dmref, NULL, &nx, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
        /*PetscCall(DMView(dmref,PETSC_VIEWER_STDOUT_(commlist[d])));*/
      }
      dalist[d * levelrefs + k] = dmref;
    }
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &nx, 1, MPIU_INT, MPI_MAX, PETSC_COMM_WORLD));
  }

  /* create the hierarchy of DMShell's */
  for (d = 0; d < ndecomps; d++) {
    char name[PETSC_MAX_PATH_LEN];

    UserContext *ctx = NULL;
    if (commlist[d] != MPI_COMM_NULL) {
      PetscCall(UserContextCreate(commlist[d], &ctx));
      for (k = 0; k < levelrefs; k++) {
        PetscCall(DMShellCreate_ShellDA(dalist[d * levelrefs + k], &dmlist[d * levelrefs + k]));
        PetscCall(DMSetApplicationContext(dmlist[d * levelrefs + k], ctx));
        PetscCall(PetscSNPrintf(name, PETSC_MAX_PATH_LEN - 1, "level%" PetscInt_FMT "-decomp-%" PetscInt_FMT, k, d));
        PetscCall(PetscObjectSetName((PetscObject)dmlist[d * levelrefs + k], name));
      }
    }
  }

  /* set all the coarse DMs */
  for (k = 1; k < ndecomps * levelrefs; k++) { /* skip first DM as it doesn't have a coarse representation */
    DM dmfine = NULL, dmcoarse = NULL;

    dmfine   = dmlist[k];
    dmcoarse = dmlist[k - 1];
    if (dmfine) PetscCall(DMSetCoarseDM(dmfine, dmcoarse));
  }

  /* do special setup on the fine DM coupling different decompositions */
  for (d = 1; d < ndecomps; d++) { /* skip first decomposition as it doesn't have a coarse representation */
    DM dmfine = NULL, dmcoarse = NULL;

    dmfine   = dmlist[d * levelrefs + 0];
    dmcoarse = dmlist[(d - 1) * levelrefs + (levelrefs - 1)];
    if (dmfine) PetscCall(DMShellDASetUp_TelescopeDMScatter(dmfine, dmcoarse));
  }

  PetscCall(PetscFree(number));
  for (k = 0; k < ncoarsen; k++) PetscCall(PetscSubcommDestroy(&pscommlist[k]));
  PetscCall(PetscFree(pscommlist));

  if (_nd) *_nd = ndecomps;
  if (_nref) *_nref = levelrefs;
  if (_cl) {
    *_cl = commlist;
  } else {
    for (k = 0; k < ndecomps; k++) {
      if (commlist[k] != MPI_COMM_NULL) PetscCall(PetscCommDestroy(&commlist[k]));
    }
    PetscCall(PetscFree(commlist));
  }
  if (_dl) {
    *_dl = dmlist;
    PetscCall(PetscFree(dalist));
  } else {
    for (k = 0; k < ndecomps * levelrefs; k++) {
      PetscCall(DMDestroy(&dmlist[k]));
      PetscCall(DMDestroy(&dalist[k]));
    }
    PetscCall(PetscFree(dmlist));
    PetscCall(PetscFree(dalist));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode test_hierarchy(void)
{
  PetscInt  d, k, nd, nref;
  MPI_Comm *comms;
  DM       *dms;

  PetscFunctionBeginUser;
  PetscCall(HierarchyCreate(&nd, &nref, &comms, &dms));

  /* destroy user context */
  for (d = 0; d < nd; d++) {
    DM first = dms[d * nref + 0];

    if (first) {
      UserContext *ctx = NULL;

      PetscCall(DMGetApplicationContext(first, &ctx));
      if (ctx) PetscCall(PetscFree(ctx));
      PetscCall(DMSetApplicationContext(first, NULL));
    }
    for (k = 1; k < nref; k++) {
      DM dm = dms[d * nref + k];
      if (dm) PetscCall(DMSetApplicationContext(dm, NULL));
    }
  }

  /* destroy DMs */
  for (k = 0; k < nd * nref; k++) {
    if (dms[k]) PetscCall(DMDestroyShellDMDA(&dms[k]));
  }
  PetscCall(PetscFree(dms));

  /* destroy communicators */
  for (k = 0; k < nd; k++) {
    if (comms[k] != MPI_COMM_NULL) PetscCall(PetscCommDestroy(&comms[k]));
  }
  PetscCall(PetscFree(comms));
  PetscFunctionReturn(0);
}

PetscErrorCode test_basic(void)
{
  DM           dmF, dmdaF = NULL, dmC = NULL;
  Mat          A;
  Vec          x, b;
  KSP          ksp;
  PC           pc;
  UserContext *user = NULL;

  PetscFunctionBeginUser;
  PetscCall(UserContextCreate(PETSC_COMM_WORLD, &user));
  PetscCall(HierarchyCreate_Basic(&dmF, &dmC, user));
  PetscCall(DMShellGetContext(dmF, &dmdaF));

  PetscCall(DMCreateMatrix(dmF, &A));
  PetscCall(DMCreateGlobalVector(dmF, &x));
  PetscCall(DMCreateGlobalVector(dmF, &b));
  PetscCall(ComputeRHS_DMDA(dmdaF, b, user));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetComputeOperators(ksp, ComputeMatrix_ShellDA, user));
  /*PetscCall(KSPSetOperators(ksp,A,A));*/
  PetscCall(KSPSetDM(ksp, dmF));
  PetscCall(KSPSetDMActive(ksp, PETSC_TRUE));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCTelescopeSetUseCoarseDM(pc, PETSC_TRUE));

  PetscCall(KSPSolve(ksp, b, x));

  if (dmC) PetscCall(DMDestroyShellDMDA(&dmC));
  PetscCall(DMDestroyShellDMDA(&dmF));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x));
  PetscCall(PetscFree(user));
  PetscFunctionReturn(0);
}

PetscErrorCode test_mg(void)
{
  DM           dmF, dmdaF = NULL, *dms = NULL;
  Mat          A;
  Vec          x, b;
  KSP          ksp;
  PetscInt     k, d, nd, nref;
  MPI_Comm    *comms = NULL;
  UserContext *user  = NULL;

  PetscFunctionBeginUser;
  PetscCall(HierarchyCreate(&nd, &nref, &comms, &dms));
  dmF = dms[nd * nref - 1];

  PetscCall(DMShellGetContext(dmF, &dmdaF));
  PetscCall(DMGetApplicationContext(dmF, &user));

  PetscCall(DMCreateMatrix(dmF, &A));
  PetscCall(DMCreateGlobalVector(dmF, &x));
  PetscCall(DMCreateGlobalVector(dmF, &b));
  PetscCall(ComputeRHS_DMDA(dmdaF, b, user));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetComputeOperators(ksp, ComputeMatrix_ShellDA, user));
  /*PetscCall(KSPSetOperators(ksp,A,A));*/
  PetscCall(KSPSetDM(ksp, dmF));
  PetscCall(KSPSetDMActive(ksp, PETSC_TRUE));
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(KSPSolve(ksp, b, x));

  for (d = 0; d < nd; d++) {
    DM first = dms[d * nref + 0];

    if (first) {
      UserContext *ctx = NULL;

      PetscCall(DMGetApplicationContext(first, &ctx));
      if (ctx) PetscCall(PetscFree(ctx));
      PetscCall(DMSetApplicationContext(first, NULL));
    }
    for (k = 1; k < nref; k++) {
      DM dm = dms[d * nref + k];
      if (dm) PetscCall(DMSetApplicationContext(dm, NULL));
    }
  }

  for (k = 0; k < nd * nref; k++) {
    if (dms[k]) PetscCall(DMDestroyShellDMDA(&dms[k]));
  }
  PetscCall(PetscFree(dms));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x));

  for (k = 0; k < nd; k++) {
    if (comms[k] != MPI_COMM_NULL) PetscCall(PetscCommDestroy(&comms[k]));
  }
  PetscCall(PetscFree(comms));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscInt test_id = 0;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-tid", &test_id, NULL));
  switch (test_id) {
  case 0:
    PetscCall(test_basic());
    break;
  case 1:
    PetscCall(test_hierarchy());
    break;
  case 2:
    PetscCall(test_mg());
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "-tid must be 0,1,2");
  }
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode ComputeRHS_DMDA(DM da, Vec b, void *ctx)
{
  UserContext  *user = (UserContext *)ctx;
  PetscInt      i, j, mx, my, xm, ym, xs, ys;
  PetscScalar   Hx, Hy;
  PetscScalar **array;
  PetscBool     isda = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectTypeCompare((PetscObject)da, DMDA, &isda));
  PetscCheck(isda, PetscObjectComm((PetscObject)da), PETSC_ERR_USER, "DM provided must be a DMDA");
  PetscCall(DMDAGetInfo(da, NULL, &mx, &my, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  Hx = 1.0 / (PetscReal)(mx - 1);
  Hy = 1.0 / (PetscReal)(my - 1);
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));
  PetscCall(DMDAVecGetArray(da, b, &array));
  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) array[j][i] = PetscExpScalar(-((PetscReal)i * Hx) * ((PetscReal)i * Hx) / user->nu) * PetscExpScalar(-((PetscReal)j * Hy) * ((PetscReal)j * Hy) / user->nu) * Hx * Hy;
  }
  PetscCall(DMDAVecRestoreArray(da, b, &array));
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));

  /* force right hand side to be consistent for singular matrix */
  /* note this is really a hack, normally the model would provide you with a consistent right handside */
  if (user->bcType == NEUMANN) {
    MatNullSpace nullspace;

    PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace));
    PetscCall(MatNullSpaceRemove(nullspace, b));
    PetscCall(MatNullSpaceDestroy(&nullspace));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeRho(PetscInt i, PetscInt j, PetscInt mx, PetscInt my, PetscReal centerRho, PetscReal *rho)
{
  PetscFunctionBeginUser;
  if ((i > mx / 3.0) && (i < 2.0 * mx / 3.0) && (j > my / 3.0) && (j < 2.0 * my / 3.0)) {
    *rho = centerRho;
  } else {
    *rho = 1.0;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeMatrix_DMDA(DM da, Mat J, Mat jac, void *ctx)
{
  UserContext *user = (UserContext *)ctx;
  PetscReal    centerRho;
  PetscInt     i, j, mx, my, xm, ym, xs, ys;
  PetscScalar  v[5];
  PetscReal    Hx, Hy, HydHx, HxdHy, rho;
  MatStencil   row, col[5];
  PetscBool    isda = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectTypeCompare((PetscObject)da, DMDA, &isda));
  PetscCheck(isda, PetscObjectComm((PetscObject)da), PETSC_ERR_USER, "DM provided must be a DMDA");
  PetscCall(MatZeroEntries(jac));
  centerRho = user->rho;
  PetscCall(DMDAGetInfo(da, NULL, &mx, &my, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  Hx    = 1.0 / (PetscReal)(mx - 1);
  Hy    = 1.0 / (PetscReal)(my - 1);
  HxdHy = Hx / Hy;
  HydHx = Hy / Hx;
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));
  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      row.i = i;
      row.j = j;
      PetscCall(ComputeRho(i, j, mx, my, centerRho, &rho));
      if (i == 0 || j == 0 || i == mx - 1 || j == my - 1) {
        if (user->bcType == DIRICHLET) {
          v[0] = 2.0 * rho * (HxdHy + HydHx);
          PetscCall(MatSetValuesStencil(jac, 1, &row, 1, &row, v, INSERT_VALUES));
        } else if (user->bcType == NEUMANN) {
          PetscInt numx = 0, numy = 0, num = 0;
          if (j != 0) {
            v[num]     = -rho * HxdHy;
            col[num].i = i;
            col[num].j = j - 1;
            numy++;
            num++;
          }
          if (i != 0) {
            v[num]     = -rho * HydHx;
            col[num].i = i - 1;
            col[num].j = j;
            numx++;
            num++;
          }
          if (i != mx - 1) {
            v[num]     = -rho * HydHx;
            col[num].i = i + 1;
            col[num].j = j;
            numx++;
            num++;
          }
          if (j != my - 1) {
            v[num]     = -rho * HxdHy;
            col[num].i = i;
            col[num].j = j + 1;
            numy++;
            num++;
          }
          v[num]     = numx * rho * HydHx + numy * rho * HxdHy;
          col[num].i = i;
          col[num].j = j;
          num++;
          PetscCall(MatSetValuesStencil(jac, 1, &row, num, col, v, INSERT_VALUES));
        }
      } else {
        v[0]     = -rho * HxdHy;
        col[0].i = i;
        col[0].j = j - 1;
        v[1]     = -rho * HydHx;
        col[1].i = i - 1;
        col[1].j = j;
        v[2]     = 2.0 * rho * (HxdHy + HydHx);
        col[2].i = i;
        col[2].j = j;
        v[3]     = -rho * HydHx;
        col[3].i = i + 1;
        col[3].j = j;
        v[4]     = -rho * HxdHy;
        col[4].i = i;
        col[4].j = j + 1;
        PetscCall(MatSetValuesStencil(jac, 1, &row, 5, col, v, INSERT_VALUES));
      }
    }
  }
  PetscCall(MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeMatrix_ShellDA(KSP ksp, Mat J, Mat jac, void *ctx)
{
  DM dm, da;
  PetscFunctionBeginUser;
  PetscCall(KSPGetDM(ksp, &dm));
  PetscCall(DMShellGetContext(dm, &da));
  PetscCall(ComputeMatrix_DMDA(da, J, jac, ctx));
  PetscFunctionReturn(0);
}

/*TEST

  test:
    suffix: basic_dirichlet
    nsize: 4
    args: -tid 0 -ksp_monitor_short -pc_type telescope -telescope_ksp_max_it 100000 -telescope_pc_type lu -telescope_ksp_type fgmres -telescope_ksp_monitor_short -ksp_type gcr

  test:
    suffix: basic_neumann
    nsize: 4
    requires: !single
    args: -tid 0 -ksp_monitor_short -pc_type telescope -telescope_ksp_max_it 100000 -telescope_pc_type jacobi -telescope_ksp_type fgmres -telescope_ksp_monitor_short -ksp_type gcr -bc_type neumann

  test:
    suffix: mg_2lv_2mg
    nsize: 6
    args: -tid 2 -ksp_monitor_short -ksp_type gcr -pc_type mg -pc_mg_levels 3 -level_nrefs 3 -ndecomps 2 -mg_coarse_pc_type telescope -mg_coarse_pc_telescope_use_coarse_dm -mg_coarse_telescope_pc_type mg -mg_coarse_telescope_pc_mg_levels 3 -level_comm_red_factor 6 -mg_coarse_telescope_mg_coarse_pc_type lu

  test:
    suffix: mg_3lv_2mg
    nsize: 4
    args: -tid 2 -ksp_monitor_short -ksp_type gcr -pc_type mg -pc_mg_levels 3 -level_nrefs 3 -ndecomps 3 -mg_coarse_pc_type telescope -mg_coarse_pc_telescope_use_coarse_dm -mg_coarse_telescope_pc_type mg -mg_coarse_telescope_pc_mg_levels 3 -mg_coarse_telescope_mg_coarse_pc_type telescope -mg_coarse_telescope_mg_coarse_pc_telescope_use_coarse_dm -mg_coarse_telescope_mg_coarse_telescope_pc_type lu -m 5

  test:
    suffix: mg_3lv_2mg_customcommsize
    nsize: 12
    args: -tid 2 -ksp_monitor_short -ksp_type gcr -pc_type mg -pc_mg_levels 3 -level_nrefs 3 -ndecomps 3 -mg_coarse_pc_type telescope -mg_coarse_pc_telescope_use_coarse_dm  -m 5 -level_comm_red_factor 2,6 -mg_coarse_telescope_pc_type mg -mg_coarse_telescope_pc_mg_levels 3 -mg_coarse_telescope_mg_coarse_pc_type telescope -mg_coarse_telescope_mg_coarse_pc_telescope_use_coarse_dm -mg_coarse_telescope_mg_coarse_telescope_pc_type lu

 TEST*/
