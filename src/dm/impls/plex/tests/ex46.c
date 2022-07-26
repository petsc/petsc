static char help[] = "Tests 1D nested mesh refinement.\n\n";

#include <petscdmplex.h>
#include <petscds.h>

typedef struct {
  PetscInt             Nr;       /* Number of refinements */
  PetscSimplePointFunc funcs[2]; /* Functions to test */
} AppCtx;

static PetscErrorCode constant(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 1.;
  return 0;
}

static PetscErrorCode linear(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 2.*x[0] + 1.;
  return 0;
}

static PetscErrorCode quadratic(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 3.*x[0]*x[0] + 2.*x[0] + 1.;
  return 0;
}

static PetscErrorCode cubic(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 4.*x[0]*x[0]*x[0] + 3.*x[0]*x[0] + 2.*x[0] + 1.;
  return 0;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  options->Nr = 1;
  PetscOptionsBegin(comm, "", "1D Refinement Options", "DMPLEX");
  PetscCall(PetscOptionsInt("-num_refine", "Refine cycles", "ex46.c", options->Nr, &options->Nr, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMSetApplicationContext(*dm, user));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe;
  PetscSpace     sp;
  PetscInt       dim, deg;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, PETSC_FALSE, NULL, -1, &fe));
  PetscCall(PetscObjectSetName((PetscObject) fe, "scalar"));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject) fe));
  PetscCall(DMSetField(dm, 1, NULL, (PetscObject) fe));
  PetscCall(DMCreateDS(dm));
  while (cdm) {
    PetscCall(DMCopyDisc(dm,cdm));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  PetscCall(PetscFEGetBasisSpace(fe, &sp));
  PetscCall(PetscSpaceGetDegree(sp, &deg, NULL));
  switch (deg) {
  case 0: user->funcs[0] = constant;break;
  case 1: user->funcs[0] = linear;break;
  case 2: user->funcs[0] = quadratic;break;
  case 3: user->funcs[0] = cubic;break;
  default: SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Could not determine function to test for degree %" PetscInt_FMT, deg);
  }
  user->funcs[1] = user->funcs[0];
  PetscCall(PetscFEDestroy(&fe));
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckError(DM dm, Vec u, PetscSimplePointFunc funcs[])
{
  PetscReal      error, tol = PETSC_SMALL;
  MPI_Comm       comm;

  PetscFunctionBeginUser;
  PetscCall(DMComputeL2Diff(dm, 0.0, funcs, NULL, u, &error));
  PetscCall(PetscObjectGetComm((PetscObject) dm, &comm));
  if (error > tol) PetscCall(PetscPrintf(comm, "Function tests FAIL at tolerance %g error %g\n", (double)tol,(double) error));
  else             PetscCall(PetscPrintf(comm, "Function tests pass at tolerance %g\n", (double)tol));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  Vec            u;
  AppCtx         user;
  PetscInt       cStart, cEnd, c, r;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(SetupDiscretization(dm, &user));
  PetscCall(DMGetGlobalVector(dm, &u));
  PetscCall(DMProjectFunction(dm, 0.0, user.funcs, NULL, INSERT_ALL_VALUES, u));
  PetscCall(CheckError(dm, u, user.funcs));
  for (r = 0; r < user.Nr; ++r) {
    DM      adm;
    DMLabel adapt;
    Vec     au;
    Mat     Interp;

    PetscCall(DMLabelCreate(PETSC_COMM_SELF, "adapt", &adapt));
    PetscCall(DMLabelSetDefaultValue(adapt, DM_ADAPT_COARSEN));
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    for (c = cStart; c < cEnd; ++c) {
      if (c % 2) PetscCall(DMLabelSetValue(adapt, c, DM_ADAPT_REFINE));
    }
    PetscCall(DMAdaptLabel(dm, adapt, &adm));
    PetscCall(DMLabelDestroy(&adapt));
    PetscCall(PetscObjectSetName((PetscObject) adm, "Adapted Mesh"));
    PetscCall(DMViewFromOptions(adm, NULL, "-dm_view"));

    PetscCall(DMCreateInterpolation(dm, adm, &Interp, NULL));
    PetscCall(DMGetGlobalVector(adm, &au));
    PetscCall(MatInterpolate(Interp, u, au));
    PetscCall(CheckError(adm, au, user.funcs));
    PetscCall(MatDestroy(&Interp));
    PetscCall(DMRestoreGlobalVector(dm, &u));
    PetscCall(DMDestroy(&dm));
    dm   = adm;
    u    = au;
  }
  PetscCall(DMRestoreGlobalVector(dm, &u));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    args: -num_refine 4 -petscspace_degree 3 \
          -dm_plex_dim 1 -dm_plex_box_faces 5 -dm_plex_transform_type refine_1d -dm_plex_hash_location -dm_view

TEST*/
