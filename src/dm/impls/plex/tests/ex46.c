static char help[] = "Tests 1D nested mesh refinement.\n\n";

#include <petscdmplex.h>
#include <petscds.h>

typedef struct {
  PetscInt             Nr;       /* Number of refinements */
  PetscSimplePointFunc funcs[1]; /* Functions to test */
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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->Nr = 1;
  ierr = PetscOptionsBegin(comm, "", "1D Refinement Options", "DMPLEX");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsInt("-num_refine", "Refine cycles", "ex46.c", options->Nr, &options->Nr, NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(DMSetApplicationContext(*dm, user));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe;
  PetscSpace     sp;
  PetscInt       dim, deg;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, PETSC_FALSE, NULL, -1, &fe));
  CHKERRQ(PetscObjectSetName((PetscObject) fe, "scalar"));
  CHKERRQ(DMSetField(dm, 0, NULL, (PetscObject) fe));
  CHKERRQ(DMCreateDS(dm));
  while (cdm) {
    CHKERRQ(DMCopyDisc(dm,cdm));
    CHKERRQ(DMGetCoarseDM(cdm, &cdm));
  }
  CHKERRQ(PetscFEGetBasisSpace(fe, &sp));
  CHKERRQ(PetscSpaceGetDegree(sp, &deg, NULL));
  switch (deg) {
  case 0: user->funcs[0] = constant;break;
  case 1: user->funcs[0] = linear;break;
  case 2: user->funcs[0] = quadratic;break;
  case 3: user->funcs[0] = cubic;break;
  default: SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Could not determine function to test for degree %D", deg);
  }
  CHKERRQ(PetscFEDestroy(&fe));
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckError(DM dm, Vec u, PetscSimplePointFunc funcs[])
{
  PetscReal      error, tol = PETSC_SMALL;
  MPI_Comm       comm;

  PetscFunctionBeginUser;
  CHKERRQ(DMComputeL2Diff(dm, 0.0, funcs, NULL, u, &error));
  CHKERRQ(PetscObjectGetComm((PetscObject) dm, &comm));
  if (error > tol) CHKERRQ(PetscPrintf(comm, "Function tests FAIL at tolerance %g error %g\n", (double)tol,(double) error));
  else             CHKERRQ(PetscPrintf(comm, "Function tests pass at tolerance %g\n", (double)tol));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  Vec            u;
  AppCtx         user;
  PetscInt       cStart, cEnd, c, r;

  CHKERRQ(PetscInitialize(&argc, &argv, NULL, help));
  CHKERRQ(ProcessOptions(PETSC_COMM_WORLD, &user));
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  CHKERRQ(SetupDiscretization(dm, &user));
  CHKERRQ(DMGetGlobalVector(dm, &u));
  CHKERRQ(DMProjectFunction(dm, 0.0, user.funcs, NULL, INSERT_ALL_VALUES, u));
  CHKERRQ(CheckError(dm, u, user.funcs));
  for (r = 0; r < user.Nr; ++r) {
    DM      adm;
    DMLabel adapt;
    Vec     au;
    Mat     Interp;

    CHKERRQ(DMLabelCreate(PETSC_COMM_SELF, "adapt", &adapt));
    CHKERRQ(DMLabelSetDefaultValue(adapt, DM_ADAPT_COARSEN));
    CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    for (c = cStart; c < cEnd; ++c) {
      if (c % 2) CHKERRQ(DMLabelSetValue(adapt, c, DM_ADAPT_REFINE));
    }
    CHKERRQ(DMAdaptLabel(dm, adapt, &adm));
    CHKERRQ(DMLabelDestroy(&adapt));
    CHKERRQ(PetscObjectSetName((PetscObject) adm, "Adapted Mesh"));
    CHKERRQ(DMViewFromOptions(adm, NULL, "-dm_view"));

    CHKERRQ(DMCreateInterpolation(dm, adm, &Interp, NULL));
    CHKERRQ(DMGetGlobalVector(adm, &au));
    CHKERRQ(MatInterpolate(Interp, u, au));
    CHKERRQ(CheckError(adm, au, user.funcs));
    CHKERRQ(MatDestroy(&Interp));
    CHKERRQ(DMRestoreGlobalVector(dm, &u));
    CHKERRQ(DMDestroy(&dm));
    dm   = adm;
    u    = au;
  }
  CHKERRQ(DMRestoreGlobalVector(dm, &u));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    args: -num_refine 4 -petscspace_degree 3 \
          -dm_plex_dim 1 -dm_plex_box_faces 5 -dm_plex_transform_type refine_1d -dm_plex_hash_location -dm_view

TEST*/
