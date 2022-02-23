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
  ierr = PetscOptionsInt("-num_refine", "Refine cycles", "ex46.c", options->Nr, &options->Nr, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(*dm, user);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe;
  PetscSpace     sp;
  PetscInt       dim, deg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, PETSC_FALSE, NULL, -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, "scalar");CHKERRQ(ierr);
  ierr = DMSetField(dm, 0, NULL, (PetscObject) fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  while (cdm) {
    ierr = DMCopyDisc(dm,cdm);CHKERRQ(ierr);
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEGetBasisSpace(fe, &sp);CHKERRQ(ierr);
  ierr = PetscSpaceGetDegree(sp, &deg, NULL);CHKERRQ(ierr);
  switch (deg) {
  case 0: user->funcs[0] = constant;break;
  case 1: user->funcs[0] = linear;break;
  case 2: user->funcs[0] = quadratic;break;
  case 3: user->funcs[0] = cubic;break;
  default: SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Could not determine function to test for degree %D", deg);
  }
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckError(DM dm, Vec u, PetscSimplePointFunc funcs[])
{
  PetscReal      error, tol = PETSC_SMALL;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMComputeL2Diff(dm, 0.0, funcs, NULL, u, &error);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  if (error > tol) {ierr = PetscPrintf(comm, "Function tests FAIL at tolerance %g error %g\n", (double)tol,(double) error);CHKERRQ(ierr);}
  else             {ierr = PetscPrintf(comm, "Function tests pass at tolerance %g\n", (double)tol);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  Vec            u;
  AppCtx         user;
  PetscInt       cStart, cEnd, c, r;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SetupDiscretization(dm, &user);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = DMProjectFunction(dm, 0.0, user.funcs, NULL, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
  ierr = CheckError(dm, u, user.funcs);CHKERRQ(ierr);
  for (r = 0; r < user.Nr; ++r) {
    DM      adm;
    DMLabel adapt;
    Vec     au;
    Mat     Interp;

    ierr = DMLabelCreate(PETSC_COMM_SELF, "adapt", &adapt);CHKERRQ(ierr);
    ierr = DMLabelSetDefaultValue(adapt, DM_ADAPT_COARSEN);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) {
      if (c % 2) {ierr = DMLabelSetValue(adapt, c, DM_ADAPT_REFINE);CHKERRQ(ierr);}
    }
    ierr = DMAdaptLabel(dm, adapt, &adm);CHKERRQ(ierr);
    ierr = DMLabelDestroy(&adapt);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) adm, "Adapted Mesh");CHKERRQ(ierr);
    ierr = DMViewFromOptions(adm, NULL, "-dm_view");CHKERRQ(ierr);

    ierr = DMCreateInterpolation(dm, adm, &Interp, NULL);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(adm, &au);CHKERRQ(ierr);
    ierr = MatInterpolate(Interp, u, au);CHKERRQ(ierr);
    ierr = CheckError(adm, au, user.funcs);CHKERRQ(ierr);
    ierr = MatDestroy(&Interp);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm, &u);CHKERRQ(ierr);
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    dm   = adm;
    u    = au;
  }
  ierr = DMRestoreGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    args: -num_refine 4 -petscspace_degree 3 \
          -dm_plex_dim 1 -dm_plex_box_faces 5 -dm_plex_transform_type refine_1d -dm_plex_hash_location -dm_view

TEST*/
