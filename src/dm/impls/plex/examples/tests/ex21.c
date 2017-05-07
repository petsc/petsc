static char help[] = "Performance tests for DMPlex query operations\n\n";

#include <petscdmplex.h>

typedef struct {
  PetscInt  dim;             /* The topological mesh dimension */
  PetscBool cellSimplex;     /* Flag for simplices */
  PetscInt  n;               /* The number of faces per dimension for mesh */
} AppCtx;

static PetscErrorCode ProcessOptions(AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->dim         = 2;
  options->cellSimplex = PETSC_TRUE;
  options->n           = 2;

  ierr = PetscOptionsBegin(PETSC_COMM_SELF, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex21.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-cellSimplex", "Flag for simplices", "ex21.c", options->cellSimplex, &options->cellSimplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-n", "The number of faces per dimension", "ex21.c", options->n, &options->n, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim         = user->dim;
  PetscInt       n           = user->n;
  PetscBool      cellSimplex = user->cellSimplex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (cellSimplex) {
    ierr = DMPlexCreateBoxMesh(comm, dim, n, PETSC_FALSE, dm);CHKERRQ(ierr);
  } else {
    DM       odm;
    PetscInt cells[3];

    cells[0] = cells[1] = cells[2] = n;
    ierr = DMPlexCreateHexBoxMesh(comm, dim, cells, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, &odm);CHKERRQ(ierr);
    ierr = DMPlexUninterpolate(odm, dm);CHKERRQ(ierr);
    ierr = DMDestroy(&odm);CHKERRQ(ierr);
  }
  {
    DM ddm = NULL;

    ierr = DMPlexDistribute(*dm, 0, NULL, &ddm);CHKERRQ(ierr);
    if (ddm) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = ddm;
    }
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-orig_dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestInterpolate(DM dm, AppCtx *user)
{
  DM             idm = NULL;
  const char    *name;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexInterpolate(dm, &idm);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) dm, &name);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) idm, name);CHKERRQ(ierr);
  ierr = DMPlexCopyCoordinates(dm, idm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(idm, NULL, "-interp_dm_view");CHKERRQ(ierr);
  ierr = DMDestroy(&idm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(&user);CHKERRQ(ierr);
  ierr = PetscLogDefaultBegin();CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = TestInterpolate(dm, &user);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    requires: triangle
    args: -dim 2 -n 100
  test:
    suffix: 1
    requires: ctetgen
    args: -dim 3 -n 20

TEST*/
