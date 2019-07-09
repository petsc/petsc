static char help[] = "Tests for creation of submeshes\n\n";

#include <petscdmplex.h>

typedef struct {
  PetscInt  debug;       /* The debugging level */
  PetscInt  dim;         /* The topological mesh dimension */
  PetscBool cellSimplex; /* Use simplices or hexes */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug       = 0;
  options->dim         = 2;
  options->cellSimplex = PETSC_TRUE;

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-debug", "The debugging level", "ex16.c", options->debug, &options->debug, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-dim", "The topological mesh dimension", "ex16.c", options->dim, &options->dim, NULL,1,3);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-cell_simplex", "Use simplices if true, otherwise hexes", "ex16.c", options->cellSimplex, &options->cellSimplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim         = user->dim;
  PetscBool      cellSimplex = user->cellSimplex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexCreateBoxMesh(comm, dim, cellSimplex, NULL, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  {
    DM pdm = NULL;

    /* Distribute mesh over processes */
    ierr = DMPlexDistribute(*dm, 0, NULL, &pdm);CHKERRQ(ierr);
    if (pdm) {
      ierr = DMViewFromOptions(pdm, NULL, "-dm_view");CHKERRQ(ierr);
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = pdm;
    }
  }
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CreateSubmesh(DM dm, PetscBool start, DM *subdm)
{
  DMLabel        label, map;
  PetscInt       cStart, cEnd, cStartSub, cEndSub, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMCreateLabel(dm, "cells");CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "cells", &label);CHKERRQ(ierr);
  ierr = DMLabelClearStratum(label, 1);CHKERRQ(ierr);
  if (start) {cStartSub = cStart; cEndSub = cEnd/2;}
  else       {cStartSub = cEnd/2; cEndSub = cEnd;}
  for (c = cStartSub; c < cEndSub; ++c) {ierr = DMLabelSetValue(label, c, 1);CHKERRQ(ierr);}
  ierr = DMPlexFilter(dm, label, 1, subdm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *subdm, "Submesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*subdm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = DMPlexGetSubpointMap(*subdm, &map);CHKERRQ(ierr);
  ierr = DMLabelView(map, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm, subdm;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = CreateSubmesh(dm, PETSC_TRUE, &subdm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(subdm);CHKERRQ(ierr);
  ierr = DMDestroy(&subdm);CHKERRQ(ierr);
  ierr = CreateSubmesh(dm, PETSC_FALSE, &subdm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(subdm);CHKERRQ(ierr);
  ierr = DMDestroy(&subdm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    requires: triangle
    args: -dm_view ascii::ascii_info_detail -dm_plex_check_symmetry -dm_plex_check_skeleton -dm_plex_check_faces

TEST*/
