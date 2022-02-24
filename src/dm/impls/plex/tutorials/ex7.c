static char help[] = "Create a Plex sphere from quads and create a P1 section\n\n";

#include <petscdmplex.h>

static PetscErrorCode SetupSection(DM dm)
{
  PetscSection   s;
  PetscInt       vStart, vEnd, v;

  PetscFunctionBeginUser;
  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject) dm), &s));
  CHKERRQ(PetscSectionSetNumFields(s, 1));
  CHKERRQ(PetscSectionSetFieldComponents(s, 0, 1));
  CHKERRQ(PetscSectionSetChart(s, vStart, vEnd));
  for (v = vStart; v < vEnd; ++v) {
    CHKERRQ(PetscSectionSetDof(s, v, 1));
    CHKERRQ(PetscSectionSetFieldDof(s, v, 0, 1));
  }
  CHKERRQ(PetscSectionSetUp(s));
  CHKERRQ(DMSetLocalSection(dm, s));
  CHKERRQ(PetscSectionDestroy(&s));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  Vec            u;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  CHKERRQ(DMCreate(PETSC_COMM_WORLD, &dm));
  CHKERRQ(DMSetType(dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(dm));
  CHKERRQ(PetscObjectSetName((PetscObject) dm, "Sphere"));
  CHKERRQ(DMViewFromOptions(dm, NULL, "-dm_view"));

  CHKERRQ(SetupSection(dm));
  CHKERRQ(DMGetGlobalVector(dm, &u));
  CHKERRQ(VecSet(u, 2));
  CHKERRQ(VecViewFromOptions(u, NULL, "-vec_view"));
  CHKERRQ(DMRestoreGlobalVector(dm, &u));
  CHKERRQ(DMDestroy(&dm));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  testset:
    requires: !__float128
    args: -dm_plex_shape sphere -dm_view

    test:
      suffix: 2d_quad
      args: -dm_plex_simplex 0

    test:
      suffix: 2d_tri
      args:

    test:
      suffix: 3d_tri
      args: -dm_plex_dim 3

  testset:
    requires: !__float128
    args: -dm_plex_shape sphere -petscpartitioner_type simple -dm_view

    test:
      suffix: 2d_quad_parallel
      nsize: 2
      args: -dm_plex_simplex 0

    test:
      suffix: 2d_tri_parallel
      nsize: 2

    test:
      suffix: 3d_tri_parallel
      nsize: 2
      args: -dm_plex_dim 3

TEST*/
