static char help[] = "Create a Plex sphere from quads and create a P1 section\n\n";

#include <petscdmplex.h>

static PetscErrorCode SetupSection(DM dm)
{
  PetscSection   s;
  PetscInt       vStart, vEnd, v;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &s);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(s, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(s, 0, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(s, vStart, vEnd);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    ierr = PetscSectionSetDof(s, v, 1);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(s, v, 0, 1);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(s);CHKERRQ(ierr);
  ierr = DMSetLocalSection(dm, s);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  Vec            u;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = DMCreate(PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dm, "Sphere");CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);

  ierr = SetupSection(dm);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = VecSet(u, 2);CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-vec_view");CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
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
