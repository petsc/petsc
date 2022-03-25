static char help[] = "Create a Plex sphere from quads and create a P1 section\n\n";

#include <petscdmplex.h>

static PetscErrorCode SetupSection(DM dm)
{
  PetscSection   s;
  PetscInt       vStart, vEnd, v;

  PetscFunctionBeginUser;
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject) dm), &s));
  PetscCall(PetscSectionSetNumFields(s, 1));
  PetscCall(PetscSectionSetFieldComponents(s, 0, 1));
  PetscCall(PetscSectionSetChart(s, vStart, vEnd));
  for (v = vStart; v < vEnd; ++v) {
    PetscCall(PetscSectionSetDof(s, v, 1));
    PetscCall(PetscSectionSetFieldDof(s, v, 0, 1));
  }
  PetscCall(PetscSectionSetUp(s));
  PetscCall(DMSetLocalSection(dm, s));
  PetscCall(PetscSectionDestroy(&s));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  Vec            u;

  PetscCall(PetscInitialize(&argc, &argv, NULL,help));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(PetscObjectSetName((PetscObject) dm, "Sphere"));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  PetscCall(SetupSection(dm));
  PetscCall(DMGetGlobalVector(dm, &u));
  PetscCall(VecSet(u, 2));
  PetscCall(VecViewFromOptions(u, NULL, "-vec_view"));
  PetscCall(DMRestoreGlobalVector(dm, &u));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
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
