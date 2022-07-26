static char help[] = "Test section ordering for FEM discretizations\n\n";

#include <petscdmplex.h>
#include <petscds.h>

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestLocalDofOrder(DM dm)
{
  PetscFE        fe[3];
  PetscSection   s;
  PetscBool      simplex;
  PetscInt       dim, Nf, f;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, dim, simplex, "field0_", -1, &fe[0]));
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1,   simplex, "field1_", -1, &fe[1]));
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1,   simplex, "field2_", -1, &fe[2]));

  PetscCall(DMSetField(dm, 0, NULL, (PetscObject) fe[0]));
  PetscCall(DMSetField(dm, 1, NULL, (PetscObject) fe[1]));
  PetscCall(DMSetField(dm, 2, NULL, (PetscObject) fe[2]));
  PetscCall(DMCreateDS(dm));
  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(PetscObjectViewFromOptions((PetscObject) s, NULL, "-dof_view"));

  PetscCall(DMGetNumFields(dm, &Nf));
  for (f = 0; f < Nf; ++f) PetscCall(PetscFEDestroy(&fe[f]));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &dm));
  PetscCall(TestLocalDofOrder(dm));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: tri_pm
    requires: triangle
    args: -dm_plex_box_faces 1,1 -field0_petscspace_degree 2 -field1_petscspace_degree 1 -field2_petscspace_degree 1 -dm_view -dof_view

  test:
    suffix: quad_pm
    requires:
    args: -dm_plex_simplex 0 -dm_plex_box_faces 1,1 -field0_petscspace_degree 2 -field1_petscspace_degree 1 -field2_petscspace_degree 1 -dm_view -dof_view

  test:
    suffix: tri_fm
    requires: triangle
    args: -dm_coord_space 0 -dm_plex_box_faces 1,1 -field0_petscspace_degree 2 -field1_petscspace_degree 1 -field2_petscspace_degree 1 -petscsection_point_major 0 -dm_view -dof_view

  test:
    suffix: quad_fm
    requires:
    args: -dm_coord_space 0 -dm_plex_simplex 0 -dm_plex_box_faces 1,1 -field0_petscspace_degree 2 -field1_petscspace_degree 1 -field2_petscspace_degree 1 -petscsection_point_major 0 -dm_view -dof_view

TEST*/
