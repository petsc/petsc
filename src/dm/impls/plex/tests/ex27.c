static char help[] = "Test section ordering for FEM discretizations\n\n";

#include <petscdmplex.h>
#include <petscds.h>

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscFunctionBegin;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestLocalDofOrder(DM dm)
{
  PetscFE        fe[3];
  PetscSection   s;
  PetscBool      simplex;
  PetscInt       dim, Nf, f;

  PetscFunctionBegin;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexIsSimplex(dm, &simplex));
  CHKERRQ(PetscFECreateDefault(PETSC_COMM_SELF, dim, dim, simplex, "field0_", -1, &fe[0]));
  CHKERRQ(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1,   simplex, "field1_", -1, &fe[1]));
  CHKERRQ(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1,   simplex, "field2_", -1, &fe[2]));

  CHKERRQ(DMSetField(dm, 0, NULL, (PetscObject) fe[0]));
  CHKERRQ(DMSetField(dm, 1, NULL, (PetscObject) fe[1]));
  CHKERRQ(DMSetField(dm, 2, NULL, (PetscObject) fe[2]));
  CHKERRQ(DMCreateDS(dm));
  CHKERRQ(DMGetLocalSection(dm, &s));
  CHKERRQ(PetscObjectViewFromOptions((PetscObject) s, NULL, "-dof_view"));

  CHKERRQ(DMGetNumFields(dm, &Nf));
  for (f = 0; f < Nf; ++f) CHKERRQ(PetscFEDestroy(&fe[f]));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD, &dm));
  CHKERRQ(TestLocalDofOrder(dm));
  CHKERRQ(DMDestroy(&dm));
  ierr = PetscFinalize();
  return ierr;
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
