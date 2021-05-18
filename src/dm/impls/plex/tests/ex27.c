static char help[] = "Test section ordering for FEM discretizations\n\n";

#include <petscdmplex.h>
#include <petscds.h>

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestLocalDofOrder(DM dm)
{
  PetscFE        fe[3];
  PetscSection   s;
  PetscBool      simplex;
  PetscInt       dim, Nf, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexIsSimplex(dm, &simplex);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PETSC_COMM_SELF, dim, dim, simplex, "field0_", -1, &fe[0]);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PETSC_COMM_SELF, dim, 1,   simplex, "field1_", -1, &fe[1]);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PETSC_COMM_SELF, dim, 1,   simplex, "field2_", -1, &fe[2]);CHKERRQ(ierr);

  ierr = DMSetField(dm, 0, NULL, (PetscObject) fe[0]);CHKERRQ(ierr);
  ierr = DMSetField(dm, 1, NULL, (PetscObject) fe[1]);CHKERRQ(ierr);
  ierr = DMSetField(dm, 2, NULL, (PetscObject) fe[2]);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &s);CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject) s, NULL, "-dof_view");CHKERRQ(ierr);

  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {ierr = PetscFEDestroy(&fe[f]);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = CreateMesh(PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
  ierr = TestLocalDofOrder(dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
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
