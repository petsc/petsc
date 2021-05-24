static const char help[] = "Tests for regular refinement";

#include <petscdmplex.h>
#include <petscsf.h>

#include <petsc/private/dmpleximpl.h>

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

int main(int argc, char **argv)
{
  DM             dm;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;
  ierr = CreateMesh(PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  test:
    suffix: ref_tri
    args: -dm_plex_reference_cell_domain -dm_plex_cell triangle -dm_refine 2 -dm_plex_check_all

  test:
    suffix: box_tri
    requires: triangle
    nsize: {{1 3 5}}
    args: -dm_distribute -dm_plex_box_faces 3,3 -dm_refine 2 -dm_plex_check_all

  test:
    suffix: ref_quad
    args: -dm_plex_reference_cell_domain -dm_plex_cell quadrilateral -dm_refine 2 -dm_plex_check_all

  test:
    suffix: box_quad
    nsize: {{1 3 5}}
    args: -dm_distribute -dm_plex_box_faces 3,3 -dm_plex_simplex 0 -dm_refine 2 -dm_plex_check_all

  test:
    suffix: ref_tet
    args: -dm_plex_reference_cell_domain -dm_plex_cell tetrahedron -dm_refine 2 -dm_plex_check_all

  test:
    suffix: box_tet
    requires: ctetgen
    nsize: {{1 3 5}}
    args: -dm_distribute -dm_plex_dim 3 -dm_plex_box_faces 3,3,3 -dm_refine 2 -dm_plex_check_all

  test:
    suffix: ref_hex
    args: -dm_plex_reference_cell_domain -dm_plex_cell hexahedron -dm_refine 2 -dm_plex_check_all

  test:
    suffix: box_hex
    nsize: {{1 3 5}}
    args: -dm_distribute -dm_plex_dim 3 -dm_plex_box_faces 3,3,3 -dm_plex_simplex 0 -dm_refine 2 -dm_plex_check_all

  test:
    suffix: ref_trip
    args: -dm_plex_reference_cell_domain -dm_plex_cell triangular_prism -dm_refine 2 -dm_plex_check_all

  test:
    suffix: ref_tquad
    args: -dm_plex_reference_cell_domain -dm_plex_cell tensor_quad -dm_refine 2 -dm_plex_check_all

  test:
    suffix: ref_ttrip
    args: -dm_plex_reference_cell_domain -dm_plex_cell tensor_triangular_prism -dm_refine 2 -dm_plex_check_all

  test:
    suffix: ref_tquadp
    args: -dm_plex_reference_cell_domain -dm_plex_cell tensor_quadrilateral_prism -dm_refine 2 -dm_plex_check_all

  test:
    suffix: ref_tri_tobox
    args: -dm_plex_reference_cell_domain -dm_plex_cell triangle -dm_plex_cell_refiner tobox -dm_coord_space 0 -dm_refine 2 -dm_plex_check_all

  test:
    suffix: box_tri_tobox
    requires: triangle
    nsize: {{1 3 5}}
    args: -dm_distribute -dm_plex_box_faces 3,3 -dm_plex_cell_refiner tobox -dm_coord_space 0 -dm_refine 2 -dm_plex_check_all

  test:
    suffix: ref_tet_tobox
    args: -dm_plex_reference_cell_domain -dm_plex_cell tetrahedron -dm_plex_cell_refiner tobox -dm_coord_space 0 -dm_refine 2 -dm_plex_check_all

  test:
    suffix: box_tet_tobox
    requires: ctetgen
    nsize: {{1 3 5}}
    args: -dm_distribute -dm_plex_dim 3 -dm_plex_box_faces 3,3,3 -dm_plex_cell_refiner tobox -dm_coord_space 0 -dm_refine 2 -dm_plex_check_all

  test:
    suffix: ref_trip_tobox
    args: -dm_plex_reference_cell_domain -dm_plex_cell triangular_prism -dm_plex_cell_refiner tobox -dm_refine 2 -dm_plex_check_all

  test:
    suffix: ref_ttrip_tobox
    args: -dm_plex_reference_cell_domain -dm_plex_cell tensor_triangular_prism -dm_plex_cell_refiner tobox -dm_refine 2 -dm_plex_check_all

TEST*/
