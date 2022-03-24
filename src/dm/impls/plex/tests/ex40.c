static const char help[] = "Tests for Plex transforms, including regular refinement";

#include <petscdmplex.h>
#include <petscsf.h>

#include <petsc/private/dmpleximpl.h>

static PetscErrorCode LabelPoints(DM dm)
{
  DMLabel        label;
  PetscInt       pStart, pEnd, p;
  PetscBool      flg = PETSC_FALSE;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsGetBool(NULL, NULL, "-label_mesh", &flg, NULL));
  if (!flg) PetscFunctionReturn(0);
  CHKERRQ(DMCreateLabel(dm, "test"));
  CHKERRQ(DMGetLabel(dm, "test", &label));
  CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {
    CHKERRQ(DMLabelSetValue(label, p, p));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscFunctionBegin;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(LabelPoints(*dm));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) *dm, "post_label_"));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) *dm, NULL));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;

  CHKERRQ(PetscInitialize(&argc, &argv, NULL, help));
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD, &dm));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST
  test:
    suffix: ref_seg
    args: -dm_plex_reference_cell_domain -dm_plex_cell segment -dm_refine 1 -dm_plex_check_all

  test:
    suffix: ref_tri
    args: -dm_plex_reference_cell_domain -dm_plex_cell triangle -dm_refine 2 -dm_plex_check_all

  test:
    suffix: box_tri
    requires: triangle
    nsize: {{1 3 5}}
    args: -dm_plex_box_faces 3,3 -dm_refine 2 -dm_plex_check_all

  test:
    suffix: ref_quad
    args: -dm_plex_reference_cell_domain -dm_plex_cell quadrilateral -dm_refine 2 -dm_plex_check_all

  test:
    suffix: box_quad
    nsize: {{1 3 5}}
    args: -dm_plex_box_faces 3,3 -dm_plex_simplex 0 -dm_refine 2 -dm_plex_check_all

  test:
    suffix: ref_tet
    args: -dm_plex_reference_cell_domain -dm_plex_cell tetrahedron -dm_refine 2 -dm_plex_check_all

  test:
    suffix: box_tet
    requires: ctetgen
    nsize: {{1 3 5}}
    args: -dm_plex_dim 3 -dm_plex_box_faces 3,3,3 -dm_refine 2 -dm_plex_check_all

  test:
    suffix: ref_hex
    args: -dm_plex_reference_cell_domain -dm_plex_cell hexahedron -dm_refine 2 -dm_plex_check_all

  test:
    suffix: box_hex
    nsize: {{1 3 5}}
    args: -dm_plex_dim 3 -dm_plex_box_faces 3,3,3 -dm_plex_simplex 0 -dm_refine 2 -dm_plex_check_all

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
    suffix: ref_pyramid
    args: -dm_plex_reference_cell_domain -dm_plex_cell pyramid -dm_refine 2 -dm_plex_check_all

  testset:
    args: -dm_coord_space 0 -dm_plex_transform_type refine_tobox -dm_plex_check_all

    test:
      suffix: ref_tri_tobox
      args: -dm_plex_reference_cell_domain -dm_plex_cell triangle -dm_refine 2

    test:
      suffix: box_tri_tobox
      requires: triangle
      nsize: {{1 3 5}}
      args: -dm_plex_box_faces 3,3 -dm_refine 2

    test:
      suffix: ref_tet_tobox
      args: -dm_plex_reference_cell_domain -dm_plex_cell tetrahedron -dm_refine 2

    test:
      suffix: box_tet_tobox
      requires: ctetgen
      nsize: {{1 3 5}}
      args: -dm_plex_dim 3 -dm_plex_box_faces 3,3,3 -dm_refine 2

    test:
      suffix: ref_trip_tobox
      args: -dm_plex_reference_cell_domain -dm_plex_cell triangular_prism -dm_refine 2

    test:
      suffix: ref_ttrip_tobox
      args: -dm_plex_reference_cell_domain -dm_plex_cell tensor_triangular_prism -dm_refine 2

    test:
      suffix: ref_tquadp_tobox
      args: -dm_plex_reference_cell_domain -dm_plex_cell tensor_quadrilateral_prism -dm_refine 2

  testset:
    args: -dm_coord_space 0 -label_mesh -post_label_dm_extrude 2 -post_label_dm_plex_check_all -dm_view ::ascii_info_detail

    test:
      suffix: extrude_quad
      args: -dm_plex_simplex 0

TEST*/
