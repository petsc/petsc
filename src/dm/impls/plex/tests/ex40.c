static const char help[] = "Tests for regular refinement";

/* TODO
  - Add in simplex-to-hex tests
*/

#include <petscdmplex.h>
#include <petscsf.h>

#include <petsc/private/dmpleximpl.h>

typedef struct {
  DMPolytopeType refCell; /* Use the reference cell */
  PetscInt       dim;     /* The topological dimension */
  PetscBool      simplex; /* Flag for simplices */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->refCell = DM_POLYTOPE_UNKNOWN;
  options->dim     = 2;
  options->simplex = PETSC_TRUE;

  ierr = PetscOptionsBegin(comm, "", "Parallel Mesh Orientation Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological dimension", "ex40.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-ref_cell", "Use the reference cell", "ex40.c", DMPolytopeTypes, (PetscEnum) options->refCell, (PetscEnum *) &options->refCell, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Flag for simplices", "ex40.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *ctx, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ctx->refCell != DM_POLYTOPE_UNKNOWN) {
    ierr = DMPlexCreateReferenceCellByType(comm, ctx->refCell, dm);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateBoxMesh(comm, ctx->dim, ctx->simplex, NULL, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         ctx;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &ctx);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &ctx, &dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  test:
    suffix: ref_tri
    args: -ref_cell triangle -dm_refine 2 -dm_plex_check_all

  test:
    suffix: box_tri
    requires: triangle
    nsize: {{1 3 5}}
    args: -dm_plex_box_faces 3,3 -dm_refine 2 -dm_plex_check_all

  test:
    suffix: ref_quad
    args: -ref_cell quadrilateral -dm_refine 2 -dm_plex_check_all

  test:
    suffix: box_quad
    nsize: {{1 3 5}}
    args: -dm_plex_box_faces 3,3 -simplex 0 -dm_refine 2 -dm_plex_check_all

  test:
    suffix: ref_tet
    args: -ref_cell tetrahedron -dm_refine 2 -dm_plex_check_all

  test:
    suffix: box_tet
    requires: ctetgen
    nsize: {{1 3 5}}
    args: -dim 3 -dm_plex_box_faces 3,3,3 -dm_refine 2 -dm_plex_check_all

  test:
    suffix: ref_hex
    args: -ref_cell hexahedron -simplex 0 -dm_refine 2 -dm_plex_check_all

  test:
    suffix: box_hex
    nsize: {{1 3 5}}
    args: -dim 3 -dm_plex_box_faces 3,3,3 -simplex 0 -dm_refine 2 -dm_plex_check_all

  test:
    suffix: ref_trip
    args: -ref_cell triangular_prism -dm_refine 2 -dm_plex_check_all

  test:
    suffix: ref_tquad
    args: -ref_cell tensor_quad -dm_refine 2 -dm_plex_check_all

  test:
    suffix: ref_ttrip
    args: -ref_cell tensor_triangular_prism -dm_refine 2 -dm_plex_check_all

  test:
    suffix: ref_tquadp
    args: -ref_cell tensor_quadrilateral_prism -dm_refine 2 -dm_plex_check_all

  test:
    suffix: ref_tri_tobox
    args: -ref_cell triangle -dm_plex_cell_refiner tobox -dm_refine 2 -dm_plex_check_all

  test:
    suffix: box_tri_tobox
    requires: triangle
    nsize: {{1 3 5}}
    args: -dm_plex_box_faces 3,3 -dm_plex_cell_refiner tobox -dm_refine 2 -dm_plex_check_all

  test:
    suffix: ref_tet_tobox
    args: -ref_cell tetrahedron -dm_plex_cell_refiner tobox -dm_refine 2 -dm_plex_check_all

  test:
    suffix: box_tet_tobox
    requires: ctetgen
    nsize: {{1 3 5}}
    args: -dim 3 -dm_plex_box_faces 3,3,3 -dm_plex_cell_refiner tobox -dm_refine 2 -dm_plex_check_all

  test:
    suffix: ref_trip_tobox
    args: -ref_cell triangular_prism -dm_plex_cell_refiner tobox -dm_refine 2 -dm_plex_check_all

  test:
    suffix: ref_ttrip_tobox
    args: -ref_cell tensor_triangular_prism -dm_plex_cell_refiner tobox -dm_refine 2 -dm_plex_check_all

TEST*/
