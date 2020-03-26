const char help[] = "Construct and set a Lagrange dual space from options, then view it to\n"
                    "understand the effects of different parameters.";

#include <petscfe.h>
#include <petscdmplex.h>

int main(int argc, char **argv)
{
  PetscInt       dim;
  PetscBool      tensorCell;
  DM             K;
  PetscDualSpace dsp;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;
  dim = 2;
  tensorCell = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Options for PETSCDUALSPACELAGRANGE test","none");CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-dim", "The spatial dimension","ex1.c",dim,&dim,NULL,0,3);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-tensor", "Whether the cell is a tensor product cell or a simplex","ex1.c",tensorCell,&tensorCell,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = PetscDualSpaceCreate(PETSC_COMM_WORLD, &dsp);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)dsp, "Lagrange dual space");CHKERRQ(ierr);
  ierr = PetscDualSpaceSetType(dsp, PETSCDUALSPACELAGRANGE);CHKERRQ(ierr);
  /* While Lagrange nodes don't require the existence of a reference cell to
   * be refined, when we construct finite element dual spaces we have to be
   * careful about what kind of continuity is maintained when cells are glued
   * together to make a mesh.  The PetscDualSpace object is responsible for
   * conveying continuity requirements to a finite element assembly routines,
   * so a PetscDualSpace needs a reference element: a single element mesh,
   * whose boundary points are the interstitial points in a mesh */
  ierr = DMPlexCreateReferenceCell(PETSC_COMM_WORLD, dim, (PetscBool) !tensorCell, &K);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetDM(dsp, K);CHKERRQ(ierr);
  /* This gives us the opportunity to change the parameters of the dual space
   * from the command line, as we do in the tests below.  When
   * PetscDualSpaceSetFromOptions() is called, it also enables other optional
   * behavior (see the next step) */
  ierr = PetscDualSpaceSetFromOptions(dsp);CHKERRQ(ierr);
  /* This step parses the parameters of the dual space into
   * sets of functionals that are assigned to each of the mesh points in K.
   *
   * The functionals can be accessed individually by
   * PetscDualSpaceGetFunctional(), or more efficiently all at once by
   * PetscDualSpaceGetAllData(), which returns a set of quadrature points
   * at which to evaluate a function, and a matrix that takes those
   * evaluations and turns them into the evaluation of the dual space's
   * functionals on the function.
   *
   * (TODO: tutorial for PetscDualSpaceGetAllData() and
   * PetscDualSpaceGetInteriorData().)
   *
   * Because we called PetscDualSpaceSetFromOptions(), we have the opportunity
   * to inspect the results of PetscDualSpaceSetUp() from the command line
   * with "-petscdualspace_view", followed by an optional description of how
   * we would like to see the dual space (see examples in the tests below).
   * */
  ierr = PetscDualSpaceSetUp(dsp);CHKERRQ(ierr);
  ierr = DMDestroy(&K);CHKERRQ(ierr);
  ierr = PetscDualSpaceDestroy(&dsp);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  # quadratic nodes on the triangle
  test:
    suffix: 0
    args: -dim 2 -tensor 0 -petscdualspace_order 2 -petscdualspace_view ascii::ascii_info_detail

  # linear nodes on the quadrilateral
  test:
    suffix: 1
    args: -dim 2 -tensor 1 -petscdualspace_order 1 -petscdualspace_view ascii::ascii_info_detail

  # lowest order Raviart-Thomas / Nedelec edge nodes on the hexahedron
  test:
    suffix: 2
    args: -dim 3 -tensor 1 -petscdualspace_order 1 -petscdualspace_components 3 -petscdualspace_form_degree 1 -petscdualspace_lagrange_trimmed 1 -petscdualspace_lagrange_tensor 1 -petscdualspace_view ascii::ascii_info_detail

  # first order Nedelec second type face nodes on the tetrahedron
  test:
    suffix: 3
    args: -dim 3 -tensor 0 -petscdualspace_order 1 -petscdualspace_components 3 -petscdualspace_form_degree -2 -petscdualspace_view ascii::ascii_info_detail

  ## Comparing different node types

  test:
    suffix: 4
    args: -dim 2 -tensor 0 -petscdualspace_order 3 -petscdualspace_lagrange_continuity 0 -petscdualspace_lagrange_node_type equispaced -petscdualspace_lagrange_node_endpoints 0 -petscdualspace_view ascii::ascii_info_detail

  test:
    suffix: 5
    args: -dim 2 -tensor 0 -petscdualspace_order 3 -petscdualspace_lagrange_continuity 0 -petscdualspace_lagrange_node_type equispaced -petscdualspace_lagrange_node_endpoints 1 -petscdualspace_view ascii::ascii_info_detail

  test:
    suffix: 6
    args: -dim 2 -tensor 0 -petscdualspace_order 3 -petscdualspace_lagrange_continuity 0 -petscdualspace_lagrange_node_type gaussjacobi -petscdualspace_lagrange_node_endpoints 0 -petscdualspace_view ascii::ascii_info_detail

  test:
    suffix: 7
    args: -dim 2 -tensor 0 -petscdualspace_order 3 -petscdualspace_lagrange_continuity 0 -petscdualspace_lagrange_node_type gaussjacobi -petscdualspace_lagrange_node_endpoints 1 -petscdualspace_view ascii::ascii_info_detail

TEST*/
