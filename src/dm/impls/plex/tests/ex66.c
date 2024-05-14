static const char help[] = "Test for non-manifold interpolation";

#include <petscdmplex.h>

/*
Test 0:

       3-------------7
      /|            /|
     / |           / |
    /  |          /  |
   1-------------5   |
   |   |         |   |
   |   |         |   |
   |   |         |   |
   |   |         |   |
   z   4---------|---8
   ^  /          |  /
   | y           | /
   |/            |/
   2--->-x-------6-------------9

Test 1:

       3-------------7
      /|            /|
     / |           / |
    /  |          /  |
   1-------------5   |
   |   |         |   |
   |   |         |   |
   |   |         |   |
   |   |         |   |
   z   4---------|---8
   ^  /          |  / \
   | y           | /   \
   |/            |/     \
   2--->-x-------6-------9
*/
int main(int argc, char **argv)
{
  DM       dm, idm;
  DMLabel  ctLabel;
  PetscInt testNum = 0;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Non-Manifold Options", "ex66");
  PetscCall(PetscOptionsInt("-test_num", "Test number", "", testNum, &testNum, NULL));
  PetscOptionsEnd();

  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(PetscObjectSetName((PetscObject)dm, "cubeline-fromdag"));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetDimension(dm, 3));

  switch (testNum) {
  case 0: {
    // 9 vertices
    // 1 edge
    // 0 faces
    // 1 volume
    PetscInt num_points[4] = {9, 1, 0, 1};

    // point 0 = hexahedron (defined by 8 vertices)
    // points 1-9 = vertices
    // point 10 = edged (defined by 2 vertices)
    PetscInt cone_size[11] = {8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2};

    // hexahedron defined by points
    PetscInt    cones[11]             = {3, 4, 2, 1, 7, 5, 6, 8, 6, 9};
    PetscInt    cone_orientations[11] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    PetscScalar vertex_coords[3 * 9]  = {0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 2, 0, 0};

    PetscCall(DMPlexCreateFromDAG(dm, 3, num_points, cone_size, cones, cone_orientations, vertex_coords));
  } break;
  case 1: {
    // 9 vertices
    // 0 edges
    // 1 face
    // 1 volume
    PetscInt num_points[4] = {9, 0, 1, 1};

    // point 0 = hexahedron (defined by 8 vertices)
    // points 1-9 = vertices
    // point 10 = triangle (defined by 3 vertices)
    PetscInt cone_size[11] = {8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3};

    // hexahedron defined by 8 points (point 0)
    PetscInt    cones[11]             = {3, 4, 2, 1, 7, 5, 6, 8, 6, 9, 8};
    PetscInt    cone_orientations[11] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    PetscScalar vertex_coords[3 * 9]  = {0, 0, 1,  // point 1
                                         0, 0, 0,  // point 2
                                         0, 1, 1,  // point 3
                                         0, 1, 0,  // point 4
                                         1, 0, 1,  // point 5
                                         1, 0, 0,  // point 6
                                         1, 1, 1,  // point 7
                                         1, 1, 0,  // point 8
                                         2, 0, 0}; // point 9

    PetscCall(DMPlexCreateFromDAG(dm, 3, num_points, cone_size, cones, cone_orientations, vertex_coords));
  } break;
  default:
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Invalid test number %" PetscInt_FMT, testNum);
  }
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  // TODO: make it work with a DM made from a msh file
  // PetscCall(DMPlexCreateGmshFromFile(PETSC_COMM_WORLD, "cube-line.msh", PETSC_FALSE, &dm));
  // PetscCall(PetscObjectSetName((PetscObject)dm, "msh"));

  // Must set cell types
  PetscCall(DMPlexGetCellTypeLabel(dm, &ctLabel));
  switch (testNum) {
  case 0:
    PetscCall(DMLabelSetValue(ctLabel, 0, DM_POLYTOPE_HEXAHEDRON));
    PetscCall(DMLabelSetValue(ctLabel, 1, DM_POLYTOPE_POINT));
    PetscCall(DMLabelSetValue(ctLabel, 2, DM_POLYTOPE_POINT));
    PetscCall(DMLabelSetValue(ctLabel, 3, DM_POLYTOPE_POINT));
    PetscCall(DMLabelSetValue(ctLabel, 4, DM_POLYTOPE_POINT));
    PetscCall(DMLabelSetValue(ctLabel, 5, DM_POLYTOPE_POINT));
    PetscCall(DMLabelSetValue(ctLabel, 6, DM_POLYTOPE_POINT));
    PetscCall(DMLabelSetValue(ctLabel, 7, DM_POLYTOPE_POINT));
    PetscCall(DMLabelSetValue(ctLabel, 8, DM_POLYTOPE_POINT));
    PetscCall(DMLabelSetValue(ctLabel, 9, DM_POLYTOPE_POINT));
    PetscCall(DMLabelSetValue(ctLabel, 10, DM_POLYTOPE_SEGMENT));
    break;
  case 1:
    PetscCall(DMLabelSetValue(ctLabel, 0, DM_POLYTOPE_HEXAHEDRON));
    PetscCall(DMLabelSetValue(ctLabel, 1, DM_POLYTOPE_POINT));
    PetscCall(DMLabelSetValue(ctLabel, 2, DM_POLYTOPE_POINT));
    PetscCall(DMLabelSetValue(ctLabel, 3, DM_POLYTOPE_POINT));
    PetscCall(DMLabelSetValue(ctLabel, 4, DM_POLYTOPE_POINT));
    PetscCall(DMLabelSetValue(ctLabel, 5, DM_POLYTOPE_POINT));
    PetscCall(DMLabelSetValue(ctLabel, 6, DM_POLYTOPE_POINT));
    PetscCall(DMLabelSetValue(ctLabel, 7, DM_POLYTOPE_POINT));
    PetscCall(DMLabelSetValue(ctLabel, 8, DM_POLYTOPE_POINT));
    PetscCall(DMLabelSetValue(ctLabel, 9, DM_POLYTOPE_POINT));
    PetscCall(DMLabelSetValue(ctLabel, 10, DM_POLYTOPE_TRIANGLE));
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Invalid test number %" PetscInt_FMT, testNum);
  }

  // interpolate (make sure to use -interp_dm_plex_stratify_celltype)
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)dm, "interp_"));
  PetscCall(DMPlexInterpolate(dm, &idm));
  PetscCall(DMDestroy(&dm));
  dm = idm;

  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    args: -interp_dm_plex_stratify_celltype -dm_view ::ascii_info_detail -interp_dm_view ::ascii_info_detail

    test:
      suffix: 0
    test:
      suffix: 1
      args: -test_num 1

TEST*/
