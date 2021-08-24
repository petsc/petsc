static char help[] = "Tests for mesh interpolation\n\n";

#include <petscdmplex.h>

/* List of test meshes

Triangle
--------
Test 0:
Two triangles sharing a face

        4
      / | \
     /  |  \
    /   |   \
   2  0 | 1  5
    \   |   /
     \  |  /
      \ | /
        3

should become

        4
      / | \
     8  |  9
    /   |   \
   2  0 7 1  5
    \   |   /
     6  |  10
      \ | /
        3

Tetrahedron
-----------
Test 0:
Two tets sharing a face

 cell   5 _______    cell
 0    / | \      \       1
     /  |  \      \
    /   |   \      \
   2----|----4-----6
    \   |   /      /
     \  |  /     /
      \ | /      /
        3-------

should become

 cell   5 _______    cell
 0    / | \      \       1
     /  |  \      \
   17   |   18 13  22
   / 8 19 10 \      \
  /     |     \      \
 2---14-|------4--20--6
  \     |     /      /
   \ 9  | 7  /      /
   16   |   15 11  21
     \  |  /      /
      \ | /      /
        3-------

Quadrilateral
-------------
Test 0:
Two quads sharing a face

   5-------4-------7
   |       |       |
   |   0   |   1   |
   |       |       |
   2-------3-------6

should become

   5--10---4--14---7
   |       |       |
  11   0   9   1  13
   |       |       |
   2---8---3--12---6

Test 1:
A quad and a triangle sharing a face

   5-------4
   |       | \
   |   0   |  \
   |       | 1 \
   2-------3----6

should become

   5---9---4
   |       | \
  10   0   8  12
   |       | 1 \
   2---7---3-11-6

Hexahedron
----------
Test 0:
Two hexes sharing a face

cell   9-------------8-------------13 cell
0     /|            /|            /|     1
     / |   15      / |   21      / |
    /  |          /  |          /  |
   6-------------7-------------12  |
   |   |     18  |   |     24  |   |
   |   |         |   |         |   |
   |19 |         |17 |         |23 |
   |   |  16     |   |   22    |   |
   |   5---------|---4---------|---11
   |  /          |  /          |  /
   | /   14      | /    20     | /
   |/            |/            |/
   2-------------3-------------10

should become

cell   9-----31------8-----42------13 cell
0     /|            /|            /|     1
    32 |   15      30|   21      41|
    /  |          /  |          /  |
   6-----29------7-----40------12  |
   |   |     17  |   |     23  |   |
   |  35         |  36         |   44
   |19 |         |18 |         |24 |
  34   |  16    33   |   22   43   |
   |   5-----26--|---4-----37--|---11
   |  /          |  /          |  /
   | 25   14     | 27    20    | 38
   |/            |/            |/
   2-----28------3-----39------10

*/

typedef struct {
  PetscInt  testNum;      /* Indicates the mesh to create */
  PetscInt  dim;          /* The topological mesh dimension */
  PetscBool simplex;      /* Use simplices or hexes */
  PetscBool useGenerator; /* Construct mesh with a mesh generator */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->testNum      = 0;
  options->dim          = 2;
  options->simplex      = PETSC_TRUE;
  options->useGenerator = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Meshing Interpolation Test Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-testnum", "The mesh to create", "ex7.c", options->testNum, &options->testNum, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-dim", "The topological mesh dimension", "ex7.c", options->dim, &options->dim, NULL,1,3);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Use simplices if true, otherwise hexes", "ex7.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-use_generator", "Use a mesh generator to build the mesh", "ex7.c", options->useGenerator, &options->useGenerator, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CreateSimplex_2D(MPI_Comm comm, DM dm)
{
  PetscInt       depth = 1, testNum  = 0, p;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  if (!rank) {
    switch (testNum) {
    case 0:
    {
      PetscInt    numPoints[2]        = {4, 2};
      PetscInt    coneSize[6]         = {3, 3, 0, 0, 0, 0};
      PetscInt    cones[6]            = {2, 3, 4,  5, 4, 3};
      PetscInt    coneOrientations[6] = {0, 0, 0,  0, 0, 0};
      PetscScalar vertexCoords[8]     = {-0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5};
      PetscInt    markerPoints[8]     = {2, 1, 3, 1, 4, 1, 5, 1};

      ierr = DMPlexCreateFromDAG(dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
      for (p = 0; p < 4; ++p) {
        ierr = DMSetLabelValue(dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);
      }
    }
    break;
    default:
      SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %d", testNum);
    }
  } else {
    PetscInt numPoints[2] = {0, 0};

    ierr = DMPlexCreateFromDAG(dm, depth, numPoints, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = DMCreateLabel(dm, "marker");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CreateSimplex_3D(MPI_Comm comm, DM dm)
{
  PetscInt       depth = 1, testNum  = 0, p;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  if (!rank) {
    switch (testNum) {
    case 0:
    {
      PetscInt    numPoints[2]        = {5, 2};
      PetscInt    coneSize[23]        = {4, 4, 0, 0, 0, 0, 0};
      PetscInt    cones[8]            = {2, 4, 3, 5,  3, 4, 6, 5};
      PetscInt    coneOrientations[8] = {0, 0, 0, 0,  0, 0, 0, 0};
      PetscScalar vertexCoords[15]    = {0.0, 0.0, -0.5,  0.0, -0.5, 0.0,  1.0, 0.0, 0.0,  0.0, 0.5, 0.0,  0.0, 0.0, 0.5};
      PetscInt    markerPoints[8]     = {2, 1, 3, 1, 4, 1, 5, 1};

      ierr = DMPlexCreateFromDAG(dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
      for (p = 0; p < 4; ++p) {
        ierr = DMSetLabelValue(dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);
      }
    }
    break;
    default:
      SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %d", testNum);
    }
  } else {
    PetscInt numPoints[2] = {0, 0};

    ierr = DMPlexCreateFromDAG(dm, depth, numPoints, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = DMCreateLabel(dm, "marker");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CreateQuad_2D(MPI_Comm comm, PetscInt testNum, DM dm)
{
  PetscInt       depth = 1, p;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  if (!rank) {
    switch (testNum) {
    case 0:
    {
      PetscInt    numPoints[2]        = {6, 2};
      PetscInt    coneSize[8]         = {4, 4, 0, 0, 0, 0, 0, 0};
      PetscInt    cones[8]            = {2, 3, 4, 5,  3, 6, 7, 4};
      PetscInt    coneOrientations[8] = {0, 0, 0, 0,  0, 0, 0, 0};
      PetscScalar vertexCoords[12]    = {-0.5, 0.0, 0.0, 0.0, 0.0, 1.0, -0.5, 1.0, 0.5, 0.0, 0.5, 1.0};
      PetscInt    markerPoints[12]    = {2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7, 1};

      ierr = DMPlexCreateFromDAG(dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
      for (p = 0; p < 6; ++p) {
        ierr = DMSetLabelValue(dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);
      }
    }
    break;
    case 1:
    {
      PetscInt    numPoints[2]        = {5, 2};
      PetscInt    coneSize[7]         = {4, 3, 0, 0, 0, 0, 0};
      PetscInt    cones[7]            = {2, 3, 4, 5,  3, 6, 4};
      PetscInt    coneOrientations[7] = {0, 0, 0, 0,  0, 0, 0};
      PetscScalar vertexCoords[10]    = {-0.5, 0.0, 0.0, 0.0, 0.0, 1.0, -0.5, 1.0, 0.5, 0.0};
      PetscInt    markerPoints[10]    = {2, 1, 3, 1, 4, 1, 5, 1, 6, 1};

      ierr = DMPlexCreateFromDAG(dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
      for (p = 0; p < 5; ++p) {
        ierr = DMSetLabelValue(dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);
      }
    }
    break;
    default:
      SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %d", testNum);
    }
  } else {
    PetscInt numPoints[2] = {0, 0};

    ierr = DMPlexCreateFromDAG(dm, depth, numPoints, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = DMCreateLabel(dm, "marker");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CreateHex_3D(MPI_Comm comm, DM dm)
{
  PetscInt       depth = 1, testNum  = 0, p;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  if (!rank) {
    switch (testNum) {
    case 0:
    {
      PetscInt    numPoints[2]         = {12, 2};
      PetscInt    coneSize[14]         = {8, 8, 0,0,0,0,0,0,0,0,0,0,0,0};
      PetscInt    cones[16]            = {2,5,4,3,6,7,8,9,  3,4,11,10,7,12,13,8};
      PetscInt    coneOrientations[16] = {0,0,0,0,0,0,0,0,  0,0, 0, 0,0, 0, 0,0};
      PetscScalar vertexCoords[36]     = {-0.5,0.0,0.0, 0.0,0.0,0.0, 0.0,1.0,0.0, -0.5,1.0,0.0,
                                          -0.5,0.0,1.0, 0.0,0.0,1.0, 0.0,1.0,1.0, -0.5,1.0,1.0,
                                           0.5,0.0,0.0, 0.5,1.0,0.0, 0.5,0.0,1.0,  0.5,1.0,1.0};
      PetscInt    markerPoints[16]     = {2,1,3,1,4,1,5,1,6,1,7,1,8,1,9,1};

      ierr = DMPlexCreateFromDAG(dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
      for (p = 0; p < 8; ++p) {
        ierr = DMSetLabelValue(dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);
      }
    }
    break;
    default:
      SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %d", testNum);
    }
  } else {
    PetscInt numPoints[2] = {0, 0};

    ierr = DMPlexCreateFromDAG(dm, depth, numPoints, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = DMCreateLabel(dm, "marker");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(MPI_Comm comm, PetscInt testNum, AppCtx *user, DM *dm)
{
  PetscBool      useGenerator = user->useGenerator;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  if (!useGenerator) {
    PetscInt  dim     = user->dim;
    PetscBool simplex = user->simplex;

    ierr = DMSetDimension(*dm, dim);CHKERRQ(ierr);
    switch (dim) {
    case 2:
      if (simplex) {
        ierr = CreateSimplex_2D(comm, *dm);CHKERRQ(ierr);
      } else {
        ierr = CreateQuad_2D(comm, testNum, *dm);CHKERRQ(ierr);
      }
      break;
    case 3:
      if (simplex) {
        ierr = CreateSimplex_3D(comm, *dm);CHKERRQ(ierr);
      } else {
        ierr = CreateHex_3D(comm, *dm);CHKERRQ(ierr);
      }
      break;
    default:
      SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Cannot make meshes for dimension %d", dim);
    }
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dm, "Interpolated Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;                 /* user-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, user.testNum, &user, &dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  testset:
    args: -dm_plex_interpolate_pre -dm_plex_check_all

    # Two cell test meshes 0-7
    test:
      suffix: 0
      args: -dim 2 -dm_plex_interpolate -dm_view ascii::ascii_info_detail
    test:
      suffix: 1
      nsize: 2
      args: -dm_distribute -petscpartitioner_type simple -dim 2 -dm_view ascii::ascii_info_detail
    test:
      suffix: 2
      args: -dim 2 -simplex 0 -dm_view ascii::ascii_info_detail
    test:
      suffix: 3
      nsize: 2
      args: -dm_distribute -petscpartitioner_type simple -dim 2 -simplex 0 -dm_view ascii::ascii_info_detail
    test:
      suffix: 4
      args: -dim 3 -dm_view ascii::ascii_info_detail
    test:
      suffix: 5
      nsize: 2
      args: -dm_distribute -petscpartitioner_type simple -dim 3 -dm_view ascii::ascii_info_detail
    test:
      suffix: 6
      args: -dim 3 -simplex 0 -dm_view ascii::ascii_info_detail
    test:
      suffix: 7
      nsize: 2
      args: -dm_distribute -petscpartitioner_type simple -dim 3 -simplex 0 -dm_view ascii::ascii_info_detail
    # 2D Hybrid Mesh 8
    test:
      suffix: 8
      args: -dim 2 -simplex 0 -testnum 1 -dm_view ascii::ascii_info_detail

  testset:
    args: -dm_plex_check_all

    # Reference cells
    test:
      suffix: 12
      args: -use_generator -dm_plex_reference_cell_domain -dm_plex_cell pyramid -dm_plex_check_all
    # TetGen meshes 9-10
    test:
      suffix: 9
      requires: triangle
      args: -use_generator -dm_view ascii::ascii_info_detail -dm_coord_space 0
    test:
      suffix: 10
      requires: ctetgen
      args: -use_generator -dm_plex_dim 3 -dm_view ascii::ascii_info_detail -dm_coord_space 0
    # Cubit meshes 11
    test:
      suffix: 11
      requires: exodusii
      args: -use_generator -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo -dm_view ascii::ascii_info_detail -dm_coord_space 0

TEST*/
