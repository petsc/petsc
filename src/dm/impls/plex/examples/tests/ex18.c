static char help[] = "Tests for parallel mesh loading\n\n";

#include <petscdmplex.h>

/* List of test meshes

Triangle
--------
Test 0 (2 ranks):
Two triangles sharing a face

        2
      / | \
     /  |  \
    /   |   \
   0  0 | 1  3
    \   |   /
     \  |  /
      \ | /
        1

  vertex distribution:
    rank 0: 0 1
    rank 1: 2 3
  cell distribution:
    rank 0: 0
    rank 1: 1

Test 1 (3 ranks):
Four triangles partitioned across 3 ranks

   0 _______ 3
   | \     / |
   |  \ 1 /  |
   |   \ /   |
   | 0  2  2 |
   |   / \   |
   |  / 3 \  |
   | /     \ |
   1 ------- 4

  vertex distribution:
    rank 0: 0 1
    rank 1: 2 3
    rank 2: 4
  cell distribution:
    rank 0: 0
    rank 1: 1
    rank 2: 2 3

Tetrahedron
-----------
Test 0:
Two tets sharing a face

 cell   3 _______    cell
 0    / | \      \   1
     /  |  \      \
    /   |   \      \
   0----|----4-----2
    \   |   /      /
     \  |  /      /
      \ | /      /
        1-------
   y
   | x
   |/
   *----z

  vertex distribution:
    rank 0: 0 1
    rank 1: 2 3 4
  cell distribution:
    rank 0: 0
    rank 1: 1

Quadrilateral
-------------
Test 0 (2 ranks):
Two quads sharing a face

   3-------2-------5
   |       |       |
   |   0   |   1   |
   |       |       |
   0-------1-------4

  vertex distribution:
    rank 0: 0 1 2
    rank 1: 3 4 5
  cell distribution:
    rank 0: 0
    rank 1: 1

TODO Test 1:
A quad and a triangle sharing a face

   5-------4
   |       | \
   |   0   |  \
   |       | 1 \
   2-------3----6

Hexahedron
----------
Test 0 (2 ranks):
Two hexes sharing a face

cell   7-------------6-------------11 cell
0     /|            /|            /|     1
     / |   F1      / |   F7      / |
    /  |          /  |          /  |
   4-------------5-------------10  |
   |   |     F4  |   |     F10 |   |
   |   |         |   |         |   |
   |F5 |         |F3 |         |F9 |
   |   |  F2     |   |   F8    |   |
   |   3---------|---2---------|---9
   |  /          |  /          |  /
   | /   F0      | /    F6     | /
   |/            |/            |/
   0-------------1-------------8

  vertex distribution:
    rank 0: 0 1 2 3 4 5
    rank 1: 6 7 8 9 10 11
  cell distribution:
    rank 0: 0
    rank 1: 1

*/

typedef struct {
  DM        dm;
  PetscInt  debug;                        /* The debugging level */
  PetscInt  testNum;                      /* Indicates the mesh to create */
  PetscInt  dim;                          /* The topological mesh dimension */
  PetscBool cellSimplex;                  /* Use simplices or hexes */
  PetscBool interpolate;                  /* Interpolate mesh */
  PetscBool useGenerator;                 /* Construct mesh with a mesh generator */
  char      filename[PETSC_MAX_PATH_LEN]; /* Import mesh from file */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug        = 0;
  options->testNum      = 0;
  options->dim          = 2;
  options->cellSimplex  = PETSC_TRUE;
  options->interpolate  = PETSC_FALSE;
  options->useGenerator = PETSC_FALSE;
  options->filename[0]  = '\0';

  ierr = PetscOptionsBegin(comm, "", "Meshing Interpolation Test Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex18.c", options->debug, &options->debug, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-testnum", "The mesh to create", "ex18.c", options->testNum, &options->testNum, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex18.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-cell_simplex", "Use simplices if true, otherwise hexes", "ex18.c", options->cellSimplex, &options->cellSimplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Interpolate the mesh", "ex18.c", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-use_generator", "Use a mesh generator to build the mesh", "ex18.c", options->useGenerator, &options->useGenerator, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The mesh file", "ex18.c", options->filename, options->filename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

PetscErrorCode CreateSimplex_2D(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       testNum = user->testNum, p;
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  switch (testNum) {
  case 0:
    if (size != 2) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh %d only for 2 processes", testNum);
    switch (rank) {
      case 0:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 3;
        const int      cells[3]  = {0, 1, 2};
        PetscReal      coords[4] = {-0.5, 0.5, 0.0, 0.0};
        PetscInt       markerPoints[6] = {1, 1, 2, 1, 3, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, user->interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 3; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      case 1:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 3;
        const int      cells[3]  = {1, 3, 2};
        PetscReal      coords[4] = {0.0, 1.0, 0.5, 0.5};
        PetscInt       markerPoints[6] = {1, 1, 2, 1, 3, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, user->interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 3; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
    break;
  case 1:
    if (size != 3) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh %d only for 3 processes", testNum);
    switch (rank) {
      case 0:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 3;
        const int      cells[3]  = {0, 1, 2};
        PetscReal      coords[4] = {0.0, 1.0, 0.0, 0.0};
        PetscInt       markerPoints[6] = {1, 1, 2, 1, 3, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, user->interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 3; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      case 1:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 3;
        const int      cells[3]  = {0, 2, 3};
        PetscReal      coords[4] = {0.5, 0.5, 1.0, 1.0};
        PetscInt       markerPoints[6] = {1, 1, 2, 1, 3, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, user->interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 3; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      case 2:
      {
        const PetscInt numCells  = 2, numVertices = 1, numCorners = 3;
        const int      cells[6]  = {2, 4, 3, 2, 1, 4};
        PetscReal      coords[2] = {1.0, 0.0};
        PetscInt       markerPoints[10] = {2, 1, 3, 1, 4, 1, 5, 1, 6, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, user->interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 3; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
    break;
  default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %D", testNum);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CreateSimplex_3D(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       testNum = user->testNum, p;
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  switch (testNum) {
  case 0:
    if (size != 2) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh %d only for 2 processes", testNum);
    switch (rank) {
      case 0:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 4;
        const int      cells[4]  = {0, 2, 1, 3};
        PetscReal      coords[6] = {0.0, 0.0, -0.5,  0.0, -0.5, 0.0};
        PetscInt       markerPoints[8] = {1, 1, 2, 1, 3, 1, 4, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, user->interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 4; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      case 1:
      {
        const PetscInt numCells  = 1, numVertices = 3, numCorners = 4;
        const int      cells[4]  = {1, 2, 4, 3};
        PetscReal      coords[9] = {1.0, 0.0, 0.0,  0.0, 0.5, 0.0,  0.0, 0.0, 0.5};
        PetscInt       markerPoints[8] = {1, 1, 2, 1, 3, 1, 4, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, user->interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 4; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
    break;
  default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %D", testNum);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CreateQuad_2D(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       testNum = user->testNum, p;
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  switch (testNum) {
  case 0:
    if (size != 2) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh %d only for 2 processes", testNum);
    switch (rank) {
      case 0:
      {
        const PetscInt numCells  = 1, numVertices = 3, numCorners = 4;
        const int      cells[4]  = {0, 1, 2, 3};
        PetscReal      coords[6] = {-0.5, 0.0, 0.0, 0.0, 0.0, 1.0};
        PetscInt       markerPoints[4*2] = {1, 1, 2, 1, 3, 1, 4, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, user->interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 4; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      case 1:
      {
        const PetscInt numCells  = 1, numVertices = 3, numCorners = 4;
        const int      cells[4]  = {1, 4, 5, 2};
        PetscReal      coords[6] = {-0.5, 1.0, 0.5, 0.0, 0.5, 1.0};
        PetscInt       markerPoints[4*2] = {1, 1, 2, 1, 3, 1, 4, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, user->interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 4; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
    break;
  default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %D", testNum);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CreateHex_3D(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       testNum = user->testNum, p;
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  switch (testNum) {
  case 0:
    if (size != 2) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh %d only for 2 processes", testNum);
    switch (rank) {
    case 0:
    {
      const PetscInt numCells  = 1, numVertices = 6, numCorners = 8;
      const int      cells[8]  = {0, 3, 2, 1, 4, 5, 6, 7};
      PetscReal      coords[6*3] = {-0.5,0.0,0.0, 0.0,0.0,0.0, 0.0,1.0,0.0, -0.5,1.0,0.0, -0.5,0.0,1.0, 0.0,0.0,1.0};
      PetscInt       markerPoints[8*2] = {2,1,3,1,4,1,5,1,6,1,7,1,8,1,9,1};

      ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, user->interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
      for (p = 0; p < 4; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
    }
    break;
    case 1:
    {
      const PetscInt numCells  = 1, numVertices = 6, numCorners = 8;
      const int      cells[8]  = {1, 2, 9, 8, 5, 10, 11, 6};
      PetscReal      coords[6*3] = {0.0,1.0,1.0, -0.5,1.0,1.0, 0.5,0.0,0.0, 0.5,1.0,0.0, 0.5,0.0,1.0,  0.5,1.0,1.0};
      PetscInt       markerPoints[8*2] = {2,1,3,1,4,1,5,1,6,1,7,1,8,1,9,1};

      ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, user->interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
      for (p = 0; p < 4; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
    }
    break;
    default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
  break;
  default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %D", testNum);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CheckMesh(DM dm, AppCtx *user)
{
  PetscReal      detJ, J[9], refVol = 1.0;
  PetscReal      vol;
  PetscInt       dim, depth, d, cStart, cEnd, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  for (d = 0; d < dim; ++d) {
    refVol *= 2.0;
  }
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, NULL, J, NULL, &detJ);CHKERRQ(ierr);
    if (detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mesh cell %D is inverted, |J| = %g", c, (double)detJ);
    if (user->debug) {PetscPrintf(PETSC_COMM_SELF, "FEM Volume: %g\n", (double)detJ*refVol);CHKERRQ(ierr);}
    if (depth > 1) {
      ierr = DMPlexComputeCellGeometryFVM(dm, c, &vol, NULL, NULL);CHKERRQ(ierr);
      if (vol <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mesh cell %d is inverted, vol = %g", c, (double)vol);
      if (user->debug) {PetscPrintf(PETSC_COMM_SELF, "FVM Volume: %g\n", (double)vol);CHKERRQ(ierr);}
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(MPI_Comm comm, PetscInt testNum, AppCtx *user, DM *dm)
{
  PetscInt       dim          = user->dim;
  PetscBool      cellSimplex  = user->cellSimplex;
  PetscBool      useGenerator = user->useGenerator;
  const char    *filename     = user->filename;
  size_t         len;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (len) {
    ierr = DMPlexCreateFromFile(comm, filename, PETSC_FALSE, dm);CHKERRQ(ierr);
    ierr = DMGetDimension(*dm, &dim);CHKERRQ(ierr);
  } else if (useGenerator) {
    const PetscInt cells[3] = {2, 2, 2};

    ierr = DMPlexCreateBoxMesh(comm, dim, cellSimplex, cells, NULL, NULL, NULL, PETSC_FALSE, dm);CHKERRQ(ierr);
  } else {
    switch (dim) {
    case 2:
      if (cellSimplex) {
        ierr = CreateSimplex_2D(comm, user, dm);CHKERRQ(ierr);
      } else {
        ierr = CreateQuad_2D(comm, user, dm);CHKERRQ(ierr);
      }
      break;
    case 3:
      if (cellSimplex) {
        ierr = CreateSimplex_3D(comm, user, dm);CHKERRQ(ierr);
      } else {
        ierr = CreateHex_3D(comm, user, dm);CHKERRQ(ierr);
      }
      break;
    default:
      SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Cannot make meshes for dimension %D", dim);
    }
  }
  /* TODO: Turn on redistribution */
  if (0) {
    DM distributedMesh = NULL;

    /* Redistribute mesh over processes */
    ierr = DMPlexDistribute(*dm, 0, NULL, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMViewFromOptions(distributedMesh, NULL, "-dm_view");CHKERRQ(ierr);
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    }
  }
  ierr = PetscObjectSetName((PetscObject) *dm, "Parallel Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  user->dm = *dm;
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, user.testNum, &user, &user.dm);CHKERRQ(ierr);
  ierr = DMPlexCheckSymmetry(user.dm);CHKERRQ(ierr);
  ierr = DMPlexCheckSkeleton(user.dm, user.cellSimplex, 0);CHKERRQ(ierr);
  if (user.interpolate) {ierr = DMPlexCheckFaces(user.dm, user.cellSimplex, 0);CHKERRQ(ierr);}
  ierr = CheckMesh(user.dm, &user);CHKERRQ(ierr);
  ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    nsize: 2
    args: -dm_view ascii::ascii_info_detail
  test:
    suffix: 1
    nsize: 2
    args: -interpolate -dm_view ascii::ascii_info_detail
  test:
    suffix: 2
    nsize: 3
    args: -testnum 1 -interpolate -dm_view ascii::ascii_info_detail
  test:
    suffix: 3
    nsize: 2
    args: -dim 3 -dm_view ascii::ascii_info_detail
  test:
    suffix: 4
    nsize: 2
    args: -dim 3 -interpolate -dm_view ascii::ascii_info_detail
  test:
    suffix: quad_0
    nsize: 2
    args: -cell_simplex 0 -interpolate -dm_view ascii::ascii_info_detail
  test:
    suffix: quad_1
    nsize: 2
    args: -cell_simplex 0 -dim 3 -interpolate -dm_view ascii::ascii_info_detail

TEST*/
