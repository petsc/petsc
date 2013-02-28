static char help[] = "Tests for mesh interpolation\n\n";

/* TODO
*/

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
   18   |   19 13  22
   / 9 17 10 \      \
  /     |     \      \
 2---14-|------4--20--6
  \     |     /      /
   \ 8  | 7  /      /
   16   |   15 11  21
     \  |  /      /
      \ | /      /
        3-------

In parallel,

 cell   5 ___25____8      4______    cell
 0    / | \        |\     |\      \     0
    16  |   18     | 21   | 13  6  11
    /10 19 12\    22  \   |8 \      \
   2-15-|----4--24-|---7  14  3--10--1
    \ 11| 9 /      |13 /  |  /      /
    14  |  17      | 20   | 12  5  9
      \ | /        |/     |/      /
        3----23----6      2------
         cell 1

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
   |   |     18  |   |     24  |   |
   |  36         |  35         |   44
   |19 |         |17 |         |23 |
  33   |  16    34   |   22   43   |
   |   5-----27--|---4-----39--|---11
   |  /          |  /          |  /
   | 28   14     | 26    20    | 38
   |/            |/            |/
   2-----25------3-----37------10

In parallel,

                         cell 2
cell   9-----27------8-----40------13     8----20------4  cell
0     /|            /|            /|     /|           /|     1
    28 |   15      26|           34|   24 |  10      19|
    /  |          /  |          /  |   /  |         /  |
   6-----25------7-----39------12  |  7----18------3   |
   |   |     18  |   |         |   |  |   |    13  |   |
   |  32         |  31         |   36 |  26        |   22
   |19 |         |17 |         |20 |  |14 |        |12 |
  29   |  16    30   |        35   |  25  |  11   21   |
   |   5-----23--|---4-----38--|---11 |   6----17--|---2
   |  /          |  /          |  /   |  /         |  /
   | 24   14     | 22          | 33   |23     9    | 16
   |/            |/            |/     |/           |/
   2-----21------3-----37------10     5----15------1

*/

typedef struct {
  DM        dm;
  PetscInt  debug;       /* The debugging level */
  PetscInt  dim;         /* The topological mesh dimension */
  PetscBool cellSimplex; /* Use simplices or hexes */
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug       = 0;
  options->dim         = 2;
  options->cellSimplex = PETSC_TRUE;

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex7.c", options->debug, &options->debug, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex7.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-cell_simplex", "Use simplices if true, otherwise hexes", "ex7.c", options->cellSimplex, &options->cellSimplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "CreateSimplex_2D"
PetscErrorCode CreateSimplex_2D(MPI_Comm comm, DM dm)
{
  PetscInt       depth = 1, testNum  = 0, p;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
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
        ierr = DMPlexSetLabelValue(dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);
      }
    }
    break;
    default:
      SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %d", testNum);
    }
  } else {
    PetscInt numPoints[2] = {0, 0};

    ierr = DMPlexCreateFromDAG(dm, depth, numPoints, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateSimplex_3D"
PetscErrorCode CreateSimplex_3D(MPI_Comm comm, DM dm)
{
  PetscInt       depth = 1, testNum  = 0, p;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (!rank) {
    switch (testNum) {
    case 0:
    {
      PetscInt    numPoints[2]        = {5, 2};
      PetscInt    coneSize[23]        = {4, 4, 0, 0, 0, 0, 0};
      PetscInt    cones[8]            = {2, 3, 4, 5,  3, 6, 4, 5};
      PetscInt    coneOrientations[8] = {0, 0, 0, 0,  0, 0, 0, 0};
      PetscScalar vertexCoords[15]    = {0.0, 0.0, -0.5,  0.0, -0.5, 0.0,  1.0, 0.0, 0.0,  0.0, 0.5, 0.0,  0.0, 0.0, 0.5};
      PetscInt    markerPoints[8]     = {2, 1, 3, 1, 4, 1, 5, 1};

      ierr = DMPlexCreateFromDAG(dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
      for (p = 0; p < 4; ++p) {
        ierr = DMPlexSetLabelValue(dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);
      }
    }
    break;
    default:
      SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %d", testNum);
    }
  } else {
    PetscInt numPoints[2] = {0, 0};

    ierr = DMPlexCreateFromDAG(dm, depth, numPoints, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateQuad_2D"
PetscErrorCode CreateQuad_2D(MPI_Comm comm, DM dm)
{
  PetscInt       depth = 1, testNum  = 0, p;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
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
        ierr = DMPlexSetLabelValue(dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);
      }
    }
    break;
    default:
      SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %d", testNum);
    }
  } else {
    PetscInt numPoints[2] = {0, 0};

    ierr = DMPlexCreateFromDAG(dm, depth, numPoints, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateHex_3D"
PetscErrorCode CreateHex_3D(MPI_Comm comm, DM dm)
{
  PetscInt       depth = 1, testNum  = 0, p;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (!rank) {
    switch (testNum) {
    case 0:
    {
      PetscInt    numPoints[2]         = {12, 2};
      PetscInt    coneSize[14]         = {8, 8, 0,0,0,0,0,0,0,0,0,0,0,0};
      PetscInt    cones[16]            = {2,3,4,5,6,7,8,9,  3,10,11,4,7,12,13,8};
      PetscInt    coneOrientations[16] = {0,0,0,0,0,0,0,0,  0, 0, 0,0,0, 0, 0,0};
      PetscScalar vertexCoords[36]     = {-0.5,0.0,0.0, 0.0,0.0,0.0, 0.0,1.0,0.0, -0.5,1.0,0.0,
                                          -0.5,0.0,1.0, 0.0,0.0,1.0, 0.0,1.0,1.0, -0.5,1.0,1.0,
                                           0.5,0.0,0.0, 0.5,1.0,0.0, 0.5,0.0,1.0,  0.5,1.0,1.0};
      PetscInt    markerPoints[16]     = {2,1,3,1,4,1,5,1,6,1,7,1,8,1,9,1};

      ierr = DMPlexCreateFromDAG(dm, depth, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
      for(p = 0; p < 8; ++p) {
        ierr = DMPlexSetLabelValue(dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);
      }
    }
    break;
    default:
      SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %d", testNum);
    }
  } else {
    PetscInt numPoints[2] = {0, 0};

    ierr = DMPlexCreateFromDAG(dm, depth, numPoints, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim          = user->dim;
  PetscBool      cellSimplex  = user->cellSimplex;
  const char     *partitioner = "chaco";
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexSetDimension(*dm, dim);CHKERRQ(ierr);
  switch (dim) {
  case 2:
    if (cellSimplex) {
      ierr = CreateSimplex_2D(comm, *dm);CHKERRQ(ierr);
    } else {
      ierr = CreateQuad_2D(comm, *dm);CHKERRQ(ierr);
    }
    break;
  case 3:
    if (cellSimplex) {
      ierr = CreateSimplex_3D(comm, *dm);CHKERRQ(ierr);
    } else {
      ierr = CreateHex_3D(comm, *dm);CHKERRQ(ierr);
    }
    break;
  default:
    SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Cannot make meshes for dimension %d", dim);
  }
  {
    DM interpolatedMesh = NULL;

    ierr = DMPlexInterpolate(*dm, &interpolatedMesh);CHKERRQ(ierr);
    ierr = DMPlexCopyCoordinates(*dm, interpolatedMesh);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = interpolatedMesh;
  }
  {
    DM distributedMesh = NULL;

    /* Distribute mesh over processes */
    ierr = DMPlexDistribute(*dm, partitioner, 0, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    }
  }
  ierr     = PetscObjectSetName((PetscObject) *dm, "Interpolated Mesh");CHKERRQ(ierr);
  ierr     = DMSetFromOptions(*dm);CHKERRQ(ierr);
  user->dm = *dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  AppCtx         user;                 /* user-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &user.dm);CHKERRQ(ierr);
  ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
