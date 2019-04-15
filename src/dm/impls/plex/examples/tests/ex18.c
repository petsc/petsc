static char help[] = "Tests for parallel mesh loading\n\n";

#include <petsc/private/dmpleximpl.h>

/* List of test meshes

Network
-------
Test 0 (2 ranks):

network=0:
---------
  cell 0   cell 1   cell 2          nCells-1       (edge)
0 ------ 1 ------ 2 ------ 3 -- -- v --  -- nCells (vertex)

  vertex distribution:
    rank 0: 0 1
    rank 1: 2 3 ... nCells
  cell(edge) distribution:
    rank 0: 0 1
    rank 1: 2 ... nCells-1

network=1:
---------
               v2
                ^
                |
               cell 2
                |
 v0 --cell 0--> v3--cell 1--> v1

  vertex distribution:
    rank 0: 0 1 3
    rank 1: 2
  cell(edge) distribution:
    rank 0: 0 1
    rank 1: 2

  example:
    mpiexec -n 2 ./ex18 -distribute 1 -dim 1 -orig_dm_view -dist_dm_view -dist_dm_view -petscpartitioner_type parmetis -ncells 50

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

Test 2 (3 ranks):
Four triangles partitioned across 3 ranks

   1 _______ 3
   | \     / |
   |  \ 1 /  |
   |   \ /   |
   | 0  0  2 |
   |   / \   |
   |  / 3 \  |
   | /     \ |
   2 ------- 4

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

typedef enum {NONE, SERIAL, PARALLEL} InterpType;

typedef struct {
  PetscInt   debug;                        /* The debugging level */
  PetscInt   testNum;                      /* Indicates the mesh to create */
  PetscInt   dim;                          /* The topological mesh dimension */
  PetscBool  cellSimplex;                  /* Use simplices or hexes */
  PetscBool  distribute;                   /* Distribute the mesh */
  InterpType interpolate;                  /* Interpolate the mesh before or after DMPlexDistribute() */
  PetscBool  useGenerator;                 /* Construct mesh with a mesh generator */
  PetscBool  testOrientIF;                 /* Test for different original interface orientations */
  PetscInt   ornt[2];                      /* Orientation of interface on rank 0 and rank 1 */
  PetscInt   faces[3];                     /* Number of faces per dimension for generator */
  char       filename[PETSC_MAX_PATH_LEN]; /* Import mesh from file */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  const char    *interpTypes[3]  = {"none", "serial", "parallel"};
  PetscInt       interp=NONE, dim;
  PetscBool      flg1, flg2;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug        = 0;
  options->testNum      = 0;
  options->dim          = 2;
  options->cellSimplex  = PETSC_TRUE;
  options->distribute   = PETSC_FALSE;
  options->interpolate  = NONE;
  options->useGenerator = PETSC_FALSE;
  options->testOrientIF = PETSC_FALSE;
  options->ornt[0]      = 0;
  options->ornt[1]      = 0;
  options->faces[0]     = 2;
  options->faces[1]     = 2;
  options->faces[2]     = 2;
  options->filename[0]  = '\0';

  ierr = PetscOptionsBegin(comm, "", "Meshing Interpolation Test Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex18.c", options->debug, &options->debug, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-testnum", "The mesh to create", "ex18.c", options->testNum, &options->testNum, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex18.c", options->dim, &options->dim, &flg1);CHKERRQ(ierr);
  if (options->dim < 1 || options->dim > 3) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "dimension set to %d, must be between 1 and 3", options->dim);
  ierr = PetscOptionsBool("-cell_simplex", "Use simplices if true, otherwise hexes", "ex18.c", options->cellSimplex, &options->cellSimplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-distribute", "Distribute the mesh", "ex18.c", options->distribute, &options->distribute, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-interpolate", "Type of mesh interpolation, e.g. none, serial, parallel", "ex18.c", interpTypes, 3, interpTypes[options->interpolate], &interp, NULL);CHKERRQ(ierr);
  options->interpolate = (InterpType) interp;
  if (!options->distribute && options->interpolate == PARALLEL) SETERRQ(comm, PETSC_ERR_SUP, "-interpolate parallel  needs  -distribute 1");
  ierr = PetscOptionsBool("-use_generator", "Use a mesh generator to build the mesh", "ex18.c", options->useGenerator, &options->useGenerator, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-rotate_interface_0", "Rotation (relative orientation) of interface on rank 0; implies -interpolate serial -distribute 0", "ex18.c", options->ornt[0], &options->ornt[0], &options->testOrientIF);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-rotate_interface_1", "Rotation (relative orientation) of interface on rank 1; implies -interpolate serial -distribute 0", "ex18.c", options->ornt[1], &options->ornt[1], &flg2);CHKERRQ(ierr);
  if (flg2 != options->testOrientIF) SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "neither or both -rotate_interface_0 -rotate_interface_1 must be set");
  if (options->testOrientIF) {
    PetscInt i;
    for (i=0; i<2; i++) {
      if (options->ornt[i] >= 10) options->ornt[i] = -(options->ornt[i]-10);  /* 11 12 13 become -1 -2 -3 */
    }
    options->interpolate = SERIAL;
    options->distribute = PETSC_FALSE;
  }
  dim = 3;
  ierr = PetscOptionsIntArray("-faces", "Number of faces per dimension", "ex18.c", options->faces, &dim, &flg2);CHKERRQ(ierr);
  if (flg2) {
    if (flg1 && dim != options->dim) SETERRQ2(comm, PETSC_ERR_ARG_OUTOFRANGE, "specified -dim %D is not equal to length %D of -faces (note that -dim can be omitted)", options->dim, dim);
    options->dim = dim;
  }
  ierr = PetscOptionsString("-filename", "The mesh file", "ex18.c", options->filename, options->filename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh_1D(MPI_Comm comm, PetscBool interpolate, AppCtx *user, DM *dm)
{
  PetscInt       testNum = user->testNum;
  PetscMPIInt    rank,size;
  PetscErrorCode ierr;
  PetscInt       spacedim=2,numCorners=2,i;
  PetscInt       numCells,numVertices,network;
  int            *cells;
  PetscReal      *coords;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  if (size > 2) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh %d only for <=2 processes",testNum);

  numCells = 3;
  ierr = PetscOptionsGetInt(NULL, NULL, "-ncells", &numCells, NULL);CHKERRQ(ierr);
  if (numCells < 3) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Test ncells must >=3",numCells);

  if (size == 1) {
    numVertices = numCells + 1;
    ierr = PetscMalloc2(2*numCells,&cells,2*numVertices,&coords);CHKERRQ(ierr);
    for (i=0; i<numCells; i++) {
      cells[2*i] = i; cells[2*i+1] = i + 1;
    }

    ierr = DMPlexCreateFromCellList(comm, user->dim, numCells, numVertices, numCorners, PETSC_FALSE, (const int*)cells, spacedim, (const double*)coords, dm);CHKERRQ(ierr);
    ierr = PetscFree2(cells,coords);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  network = 0;
  ierr = PetscOptionsGetInt(NULL, NULL, "-network_case", &network, NULL);CHKERRQ(ierr);
  if (network == 0) {
    switch (rank) {
    case 0:
    {
      numCells    = 2;
      numVertices = numCells;
      ierr = PetscMalloc2(2*numCells,&cells,2*numCells,&coords);CHKERRQ(ierr);
      cells[0] = 0; cells[1] = 1;
      cells[2] = 1; cells[3] = 2;
    }
    break;
    case 1:
    {
      numCells    -= 2;
      numVertices = numCells + 1;
      ierr = PetscMalloc2(2*numCells,&cells,2*numCells,&coords);CHKERRQ(ierr);
      for (i=0; i<numCells; i++) {
        cells[2*i] = 2+i; cells[2*i+1] = 2 + i + 1;
      }
    }
    break;
    default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
  } else { /* network_case = 1 */
    /* ----------------------- */
    switch (rank) {
    case 0:
    {
      numCells    = 2;
      numVertices = 3;
      ierr = PetscMalloc2(2*numCells,&cells,2*numCells,&coords);CHKERRQ(ierr);
      cells[0] = 0; cells[1] = 3;
      cells[2] = 3; cells[3] = 1;
    }
    break;
    case 1:
    {
      numCells    = 1;
      numVertices = 1;
      ierr = PetscMalloc2(2*numCells,&cells,2*numCells,&coords);CHKERRQ(ierr);
      cells[0] = 3; cells[1] = 2;
    }
    break;
    default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
  }
  ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, PETSC_FALSE, (const int*)cells, spacedim, coords, NULL, dm);CHKERRQ(ierr);
  ierr = PetscFree2(cells,coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateSimplex_2D(MPI_Comm comm, PetscBool interpolate, AppCtx *user, DM *dm)
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

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 3; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      case 1:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 3;
        const int      cells[3]  = {1, 3, 2};
        PetscReal      coords[4] = {0.0, 1.0, 0.5, 0.5};
        PetscInt       markerPoints[6] = {1, 1, 2, 1, 3, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
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

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 3; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      case 1:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 3;
        const int      cells[3]  = {0, 2, 3};
        PetscReal      coords[4] = {0.5, 0.5, 1.0, 1.0};
        PetscInt       markerPoints[6] = {1, 1, 2, 1, 3, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 3; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      case 2:
      {
        const PetscInt numCells  = 2, numVertices = 1, numCorners = 3;
        const int      cells[6]  = {2, 4, 3, 2, 1, 4};
        PetscReal      coords[2] = {1.0, 0.0};
        PetscInt       markerPoints[10] = {2, 1, 3, 1, 4, 1, 5, 1, 6, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 3; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
    break;
  case 2:
    if (size != 3) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh %d only for 3 processes", testNum);
    switch (rank) {
      case 0:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 3;
        const int      cells[3]  = {1, 2, 0};
        PetscReal      coords[4] = {0.5, 0.5, 0.0, 1.0};
        PetscInt       markerPoints[6] = {1, 1, 2, 1, 3, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 3; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      case 1:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 3;
        const int      cells[3]  = {1, 0, 3};
        PetscReal      coords[4] = {0.0, 0.0, 1.0, 1.0};
        PetscInt       markerPoints[6] = {1, 1, 2, 1, 3, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 3; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      case 2:
      {
        const PetscInt numCells  = 2, numVertices = 1, numCorners = 3;
        const int      cells[6]  = {0, 4, 3, 0, 2, 4};
        PetscReal      coords[2] = {1.0, 0.0};
        PetscInt       markerPoints[10] = {2, 1, 3, 1, 4, 1, 5, 1, 6, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
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

static PetscErrorCode CreateSimplex_3D(MPI_Comm comm, PetscBool interpolate, AppCtx *user, DM *dm)
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

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 4; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      case 1:
      {
        const PetscInt numCells  = 1, numVertices = 3, numCorners = 4;
        const int      cells[4]  = {1, 2, 4, 3};
        PetscReal      coords[9] = {1.0, 0.0, 0.0,  0.0, 0.5, 0.0,  0.0, 0.0, 0.5};
        PetscInt       markerPoints[8] = {1, 1, 2, 1, 3, 1, 4, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 4; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
    break;
  default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %D", testNum);
  }
  if (user->testOrientIF) {
    PetscInt start;
    PetscBool reverse;
    PetscInt ifp[] = {8, 6};

    ierr = PetscObjectSetName((PetscObject) *dm, "Mesh before orientation");CHKERRQ(ierr);
    ierr = DMViewFromOptions(*dm, NULL, "-before_orientation_dm_view");CHKERRQ(ierr);
    /* rotate interface face ifp[rank] by given orientation ornt[rank] */
    ierr = DMPlexFixFaceOrientations_Translate_Private(user->ornt[rank], &start, &reverse);CHKERRQ(ierr);
    ierr = DMPlexOrientCell_Internal(*dm, ifp[rank], start, reverse);CHKERRQ(ierr);
    ierr = DMPlexCheckFaces(*dm, 0);CHKERRQ(ierr);
    ierr = DMPlexOrientInterface(*dm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateQuad_2D(MPI_Comm comm, PetscBool interpolate, AppCtx *user, DM *dm)
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

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 4; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      case 1:
      {
        const PetscInt numCells  = 1, numVertices = 3, numCorners = 4;
        const int      cells[4]  = {1, 4, 5, 2};
        PetscReal      coords[6] = {-0.5, 1.0, 0.5, 0.0, 0.5, 1.0};
        PetscInt       markerPoints[4*2] = {1, 1, 2, 1, 3, 1, 4, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
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

static PetscErrorCode CreateHex_3D(MPI_Comm comm, PetscBool interpolate, AppCtx *user, DM *dm)
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

      ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
      for (p = 0; p < 4; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
    }
    break;
    case 1:
    {
      const PetscInt numCells  = 1, numVertices = 6, numCorners = 8;
      const int      cells[8]  = {1, 2, 9, 8, 5, 10, 11, 6};
      PetscReal      coords[6*3] = {0.0,1.0,1.0, -0.5,1.0,1.0, 0.5,0.0,0.0, 0.5,1.0,0.0, 0.5,0.0,1.0,  0.5,1.0,1.0};
      PetscInt       markerPoints[8*2] = {2,1,3,1,4,1,5,1,6,1,7,1,8,1,9,1};

      ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
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

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim            = user->dim;
  PetscBool      cellSimplex    = user->cellSimplex;
  PetscBool      useGenerator   = user->useGenerator;
  PetscBool      interpSerial   = user->interpolate == SERIAL ? PETSC_TRUE : PETSC_FALSE;
  PetscBool      interpParallel = user->interpolate == PARALLEL ? PETSC_TRUE : PETSC_FALSE;
  const char    *filename       = user->filename;
  size_t         len;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (len) {
    ierr = DMPlexCreateFromFile(comm, filename, interpSerial, dm);CHKERRQ(ierr);
    ierr = DMGetDimension(*dm, &dim);CHKERRQ(ierr);
  } else if (useGenerator) {
    ierr = DMPlexCreateBoxMesh(comm, dim, cellSimplex, user->faces, NULL, NULL, NULL, interpSerial, dm);CHKERRQ(ierr);
  } else {
    switch (dim) {
    case 1:
      ierr = CreateMesh_1D(comm, interpSerial, user, dm);CHKERRQ(ierr);
      break;
    case 2:
      if (cellSimplex) {
        ierr = CreateSimplex_2D(comm, interpSerial, user, dm);CHKERRQ(ierr);
      } else {
        ierr = CreateQuad_2D(comm, interpSerial, user, dm);CHKERRQ(ierr);
      }
      break;
    case 3:
      if (cellSimplex) {
        ierr = CreateSimplex_3D(comm, interpSerial, user, dm);CHKERRQ(ierr);
      } else {
        ierr = CreateHex_3D(comm, interpSerial, user, dm);CHKERRQ(ierr);
      }
      break;
    default:
      SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Cannot make meshes for dimension %D", dim);
    }
  }
  ierr = PetscObjectSetName((PetscObject) *dm, "Original Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-orig_dm_view");CHKERRQ(ierr);

  if (user->distribute) {
    DM               pdm = NULL;
    PetscPartitioner part;

    /* Set partitioner options */
    ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetType(part, PETSCPARTITIONERSIMPLE);CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);

    /* Redistribute mesh over processes using that partitioner */
    ierr = DMPlexDistribute(*dm, 0, NULL, &pdm);CHKERRQ(ierr);
    if (pdm) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = pdm;
      ierr = PetscObjectSetName((PetscObject) *dm, "Redistributed Mesh");CHKERRQ(ierr);
      ierr = DMViewFromOptions(*dm, NULL, "-dist_dm_view");CHKERRQ(ierr);
    }

    if (interpParallel) {
      DM idm;

      ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm = idm;
      ierr = PetscObjectSetName((PetscObject) *dm, "Interpolated Redistributed Mesh");CHKERRQ(ierr);
      ierr = DMViewFromOptions(*dm, NULL, "-intp_dm_view");CHKERRQ(ierr);
    }
  }
  ierr = PetscObjectSetName((PetscObject) *dm, "Parallel Mesh");CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  if (user.interpolate != NONE) {
    ierr = DMPlexCheckPointSF(dm);CHKERRQ(ierr);
    ierr = DMPlexCheckFaces(dm, 0);CHKERRQ(ierr);
  }
  ierr = DMPlexCheckConesConformOnInterfaces(dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  testset:
    nsize: 2
    args: -dm_view ascii::ascii_info_detail -dm_plex_check_symmetry -dm_plex_check_skeleton -dm_plex_check_geometry
    test:
      suffix: 1_tri_dist0
      args: -distribute 0 -interpolate {{none serial}separate output}
    test:
      # Add back in 'none' and 'parallel'
      suffix: 1_tri_dist1
      args: -distribute 1 -interpolate {{serial}separate output}
    test:
      TODO: Parallel partitioning of uninterpolated meshes not supported
      suffix: 1_tri_dist1_ppu
      args: -distribute 1 -interpolate {{none parallel}separate output}
    test:
      suffix: 1_quad_dist0
      args: -cell_simplex 0 -distribute 0 -interpolate {{none serial}separate output}
    test:
      # Add back in 'none' and 'parallel'
      suffix: 1_quad_dist1
      args: -cell_simplex 0 -distribute 1 -interpolate {{serial}separate output}
    test:
      TODO: Parallel partitioning of uninterpolated meshes not supported
      suffix: 1_quad_dist1_ppu
      args: -cell_simplex 0 -distribute 1 -interpolate {{none parallel}separate output}
    test:
      suffix: 1_1d_dist1
      args: -dim 1 -distribute 1

  test:
    suffix: 2
    nsize: 3
    args: -testnum 1 -interpolate serial -dm_view ascii::ascii_info_detail -dm_plex_check_symmetry -dm_plex_check_skeleton -dm_plex_check_geometry
  test:
    requires:
    suffix: 2b
    nsize: 3
    args: -testnum 1 -interpolate serial -dm_view ascii::ascii_info_detail -dm_plex_check_symmetry -dm_plex_check_skeleton -dm_plex_check_geometry

  testset:
    # the same as 1% for 3D
    nsize: 2
    args: -dim 3 -dm_view ascii::ascii_info_detail
    test:
      suffix: 4_tet_dist0
      args: -distribute 0 -interpolate {{none serial}separate output}
    test:
      # Add back in 'none' and 'parallel'
      suffix: 4_tet_dist1
      args: -distribute 1 -interpolate {{serial}separate output}
    test:
      TODO: Parallel partitioning of uninterpolated meshes not supported
      suffix: 4_tet_dist1_ppu
      args: -distribute 1 -interpolate {{none parallel}separate output}
    test:
      suffix: 4_hex_dist0
      args: -cell_simplex 0 -distribute 0 -interpolate {{none serial}separate output}
    test:
      # Add back in 'none' and 'parallel'
      suffix: 4_hex_dist1
      args: -cell_simplex 0 -distribute 1 -interpolate {{serial}separate output}
    test:
      TODO: Parallel partitioning of uninterpolated meshes not supported
      suffix: 4_hex_dist1_ppu
      args: -cell_simplex 0 -distribute 1 -interpolate {{none parallel}separate output}

  test:
    # the same as 4_tet_dist0 but test different initial orientations
    suffix: 4_tet_test_orient
    nsize: 2
    args: -dim 3 -distribute 0 -dm_plex_check_symmetry -dm_plex_check_skeleton -dm_plex_check_geometry
    args: -rotate_interface_0 {{0 1 2 11 12 13}}
    args: -rotate_interface_1 {{0 1 2 11 12 13}}

  testset:
    requires: exodusii
    nsize: 2
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/TwoQuads.exo
    args: -cell_simplex 0 -dm_view ascii::ascii_info_detail -dm_plex_check_symmetry -dm_plex_check_skeleton -dm_plex_check_geometry
    test:
      suffix: 5_dist0
      args: -distribute 0 -interpolate {{none serial}separate output}
    test:
      suffix: 5_dist1
      args: -distribute 1 -interpolate {{none serial parallel}separate output}

  testset:
    nsize: {{1 2 4}}
    args: -use_generator -dm_plex_check_symmetry -dm_plex_check_geometry
    args: -distribute -interpolate {{none serial parallel}}
    test:
      suffix: 6_tri
      requires: triangle
      args: -faces {{2,2  1,3  7,4}} -cell_simplex 1 -dm_plex_generator triangle -dm_plex_check_skeleton
    test:
      suffix: 6_quad
      args: -faces {{2,2  1,3  7,4}} -cell_simplex 0 -dm_plex_check_skeleton
    test:
      TODO: this is failing due to DMPlexCheckPointSF() and should be fixed
      suffix: 6_tet
      requires: ctetgen
      args: -faces {{2,2,2  1,3,5  3,4,7}} -cell_simplex 1 -dm_plex_generator ctetgen -dm_plex_check_skeleton
    test:
      suffix: 6_hex
      args: -faces {{2,2,2  1,3,5  3,4,7}} -cell_simplex 0 -dm_plex_check_skeleton

  testset:
    nsize: {{1 2 4 5}}
    args: -cell_simplex 0 -distribute -dm_plex_check_symmetry -dm_plex_check_skeleton -dm_plex_check_geometry
    test:
      suffix: 7_exo
      requires: exodusii
      args: -interpolate {{none serial parallel}}
      args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo
    test:
      # Add back in 'none' and 'parallel'
      suffix: 7_hdf5
      requires: hdf5 !complex
      args: -interpolate serial
      args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.h5 -dm_plex_create_from_hdf5_xdmf
    test:
      TODO: Parallel partitioning of uninterpolated meshes not supported
      suffix: 7_hdf5_ppu
      requires: hdf5 !complex
      args: -interpolate {{none parallel}}
      args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.h5 -dm_plex_create_from_hdf5_xdmf


  test:
    suffix: 7_hdf5_hierarch
    requires: hdf5 ptscotch !complex
    nsize: {{2 3 4}separate output}
    args: -cell_simplex 0 -distribute -dm_plex_check_symmetry -dm_plex_check_skeleton -dm_plex_check_geometry
    args: -interpolate serial
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.h5 -dm_plex_create_from_hdf5_xdmf
    args: -petscpartitioner_type matpartitioning -petscpartitioner_view ::ascii_info
    args: -mat_partitioning_type hierarch -mat_partitioning_hierarchical_nfineparts 2
    args: -mat_partitioning_hierarchical_coarseparttype ptscotch -mat_partitioning_hierarchical_fineparttype ptscotch

TEST*/
