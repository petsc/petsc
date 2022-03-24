static char help[] = "Tests for parallel mesh loading and parallel topological interpolation\n\n";

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

typedef enum {NONE, CREATE, AFTER_CREATE, AFTER_DISTRIBUTE} InterpType;

typedef struct {
  PetscInt   debug;                        /* The debugging level */
  PetscInt   testNum;                      /* Indicates the mesh to create */
  PetscInt   dim;                          /* The topological mesh dimension */
  PetscBool  cellSimplex;                  /* Use simplices or hexes */
  PetscBool  distribute;                   /* Distribute the mesh */
  InterpType interpolate;                  /* Interpolate the mesh before or after DMPlexDistribute() */
  PetscBool  useGenerator;                 /* Construct mesh with a mesh generator */
  PetscBool  testOrientIF;                 /* Test for different original interface orientations */
  PetscBool  testHeavy;                    /* Run the heavy PointSF test */
  PetscBool  customView;                   /* Show results of DMPlexIsInterpolated() etc. */
  PetscInt   ornt[2];                      /* Orientation of interface on rank 0 and rank 1 */
  PetscInt   faces[3];                     /* Number of faces per dimension for generator */
  PetscScalar coords[128];
  PetscReal  coordsTol;
  PetscInt   ncoords;
  PetscInt   pointsToExpand[128];
  PetscInt   nPointsToExpand;
  PetscBool  testExpandPointsEmpty;
  char       filename[PETSC_MAX_PATH_LEN]; /* Import mesh from file */
} AppCtx;

struct _n_PortableBoundary {
  Vec coordinates;
  PetscInt depth;
  PetscSection *sections;
};
typedef struct _n_PortableBoundary * PortableBoundary;

#if defined(PETSC_USE_LOG)
static PetscLogStage  stage[3];
#endif

static PetscErrorCode DMPlexCheckPointSFHeavy(DM, PortableBoundary);
static PetscErrorCode DMPlexSetOrientInterface_Private(DM,PetscBool);
static PetscErrorCode DMPlexGetExpandedBoundary_Private(DM, PortableBoundary *);
static PetscErrorCode DMPlexExpandedConesToFaces_Private(DM, IS, PetscSection, IS *);

static PetscErrorCode PortableBoundaryDestroy(PortableBoundary *bnd)
{
  PetscInt       d;

  PetscFunctionBegin;
  if (!*bnd) PetscFunctionReturn(0);
  CHKERRQ(VecDestroy(&(*bnd)->coordinates));
  for (d=0; d < (*bnd)->depth; d++) {
    CHKERRQ(PetscSectionDestroy(&(*bnd)->sections[d]));
  }
  CHKERRQ(PetscFree((*bnd)->sections));
  CHKERRQ(PetscFree(*bnd));
  PetscFunctionReturn(0);
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  const char    *interpTypes[4]  = {"none", "create", "after_create", "after_distribute"};
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
  options->testHeavy    = PETSC_TRUE;
  options->customView   = PETSC_FALSE;
  options->testExpandPointsEmpty = PETSC_FALSE;
  options->ornt[0]      = 0;
  options->ornt[1]      = 0;
  options->faces[0]     = 2;
  options->faces[1]     = 2;
  options->faces[2]     = 2;
  options->filename[0]  = '\0';
  options->coordsTol    = PETSC_DEFAULT;

  ierr = PetscOptionsBegin(comm, "", "Meshing Interpolation Test Options", "DMPLEX");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsBoundedInt("-debug", "The debugging level", "ex18.c", options->debug, &options->debug, NULL,0));
  CHKERRQ(PetscOptionsBoundedInt("-testnum", "The mesh to create", "ex18.c", options->testNum, &options->testNum, NULL,0));
  CHKERRQ(PetscOptionsBool("-cell_simplex", "Generate simplices if true, otherwise hexes", "ex18.c", options->cellSimplex, &options->cellSimplex, NULL));
  CHKERRQ(PetscOptionsBool("-distribute", "Distribute the mesh", "ex18.c", options->distribute, &options->distribute, NULL));
  CHKERRQ(PetscOptionsEList("-interpolate", "Type of mesh interpolation (none, create, after_create, after_distribute)", "ex18.c", interpTypes, 4, interpTypes[options->interpolate], &interp, NULL));
  options->interpolate = (InterpType) interp;
  PetscCheckFalse(!options->distribute && options->interpolate == AFTER_DISTRIBUTE,comm, PETSC_ERR_SUP, "-interpolate after_distribute  needs  -distribute 1");
  CHKERRQ(PetscOptionsBool("-use_generator", "Use a mesh generator to build the mesh", "ex18.c", options->useGenerator, &options->useGenerator, NULL));
  options->ncoords = 128;
  CHKERRQ(PetscOptionsScalarArray("-view_vertices_from_coords", "Print DAG points corresponding to vertices with given coordinates", "ex18.c", options->coords, &options->ncoords, NULL));
  CHKERRQ(PetscOptionsReal("-view_vertices_from_coords_tol", "Tolerance for -view_vertices_from_coords", "ex18.c", options->coordsTol, &options->coordsTol, NULL));
  options->nPointsToExpand = 128;
  CHKERRQ(PetscOptionsIntArray("-test_expand_points", "Expand given array of DAG point using DMPlexGetConeRecursive() and print results", "ex18.c", options->pointsToExpand, &options->nPointsToExpand, NULL));
  if (options->nPointsToExpand) {
    CHKERRQ(PetscOptionsBool("-test_expand_points_empty", "For -test_expand_points, rank 0 will have empty input array", "ex18.c", options->testExpandPointsEmpty, &options->testExpandPointsEmpty, NULL));
  }
  CHKERRQ(PetscOptionsBool("-test_heavy", "Run the heavy PointSF test", "ex18.c", options->testHeavy, &options->testHeavy, NULL));
  CHKERRQ(PetscOptionsBool("-custom_view", "Custom DMPlex view", "ex18.c", options->customView, &options->customView, NULL));
  CHKERRQ(PetscOptionsRangeInt("-dim", "The topological mesh dimension", "ex18.c", options->dim, &options->dim, &flg1,1,3));
  dim = 3;
  CHKERRQ(PetscOptionsIntArray("-faces", "Number of faces per dimension", "ex18.c", options->faces, &dim, &flg2));
  if (flg2) {
    PetscCheckFalse(flg1 && dim != options->dim,comm, PETSC_ERR_ARG_OUTOFRANGE, "specified -dim %D is not equal to length %D of -faces (note that -dim can be omitted)", options->dim, dim);
    options->dim = dim;
  }
  CHKERRQ(PetscOptionsString("-filename", "The mesh file", "ex18.c", options->filename, options->filename, sizeof(options->filename), NULL));
  CHKERRQ(PetscOptionsBoundedInt("-rotate_interface_0", "Rotation (relative orientation) of interface on rank 0; implies -interpolate create -distribute 0", "ex18.c", options->ornt[0], &options->ornt[0], &options->testOrientIF,0));
  CHKERRQ(PetscOptionsBoundedInt("-rotate_interface_1", "Rotation (relative orientation) of interface on rank 1; implies -interpolate create -distribute 0", "ex18.c", options->ornt[1], &options->ornt[1], &flg2,0));
  PetscCheckFalse(flg2 != options->testOrientIF,comm, PETSC_ERR_ARG_OUTOFRANGE, "neither or both -rotate_interface_0 -rotate_interface_1 must be set");
  if (options->testOrientIF) {
    PetscInt i;
    for (i=0; i<2; i++) {
      if (options->ornt[i] >= 10) options->ornt[i] = -(options->ornt[i]-10);  /* 11 12 13 become -1 -2 -3 */
    }
    options->filename[0]  = 0;
    options->useGenerator = PETSC_FALSE;
    options->dim          = 3;
    options->cellSimplex  = PETSC_TRUE;
    options->interpolate  = CREATE;
    options->distribute   = PETSC_FALSE;
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh_1D(MPI_Comm comm, PetscBool interpolate, AppCtx *user, DM *dm)
{
  PetscInt       testNum = user->testNum;
  PetscMPIInt    rank,size;
  PetscInt       numCorners=2,i;
  PetscInt       numCells,numVertices,network;
  PetscInt       *cells;
  PetscReal      *coords;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  PetscCheckFalse(size > 2,comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh %d only for <=2 processes",testNum);

  numCells = 3;
  CHKERRQ(PetscOptionsGetInt(NULL, NULL, "-ncells", &numCells, NULL));
  PetscCheckFalse(numCells < 3,comm, PETSC_ERR_ARG_OUTOFRANGE, "Test ncells must >=3",numCells);

  if (size == 1) {
    numVertices = numCells + 1;
    CHKERRQ(PetscMalloc2(2*numCells,&cells,2*numVertices,&coords));
    for (i=0; i<numCells; i++) {
      cells[2*i] = i; cells[2*i+1] = i + 1;
      coords[2*i] = i; coords[2*i+1] = i + 1;
    }

    CHKERRQ(DMPlexCreateFromCellListPetsc(comm, user->dim, numCells, numVertices, numCorners, PETSC_FALSE, cells, user->dim, coords, dm));
    CHKERRQ(PetscFree2(cells,coords));
    PetscFunctionReturn(0);
  }

  network = 0;
  CHKERRQ(PetscOptionsGetInt(NULL, NULL, "-network_case", &network, NULL));
  if (network == 0) {
    switch (rank) {
    case 0:
    {
      numCells    = 2;
      numVertices = numCells;
      CHKERRQ(PetscMalloc2(2*numCells,&cells,2*numCells,&coords));
      cells[0] = 0; cells[1] = 1;
      cells[2] = 1; cells[3] = 2;
      coords[0] = 0.; coords[1] = 1.;
      coords[2] = 1.; coords[3] = 2.;
    }
    break;
    case 1:
    {
      numCells    -= 2;
      numVertices = numCells + 1;
      CHKERRQ(PetscMalloc2(2*numCells,&cells,2*numCells,&coords));
      for (i=0; i<numCells; i++) {
        cells[2*i] = 2+i; cells[2*i+1] = 2 + i + 1;
        coords[2*i] = 2+i; coords[2*i+1] = 2 + i + 1;
      }
    }
    break;
    default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
  } else { /* network_case = 1 */
    /* ----------------------- */
    switch (rank) {
    case 0:
    {
      numCells    = 2;
      numVertices = 3;
      CHKERRQ(PetscMalloc2(2*numCells,&cells,2*numCells,&coords));
      cells[0] = 0; cells[1] = 3;
      cells[2] = 3; cells[3] = 1;
    }
    break;
    case 1:
    {
      numCells    = 1;
      numVertices = 1;
      CHKERRQ(PetscMalloc2(2*numCells,&cells,2*numCells,&coords));
      cells[0] = 3; cells[1] = 2;
    }
    break;
    default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
  }
  CHKERRQ(DMPlexCreateFromCellListParallelPetsc(comm, user->dim, numCells, numVertices, PETSC_DECIDE, numCorners, PETSC_FALSE, cells, user->dim, coords, NULL, NULL, dm));
  CHKERRQ(PetscFree2(cells,coords));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateSimplex_2D(MPI_Comm comm, PetscBool interpolate, AppCtx *user, DM *dm)
{
  PetscInt       testNum = user->testNum, p;
  PetscMPIInt    rank, size;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  switch (testNum) {
  case 0:
    PetscCheckFalse(size != 2,comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh %d only for 2 processes", testNum);
    switch (rank) {
      case 0:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 3;
        const PetscInt cells[3]  = {0, 1, 2};
        PetscReal      coords[4] = {-0.5, 0.5, 0.0, 0.0};
        PetscInt       markerPoints[6] = {1, 1, 2, 1, 3, 1};

        CHKERRQ(DMPlexCreateFromCellListParallelPetsc(comm, user->dim, numCells, numVertices, PETSC_DECIDE, numCorners, interpolate, cells, user->dim, coords, NULL, NULL, dm));
        for (p = 0; p < 3; ++p) CHKERRQ(DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]));
      }
      break;
      case 1:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 3;
        const PetscInt cells[3]  = {1, 3, 2};
        PetscReal      coords[4] = {0.0, 1.0, 0.5, 0.5};
        PetscInt       markerPoints[6] = {1, 1, 2, 1, 3, 1};

        CHKERRQ(DMPlexCreateFromCellListParallelPetsc(comm, user->dim, numCells, numVertices, PETSC_DECIDE, numCorners, interpolate, cells, user->dim, coords, NULL, NULL, dm));
        for (p = 0; p < 3; ++p) CHKERRQ(DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]));
      }
      break;
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
    break;
  case 1:
    PetscCheckFalse(size != 3,comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh %d only for 3 processes", testNum);
    switch (rank) {
      case 0:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 3;
        const PetscInt cells[3]  = {0, 1, 2};
        PetscReal      coords[4] = {0.0, 1.0, 0.0, 0.0};
        PetscInt       markerPoints[6] = {1, 1, 2, 1, 3, 1};

        CHKERRQ(DMPlexCreateFromCellListParallelPetsc(comm, user->dim, numCells, numVertices, PETSC_DECIDE, numCorners, interpolate, cells, user->dim, coords, NULL, NULL, dm));
        for (p = 0; p < 3; ++p) CHKERRQ(DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]));
      }
      break;
      case 1:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 3;
        const PetscInt cells[3]  = {0, 2, 3};
        PetscReal      coords[4] = {0.5, 0.5, 1.0, 1.0};
        PetscInt       markerPoints[6] = {1, 1, 2, 1, 3, 1};

        CHKERRQ(DMPlexCreateFromCellListParallelPetsc(comm, user->dim, numCells, numVertices, PETSC_DECIDE, numCorners, interpolate, cells, user->dim, coords, NULL, NULL, dm));
        for (p = 0; p < 3; ++p) CHKERRQ(DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]));
      }
      break;
      case 2:
      {
        const PetscInt numCells  = 2, numVertices = 1, numCorners = 3;
        const PetscInt cells[6]  = {2, 4, 3, 2, 1, 4};
        PetscReal      coords[2] = {1.0, 0.0};
        PetscInt       markerPoints[10] = {2, 1, 3, 1, 4, 1, 5, 1, 6, 1};

        CHKERRQ(DMPlexCreateFromCellListParallelPetsc(comm, user->dim, numCells, numVertices, PETSC_DECIDE, numCorners, interpolate, cells, user->dim, coords, NULL, NULL, dm));
        for (p = 0; p < 3; ++p) CHKERRQ(DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]));
      }
      break;
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
    break;
  case 2:
    PetscCheckFalse(size != 3,comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh %d only for 3 processes", testNum);
    switch (rank) {
      case 0:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 3;
        const PetscInt cells[3]  = {1, 2, 0};
        PetscReal      coords[4] = {0.5, 0.5, 0.0, 1.0};
        PetscInt       markerPoints[6] = {1, 1, 2, 1, 3, 1};

        CHKERRQ(DMPlexCreateFromCellListParallelPetsc(comm, user->dim, numCells, numVertices, PETSC_DECIDE, numCorners, interpolate, cells, user->dim, coords, NULL, NULL, dm));
        for (p = 0; p < 3; ++p) CHKERRQ(DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]));
      }
      break;
      case 1:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 3;
        const PetscInt cells[3]  = {1, 0, 3};
        PetscReal      coords[4] = {0.0, 0.0, 1.0, 1.0};
        PetscInt       markerPoints[6] = {1, 1, 2, 1, 3, 1};

        CHKERRQ(DMPlexCreateFromCellListParallelPetsc(comm, user->dim, numCells, numVertices, PETSC_DECIDE, numCorners, interpolate, cells, user->dim, coords, NULL, NULL, dm));
        for (p = 0; p < 3; ++p) CHKERRQ(DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]));
      }
      break;
      case 2:
      {
        const PetscInt numCells  = 2, numVertices = 1, numCorners = 3;
        const PetscInt cells[6]  = {0, 4, 3, 0, 2, 4};
        PetscReal      coords[2] = {1.0, 0.0};
        PetscInt       markerPoints[10] = {2, 1, 3, 1, 4, 1, 5, 1, 6, 1};

        CHKERRQ(DMPlexCreateFromCellListParallelPetsc(comm, user->dim, numCells, numVertices, PETSC_DECIDE, numCorners, interpolate, cells, user->dim, coords, NULL, NULL, dm));
        for (p = 0; p < 3; ++p) CHKERRQ(DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]));
      }
      break;
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
    break;
  default: SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %D", testNum);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateSimplex_3D(MPI_Comm comm, PetscBool interpolate, AppCtx *user, DM *dm)
{
  PetscInt       testNum = user->testNum, p;
  PetscMPIInt    rank, size;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  switch (testNum) {
  case 0:
    PetscCheckFalse(size != 2,comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh %d only for 2 processes", testNum);
    switch (rank) {
      case 0:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 4;
        const PetscInt cells[4]  = {0, 2, 1, 3};
        PetscReal      coords[6] = {0.0, 0.0, -0.5,  0.0, -0.5, 0.0};
        PetscInt       markerPoints[8] = {1, 1, 2, 1, 3, 1, 4, 1};

        CHKERRQ(DMPlexCreateFromCellListParallelPetsc(comm, user->dim, numCells, numVertices, PETSC_DECIDE, numCorners, interpolate, cells, user->dim, coords, NULL, NULL, dm));
        for (p = 0; p < 4; ++p) CHKERRQ(DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]));
      }
      break;
      case 1:
      {
        const PetscInt numCells  = 1, numVertices = 3, numCorners = 4;
        const PetscInt cells[4]  = {1, 2, 4, 3};
        PetscReal      coords[9] = {1.0, 0.0, 0.0,  0.0, 0.5, 0.0,  0.0, 0.0, 0.5};
        PetscInt       markerPoints[8] = {1, 1, 2, 1, 3, 1, 4, 1};

        CHKERRQ(DMPlexCreateFromCellListParallelPetsc(comm, user->dim, numCells, numVertices, PETSC_DECIDE, numCorners, interpolate, cells, user->dim, coords, NULL, NULL, dm));
        for (p = 0; p < 4; ++p) CHKERRQ(DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]));
      }
      break;
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
    break;
  default: SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %D", testNum);
  }
  if (user->testOrientIF) {
    PetscInt ifp[] = {8, 6};

    CHKERRQ(PetscObjectSetName((PetscObject) *dm, "Mesh before orientation"));
    CHKERRQ(DMViewFromOptions(*dm, NULL, "-before_orientation_dm_view"));
    /* rotate interface face ifp[rank] by given orientation ornt[rank] */
    CHKERRQ(DMPlexOrientPoint(*dm, ifp[rank], user->ornt[rank]));
    CHKERRQ(DMViewFromOptions(*dm, NULL, "-before_orientation_dm_view"));
    CHKERRQ(DMPlexCheckFaces(*dm, 0));
    CHKERRQ(DMPlexOrientInterface_Internal(*dm));
    CHKERRQ(PetscPrintf(comm, "Orientation test PASSED\n"));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateQuad_2D(MPI_Comm comm, PetscBool interpolate, AppCtx *user, DM *dm)
{
  PetscInt       testNum = user->testNum, p;
  PetscMPIInt    rank, size;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  switch (testNum) {
  case 0:
    PetscCheckFalse(size != 2,comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh %d only for 2 processes", testNum);
    switch (rank) {
      case 0:
      {
        const PetscInt numCells  = 1, numVertices = 3, numCorners = 4;
        const PetscInt cells[4]  = {0, 1, 2, 3};
        PetscReal      coords[6] = {-0.5, 0.0, 0.0, 0.0, 0.0, 1.0};
        PetscInt       markerPoints[4*2] = {1, 1, 2, 1, 3, 1, 4, 1};

        CHKERRQ(DMPlexCreateFromCellListParallelPetsc(comm, user->dim, numCells, numVertices, PETSC_DECIDE, numCorners, interpolate, cells, user->dim, coords, NULL, NULL, dm));
        for (p = 0; p < 4; ++p) CHKERRQ(DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]));
      }
      break;
      case 1:
      {
        const PetscInt numCells  = 1, numVertices = 3, numCorners = 4;
        const PetscInt cells[4]  = {1, 4, 5, 2};
        PetscReal      coords[6] = {-0.5, 1.0, 0.5, 0.0, 0.5, 1.0};
        PetscInt       markerPoints[4*2] = {1, 1, 2, 1, 3, 1, 4, 1};

        CHKERRQ(DMPlexCreateFromCellListParallelPetsc(comm, user->dim, numCells, numVertices, PETSC_DECIDE, numCorners, interpolate, cells, user->dim, coords, NULL, NULL, dm));
        for (p = 0; p < 4; ++p) CHKERRQ(DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]));
      }
      break;
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
    break;
  default: SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %D", testNum);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateHex_3D(MPI_Comm comm, PetscBool interpolate, AppCtx *user, DM *dm)
{
  PetscInt       testNum = user->testNum, p;
  PetscMPIInt    rank, size;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  switch (testNum) {
  case 0:
    PetscCheckFalse(size != 2,comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh %d only for 2 processes", testNum);
    switch (rank) {
    case 0:
    {
      const PetscInt numCells  = 1, numVertices = 6, numCorners = 8;
      const PetscInt cells[8]  = {0, 3, 2, 1, 4, 5, 6, 7};
      PetscReal      coords[6*3] = {-0.5,0.0,0.0, 0.0,0.0,0.0, 0.0,1.0,0.0, -0.5,1.0,0.0, -0.5,0.0,1.0, 0.0,0.0,1.0};
      PetscInt       markerPoints[8*2] = {2,1,3,1,4,1,5,1,6,1,7,1,8,1,9,1};

      CHKERRQ(DMPlexCreateFromCellListParallelPetsc(comm, user->dim, numCells, numVertices, PETSC_DECIDE, numCorners, interpolate, cells, user->dim, coords, NULL, NULL, dm));
      for (p = 0; p < 4; ++p) CHKERRQ(DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]));
    }
    break;
    case 1:
    {
      const PetscInt numCells  = 1, numVertices = 6, numCorners = 8;
      const PetscInt cells[8]  = {1, 2, 9, 8, 5, 10, 11, 6};
      PetscReal      coords[6*3] = {0.0,1.0,1.0, -0.5,1.0,1.0, 0.5,0.0,0.0, 0.5,1.0,0.0, 0.5,0.0,1.0,  0.5,1.0,1.0};
      PetscInt       markerPoints[8*2] = {2,1,3,1,4,1,5,1,6,1,7,1,8,1,9,1};

      CHKERRQ(DMPlexCreateFromCellListParallelPetsc(comm, user->dim, numCells, numVertices, PETSC_DECIDE, numCorners, interpolate, cells, user->dim, coords, NULL, NULL, dm));
      for (p = 0; p < 4; ++p) CHKERRQ(DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]));
    }
    break;
    default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
  break;
  default: SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %D", testNum);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CustomView(DM dm, PetscViewer v)
{
  DMPlexInterpolatedFlag interpolated;
  PetscBool              distributed;

  PetscFunctionBegin;
  CHKERRQ(DMPlexIsDistributed(dm, &distributed));
  CHKERRQ(DMPlexIsInterpolatedCollective(dm, &interpolated));
  CHKERRQ(PetscViewerASCIIPrintf(v, "DMPlexIsDistributed: %s\n", PetscBools[distributed]));
  CHKERRQ(PetscViewerASCIIPrintf(v, "DMPlexIsInterpolatedCollective: %s\n", DMPlexInterpolatedFlags[interpolated]));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMeshFromFile(MPI_Comm comm, AppCtx *user, DM *dm, DM *serialDM)
{
  const char    *filename       = user->filename;
  PetscBool      testHeavy      = user->testHeavy;
  PetscBool      interpCreate   = user->interpolate == CREATE ? PETSC_TRUE : PETSC_FALSE;
  PetscBool      distributed    = PETSC_FALSE;

  PetscFunctionBegin;
  *serialDM = NULL;
  if (testHeavy && interpCreate) CHKERRQ(DMPlexSetOrientInterface_Private(NULL, PETSC_FALSE));
  CHKERRQ(PetscLogStagePush(stage[0]));
  CHKERRQ(DMPlexCreateFromFile(comm, filename, "ex18_plex", interpCreate, dm)); /* with DMPlexOrientInterface_Internal() call skipped so that PointSF issues are left to DMPlexCheckPointSFHeavy() */
  CHKERRQ(PetscLogStagePop());
  if (testHeavy && interpCreate) CHKERRQ(DMPlexSetOrientInterface_Private(NULL, PETSC_TRUE));
  CHKERRQ(DMPlexIsDistributed(*dm, &distributed));
  CHKERRQ(PetscPrintf(comm, "DMPlexCreateFromFile produced %s mesh.\n", distributed ? "distributed" : "serial"));
  if (testHeavy && distributed) {
    CHKERRQ(PetscOptionsSetValue(NULL, "-dm_plex_hdf5_force_sequential", NULL));
    CHKERRQ(DMPlexCreateFromFile(comm, filename, "ex18_plex", interpCreate, serialDM));
    CHKERRQ(DMPlexIsDistributed(*serialDM, &distributed));
    PetscCheck(!distributed,comm, PETSC_ERR_PLIB, "unable to create a serial DM from file");
  }
  CHKERRQ(DMGetDimension(*dm, &user->dim));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscPartitioner part;
  PortableBoundary boundary     = NULL;
  DM             serialDM       = NULL;
  PetscBool      cellSimplex    = user->cellSimplex;
  PetscBool      useGenerator   = user->useGenerator;
  PetscBool      interpCreate   = user->interpolate == CREATE ? PETSC_TRUE : PETSC_FALSE;
  PetscBool      interpSerial   = user->interpolate == AFTER_CREATE ? PETSC_TRUE : PETSC_FALSE;
  PetscBool      interpParallel = user->interpolate == AFTER_DISTRIBUTE ? PETSC_TRUE : PETSC_FALSE;
  PetscBool      testHeavy      = user->testHeavy;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  if (user->filename[0]) {
    CHKERRQ(CreateMeshFromFile(comm, user, dm, &serialDM));
  } else if (useGenerator) {
    CHKERRQ(PetscLogStagePush(stage[0]));
    CHKERRQ(DMPlexCreateBoxMesh(comm, user->dim, cellSimplex, user->faces, NULL, NULL, NULL, interpCreate, dm));
    CHKERRQ(PetscLogStagePop());
  } else {
    CHKERRQ(PetscLogStagePush(stage[0]));
    switch (user->dim) {
    case 1:
      CHKERRQ(CreateMesh_1D(comm, interpCreate, user, dm));
      break;
    case 2:
      if (cellSimplex) {
        CHKERRQ(CreateSimplex_2D(comm, interpCreate, user, dm));
      } else {
        CHKERRQ(CreateQuad_2D(comm, interpCreate, user, dm));
      }
      break;
    case 3:
      if (cellSimplex) {
        CHKERRQ(CreateSimplex_3D(comm, interpCreate, user, dm));
      } else {
        CHKERRQ(CreateHex_3D(comm, interpCreate, user, dm));
      }
      break;
    default:
      SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Cannot make meshes for dimension %D", user->dim);
    }
    CHKERRQ(PetscLogStagePop());
  }
  PetscCheckFalse(user->ncoords % user->dim,comm, PETSC_ERR_ARG_OUTOFRANGE, "length of coordinates array %D must be divisable by spatial dimension %D", user->ncoords, user->dim);
  CHKERRQ(PetscObjectSetName((PetscObject) *dm, "Original Mesh"));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-orig_dm_view"));

  if (interpSerial) {
    DM idm;

    if (testHeavy) CHKERRQ(DMPlexSetOrientInterface_Private(*dm, PETSC_FALSE));
    CHKERRQ(PetscLogStagePush(stage[2]));
    CHKERRQ(DMPlexInterpolate(*dm, &idm)); /* with DMPlexOrientInterface_Internal() call skipped so that PointSF issues are left to DMPlexCheckPointSFHeavy() */
    CHKERRQ(PetscLogStagePop());
    if (testHeavy) CHKERRQ(DMPlexSetOrientInterface_Private(*dm, PETSC_TRUE));
    CHKERRQ(DMDestroy(dm));
    *dm = idm;
    CHKERRQ(PetscObjectSetName((PetscObject) *dm, "Interpolated Mesh"));
    CHKERRQ(DMViewFromOptions(*dm, NULL, "-intp_dm_view"));
  }

  /* Set partitioner options */
  CHKERRQ(DMPlexGetPartitioner(*dm, &part));
  if (part) {
    CHKERRQ(PetscPartitionerSetType(part, PETSCPARTITIONERSIMPLE));
    CHKERRQ(PetscPartitionerSetFromOptions(part));
  }

  if (user->customView) CHKERRQ(CustomView(*dm, PETSC_VIEWER_STDOUT_(comm)));
  if (testHeavy) {
    PetscBool distributed;

    CHKERRQ(DMPlexIsDistributed(*dm, &distributed));
    if (!serialDM && !distributed) {
      serialDM = *dm;
      CHKERRQ(PetscObjectReference((PetscObject)*dm));
    }
    if (serialDM) {
      CHKERRQ(DMPlexGetExpandedBoundary_Private(serialDM, &boundary));
    }
    if (boundary) {
      /* check DM which has been created in parallel and already interpolated */
      CHKERRQ(DMPlexCheckPointSFHeavy(*dm, boundary));
    }
    /* Orient interface because it could be deliberately skipped above. It is idempotent. */
    CHKERRQ(DMPlexOrientInterface_Internal(*dm));
  }
  if (user->distribute) {
    DM               pdm = NULL;

    /* Redistribute mesh over processes using that partitioner */
    CHKERRQ(PetscLogStagePush(stage[1]));
    CHKERRQ(DMPlexDistribute(*dm, 0, NULL, &pdm));
    CHKERRQ(PetscLogStagePop());
    if (pdm) {
      CHKERRQ(DMDestroy(dm));
      *dm  = pdm;
      CHKERRQ(PetscObjectSetName((PetscObject) *dm, "Redistributed Mesh"));
      CHKERRQ(DMViewFromOptions(*dm, NULL, "-dist_dm_view"));
    }

    if (interpParallel) {
      DM idm;

      if (testHeavy) CHKERRQ(DMPlexSetOrientInterface_Private(*dm, PETSC_FALSE));
      CHKERRQ(PetscLogStagePush(stage[2]));
      CHKERRQ(DMPlexInterpolate(*dm, &idm)); /* with DMPlexOrientInterface_Internal() call skipped so that PointSF issues are left to DMPlexCheckPointSFHeavy() */
      CHKERRQ(PetscLogStagePop());
      if (testHeavy) CHKERRQ(DMPlexSetOrientInterface_Private(*dm, PETSC_TRUE));
      CHKERRQ(DMDestroy(dm));
      *dm = idm;
      CHKERRQ(PetscObjectSetName((PetscObject) *dm, "Interpolated Redistributed Mesh"));
      CHKERRQ(DMViewFromOptions(*dm, NULL, "-intp_dm_view"));
    }
  }
  if (testHeavy) {
    if (boundary) {
      CHKERRQ(DMPlexCheckPointSFHeavy(*dm, boundary));
    }
    /* Orient interface because it could be deliberately skipped above. It is idempotent. */
    CHKERRQ(DMPlexOrientInterface_Internal(*dm));
  }

  CHKERRQ(PetscObjectSetName((PetscObject) *dm, "Parallel Mesh"));
  CHKERRQ(DMPlexDistributeSetDefault(*dm, PETSC_FALSE));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));

  if (user->customView) CHKERRQ(CustomView(*dm, PETSC_VIEWER_STDOUT_(comm)));
  CHKERRQ(DMDestroy(&serialDM));
  CHKERRQ(PortableBoundaryDestroy(&boundary));
  PetscFunctionReturn(0);
}

#define ps2d(number) ((double) PetscRealPart(number))
static inline PetscErrorCode coord2str(char buf[], size_t len, PetscInt dim, const PetscScalar coords[], PetscReal tol)
{
  PetscFunctionBegin;
  PetscCheckFalse(dim > 3,PETSC_COMM_SELF, PETSC_ERR_SUP, "dim must be less than or equal 3");
  if (tol >= 1e-3) {
    switch (dim) {
      case 1: CHKERRQ(PetscSNPrintf(buf,len,"(%12.3f)",ps2d(coords[0])));
      case 2: CHKERRQ(PetscSNPrintf(buf,len,"(%12.3f, %12.3f)",ps2d(coords[0]),ps2d(coords[1])));
      case 3: CHKERRQ(PetscSNPrintf(buf,len,"(%12.3f, %12.3f, %12.3f)",ps2d(coords[0]),ps2d(coords[1]),ps2d(coords[2])));
    }
  } else {
    switch (dim) {
      case 1: CHKERRQ(PetscSNPrintf(buf,len,"(%12.6f)",ps2d(coords[0])));
      case 2: CHKERRQ(PetscSNPrintf(buf,len,"(%12.6f, %12.6f)",ps2d(coords[0]),ps2d(coords[1])));
      case 3: CHKERRQ(PetscSNPrintf(buf,len,"(%12.6f, %12.6f, %12.6f)",ps2d(coords[0]),ps2d(coords[1]),ps2d(coords[2])));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ViewVerticesFromCoords(DM dm, Vec coordsVec, PetscReal tol, PetscViewer viewer)
{
  PetscInt       dim, i, npoints;
  IS             pointsIS;
  const PetscInt *points;
  const PetscScalar *coords;
  char           coordstr[128];
  MPI_Comm       comm;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)dm, &comm));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(PetscViewerASCIIPushSynchronized(viewer));
  CHKERRQ(DMPlexFindVertices(dm, coordsVec, tol, &pointsIS));
  CHKERRQ(ISGetIndices(pointsIS, &points));
  CHKERRQ(ISGetLocalSize(pointsIS, &npoints));
  CHKERRQ(VecGetArrayRead(coordsVec, &coords));
  for (i=0; i < npoints; i++) {
    CHKERRQ(coord2str(coordstr, sizeof(coordstr), dim, &coords[i*dim], tol));
    if (rank == 0 && i) CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, "-----\n"));
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] %s --> points[%D] = %D\n", rank, coordstr, i, points[i]));
    CHKERRQ(PetscViewerFlush(viewer));
  }
  CHKERRQ(PetscViewerASCIIPopSynchronized(viewer));
  CHKERRQ(VecRestoreArrayRead(coordsVec, &coords));
  CHKERRQ(ISRestoreIndices(pointsIS, &points));
  CHKERRQ(ISDestroy(&pointsIS));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestExpandPoints(DM dm, AppCtx *user)
{
  IS                is;
  PetscSection      *sects;
  IS                *iss;
  PetscInt          d,depth;
  PetscMPIInt       rank;
  PetscViewer       viewer=PETSC_VIEWER_STDOUT_WORLD, sviewer;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank));
  if (user->testExpandPointsEmpty && rank == 0) {
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, 0, NULL, PETSC_USE_POINTER, &is));
  } else {
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, user->nPointsToExpand, user->pointsToExpand, PETSC_USE_POINTER, &is));
  }
  CHKERRQ(DMPlexGetConeRecursive(dm, is, &depth, &iss, &sects));
  CHKERRQ(PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
  CHKERRQ(PetscViewerASCIIPrintf(sviewer, "[%d] ==========================\n",rank));
  for (d=depth-1; d>=0; d--) {
    IS          checkIS;
    PetscBool   flg;

    CHKERRQ(PetscViewerASCIIPrintf(sviewer, "depth %D ---------------\n",d));
    CHKERRQ(PetscSectionView(sects[d], sviewer));
    CHKERRQ(ISView(iss[d], sviewer));
    /* check reverse operation */
    if (d < depth-1) {
      CHKERRQ(DMPlexExpandedConesToFaces_Private(dm, iss[d], sects[d], &checkIS));
      CHKERRQ(ISEqualUnsorted(checkIS, iss[d+1], &flg));
      PetscCheck(flg,PetscObjectComm((PetscObject) checkIS), PETSC_ERR_PLIB, "DMPlexExpandedConesToFaces_Private produced wrong IS");
      CHKERRQ(ISDestroy(&checkIS));
    }
  }
  CHKERRQ(PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
  CHKERRQ(PetscViewerFlush(viewer));
  CHKERRQ(DMPlexRestoreConeRecursive(dm, is, &depth, &iss, &sects));
  CHKERRQ(ISDestroy(&is));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexExpandedConesToFaces_Private(DM dm, IS is, PetscSection section, IS *newis)
{
  PetscInt          n,n1,ncone,numCoveredPoints,o,p,q,start,end;
  const PetscInt    *coveredPoints;
  const PetscInt    *arr, *cone;
  PetscInt          *newarr;

  PetscFunctionBegin;
  CHKERRQ(ISGetLocalSize(is, &n));
  CHKERRQ(PetscSectionGetStorageSize(section, &n1));
  CHKERRQ(PetscSectionGetChart(section, &start, &end));
  PetscCheckFalse(n != n1,PETSC_COMM_SELF, PETSC_ERR_PLIB, "IS size = %D != %D = section storage size", n, n1);
  CHKERRQ(ISGetIndices(is, &arr));
  CHKERRQ(PetscMalloc1(end-start, &newarr));
  for (q=start; q<end; q++) {
    CHKERRQ(PetscSectionGetDof(section, q, &ncone));
    CHKERRQ(PetscSectionGetOffset(section, q, &o));
    cone = &arr[o];
    if (ncone == 1) {
      numCoveredPoints = 1;
      p = cone[0];
    } else {
      PetscInt i;
      p = PETSC_MAX_INT;
      for (i=0; i<ncone; i++) if (cone[i] < 0) {p = -1; break;}
      if (p >= 0) {
        CHKERRQ(DMPlexGetJoin(dm, ncone, cone, &numCoveredPoints, &coveredPoints));
        PetscCheckFalse(numCoveredPoints > 1,PETSC_COMM_SELF, PETSC_ERR_PLIB, "more than one covered points for section point %D",q);
        if (numCoveredPoints) p = coveredPoints[0];
        else                  p = -2;
        CHKERRQ(DMPlexRestoreJoin(dm, ncone, cone, &numCoveredPoints, &coveredPoints));
      }
    }
    newarr[q-start] = p;
  }
  CHKERRQ(ISRestoreIndices(is, &arr));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, end-start, newarr, PETSC_OWN_POINTER, newis));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexExpandedVerticesToFaces_Private(DM dm, IS boundary_expanded_is, PetscInt depth, PetscSection sections[], IS *boundary_is)
{
  PetscInt          d;
  IS                is,newis;

  PetscFunctionBegin;
  is = boundary_expanded_is;
  CHKERRQ(PetscObjectReference((PetscObject)is));
  for (d = 0; d < depth-1; ++d) {
    CHKERRQ(DMPlexExpandedConesToFaces_Private(dm, is, sections[d], &newis));
    CHKERRQ(ISDestroy(&is));
    is = newis;
  }
  *boundary_is = is;
  PetscFunctionReturn(0);
}

#define CHKERRQI(incall,ierr) if (ierr) {incall = PETSC_FALSE; }

static PetscErrorCode DMLabelViewFromOptionsOnComm_Private(DMLabel label, const char optionname[], MPI_Comm comm)
{
  PetscViewer       viewer;
  PetscBool         flg;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  CHKERRQI(incall,PetscOptionsGetViewer(comm,((PetscObject)label)->options,((PetscObject)label)->prefix,optionname,&viewer,&format,&flg));
  if (flg) {
    CHKERRQI(incall,PetscViewerPushFormat(viewer,format));
    CHKERRQI(incall,DMLabelView(label, viewer));
    CHKERRQI(incall,PetscViewerPopFormat(viewer));
    CHKERRQI(incall,PetscViewerDestroy(&viewer));
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/* TODO: this is hotfixing DMLabelGetStratumIS() - it should be fixed systematically instead */
static inline PetscErrorCode DMLabelGetStratumISOnComm_Private(DMLabel label, PetscInt value, MPI_Comm comm, IS *is)
{
  IS                tmpis;

  PetscFunctionBegin;
  CHKERRQ(DMLabelGetStratumIS(label, value, &tmpis));
  if (!tmpis) CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, 0, NULL, PETSC_USE_POINTER, &tmpis));
  CHKERRQ(ISOnComm(tmpis, comm, PETSC_COPY_VALUES, is));
  CHKERRQ(ISDestroy(&tmpis));
  PetscFunctionReturn(0);
}

/* currently only for simple PetscSection without fields or constraints */
static PetscErrorCode PetscSectionReplicate_Private(MPI_Comm comm, PetscMPIInt rootrank, PetscSection sec0, PetscSection *secout)
{
  PetscSection      sec;
  PetscInt          chart[2], p;
  PetscInt          *dofarr;
  PetscMPIInt       rank;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  if (rank == rootrank) {
    CHKERRQ(PetscSectionGetChart(sec0, &chart[0], &chart[1]));
  }
  CHKERRMPI(MPI_Bcast(chart, 2, MPIU_INT, rootrank, comm));
  CHKERRQ(PetscMalloc1(chart[1]-chart[0], &dofarr));
  if (rank == rootrank) {
    for (p = chart[0]; p < chart[1]; p++) {
      CHKERRQ(PetscSectionGetDof(sec0, p, &dofarr[p-chart[0]]));
    }
  }
  CHKERRMPI(MPI_Bcast(dofarr, chart[1]-chart[0], MPIU_INT, rootrank, comm));
  CHKERRQ(PetscSectionCreate(comm, &sec));
  CHKERRQ(PetscSectionSetChart(sec, chart[0], chart[1]));
  for (p = chart[0]; p < chart[1]; p++) {
    CHKERRQ(PetscSectionSetDof(sec, p, dofarr[p-chart[0]]));
  }
  CHKERRQ(PetscSectionSetUp(sec));
  CHKERRQ(PetscFree(dofarr));
  *secout = sec;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexExpandedVerticesCoordinatesToFaces_Private(DM ipdm, PortableBoundary bnd, IS *face_is)
{
  IS                  faces_expanded_is;

  PetscFunctionBegin;
  CHKERRQ(DMPlexFindVertices(ipdm, bnd->coordinates, 0.0, &faces_expanded_is));
  CHKERRQ(DMPlexExpandedVerticesToFaces_Private(ipdm, faces_expanded_is, bnd->depth, bnd->sections, face_is));
  CHKERRQ(ISDestroy(&faces_expanded_is));
  PetscFunctionReturn(0);
}

/* hack disabling DMPlexOrientInterface() call in DMPlexInterpolate() via -dm_plex_interpolate_orient_interfaces option */
static PetscErrorCode DMPlexSetOrientInterface_Private(DM dm, PetscBool enable)
{
  PetscOptions      options = NULL;
  const char        *prefix = NULL;
  const char        opt[] = "-dm_plex_interpolate_orient_interfaces";
  char              prefix_opt[512];
  PetscBool         flg, set;
  static PetscBool  wasSetTrue = PETSC_FALSE;

  PetscFunctionBegin;
  if (dm) {
    CHKERRQ(PetscObjectGetOptionsPrefix((PetscObject)dm, &prefix));
    options = ((PetscObject)dm)->options;
  }
  CHKERRQ(PetscStrcpy(prefix_opt, "-"));
  CHKERRQ(PetscStrlcat(prefix_opt, prefix, sizeof(prefix_opt)));
  CHKERRQ(PetscStrlcat(prefix_opt, &opt[1], sizeof(prefix_opt)));
  CHKERRQ(PetscOptionsGetBool(options, prefix, opt, &flg, &set));
  if (!enable) {
    if (set && flg) wasSetTrue = PETSC_TRUE;
    CHKERRQ(PetscOptionsSetValue(options, prefix_opt, "0"));
  } else if (set && !flg) {
    if (wasSetTrue) {
      CHKERRQ(PetscOptionsSetValue(options, prefix_opt, "1"));
    } else {
      /* default is PETSC_TRUE */
      CHKERRQ(PetscOptionsClearValue(options, prefix_opt));
    }
    wasSetTrue = PETSC_FALSE;
  }
  if (PetscDefined(USE_DEBUG)) {
    CHKERRQ(PetscOptionsGetBool(options, prefix, opt, &flg, &set));
    PetscCheckFalse(set && flg != enable,PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "PetscOptionsSetValue did not have the desired effect");
  }
  PetscFunctionReturn(0);
}

/* get coordinate description of the whole-domain boundary */
static PetscErrorCode DMPlexGetExpandedBoundary_Private(DM dm, PortableBoundary *boundary)
{
  PortableBoundary       bnd0, bnd;
  MPI_Comm               comm;
  DM                     idm;
  DMLabel                label;
  PetscInt               d;
  const char             boundaryName[] = "DMPlexDistributeInterpolateMarkInterface_boundary";
  IS                     boundary_is;
  IS                     *boundary_expanded_iss;
  PetscMPIInt            rootrank = 0;
  PetscMPIInt            rank, size;
  PetscInt               value = 1;
  DMPlexInterpolatedFlag intp;
  PetscBool              flg;

  PetscFunctionBegin;
  CHKERRQ(PetscNew(&bnd));
  CHKERRQ(PetscObjectGetComm((PetscObject)dm, &comm));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRQ(DMPlexIsDistributed(dm, &flg));
  PetscCheck(!flg,comm, PETSC_ERR_ARG_WRONG, "serial DM (all points on one rank) needed");

  /* interpolate serial DM if not yet interpolated */
  CHKERRQ(DMPlexIsInterpolatedCollective(dm, &intp));
  if (intp == DMPLEX_INTERPOLATED_FULL) {
    idm = dm;
    CHKERRQ(PetscObjectReference((PetscObject)dm));
  } else {
    CHKERRQ(DMPlexInterpolate(dm, &idm));
    CHKERRQ(DMViewFromOptions(idm, NULL, "-idm_view"));
  }

  /* mark whole-domain boundary of the serial DM */
  CHKERRQ(DMLabelCreate(PETSC_COMM_SELF, boundaryName, &label));
  CHKERRQ(DMAddLabel(idm, label));
  CHKERRQ(DMPlexMarkBoundaryFaces(idm, value, label));
  CHKERRQ(DMLabelViewFromOptionsOnComm_Private(label, "-idm_boundary_view", comm));
  CHKERRQ(DMLabelGetStratumIS(label, value, &boundary_is));

  /* translate to coordinates */
  CHKERRQ(PetscNew(&bnd0));
  CHKERRQ(DMGetCoordinatesLocalSetUp(idm));
  if (rank == rootrank) {
    CHKERRQ(DMPlexGetConeRecursive(idm, boundary_is, &bnd0->depth, &boundary_expanded_iss, &bnd0->sections));
    CHKERRQ(DMGetCoordinatesLocalTuple(dm, boundary_expanded_iss[0], NULL, &bnd0->coordinates));
    /* self-check */
    {
      IS is0;
      CHKERRQ(DMPlexExpandedVerticesCoordinatesToFaces_Private(idm, bnd0, &is0));
      CHKERRQ(ISEqual(is0, boundary_is, &flg));
      PetscCheck(flg,PETSC_COMM_SELF, PETSC_ERR_PLIB, "DMPlexExpandedVerticesCoordinatesToFaces_Private produced a wrong IS");
      CHKERRQ(ISDestroy(&is0));
    }
  } else {
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF, 0, &bnd0->coordinates));
  }

  {
    Vec         tmp;
    VecScatter  sc;
    IS          xis;
    PetscInt    n;

    /* just convert seq vectors to mpi vector */
    CHKERRQ(VecGetLocalSize(bnd0->coordinates, &n));
    CHKERRMPI(MPI_Bcast(&n, 1, MPIU_INT, rootrank, comm));
    if (rank == rootrank) {
      CHKERRQ(VecCreateMPI(comm, n, n, &tmp));
    } else {
      CHKERRQ(VecCreateMPI(comm, 0, n, &tmp));
    }
    CHKERRQ(VecCopy(bnd0->coordinates, tmp));
    CHKERRQ(VecDestroy(&bnd0->coordinates));
    bnd0->coordinates = tmp;

    /* replicate coordinates from root rank to all ranks */
    CHKERRQ(VecCreateMPI(comm, n, n*size, &bnd->coordinates));
    CHKERRQ(ISCreateStride(comm, n, 0, 1, &xis));
    CHKERRQ(VecScatterCreate(bnd0->coordinates, xis, bnd->coordinates, NULL, &sc));
    CHKERRQ(VecScatterBegin(sc, bnd0->coordinates, bnd->coordinates, INSERT_VALUES, SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(  sc, bnd0->coordinates, bnd->coordinates, INSERT_VALUES, SCATTER_FORWARD));
    CHKERRQ(VecScatterDestroy(&sc));
    CHKERRQ(ISDestroy(&xis));
  }
  bnd->depth = bnd0->depth;
  CHKERRMPI(MPI_Bcast(&bnd->depth, 1, MPIU_INT, rootrank, comm));
  CHKERRQ(PetscMalloc1(bnd->depth, &bnd->sections));
  for (d=0; d<bnd->depth; d++) {
    CHKERRQ(PetscSectionReplicate_Private(comm, rootrank, (rank == rootrank) ? bnd0->sections[d] : NULL, &bnd->sections[d]));
  }

  if (rank == rootrank) {
    CHKERRQ(DMPlexRestoreConeRecursive(idm, boundary_is, &bnd0->depth, &boundary_expanded_iss, &bnd0->sections));
  }
  CHKERRQ(PortableBoundaryDestroy(&bnd0));
  CHKERRQ(DMRemoveLabelBySelf(idm, &label, PETSC_TRUE));
  CHKERRQ(DMLabelDestroy(&label));
  CHKERRQ(ISDestroy(&boundary_is));
  CHKERRQ(DMDestroy(&idm));
  *boundary = bnd;
  PetscFunctionReturn(0);
}

/* get faces of inter-partition interface */
static PetscErrorCode DMPlexGetInterfaceFaces_Private(DM ipdm, IS boundary_faces_is, IS *interface_faces_is)
{
  MPI_Comm               comm;
  DMLabel                label;
  IS                     part_boundary_faces_is;
  const char             partBoundaryName[] = "DMPlexDistributeInterpolateMarkInterface_partBoundary";
  PetscInt               value = 1;
  DMPlexInterpolatedFlag intp;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)ipdm, &comm));
  CHKERRQ(DMPlexIsInterpolatedCollective(ipdm, &intp));
  PetscCheckFalse(intp != DMPLEX_INTERPOLATED_FULL,comm, PETSC_ERR_ARG_WRONG, "only for fully interpolated DMPlex");

  /* get ipdm partition boundary (partBoundary) */
  CHKERRQ(DMLabelCreate(PETSC_COMM_SELF, partBoundaryName, &label));
  CHKERRQ(DMAddLabel(ipdm, label));
  CHKERRQ(DMPlexMarkBoundaryFaces(ipdm, value, label));
  CHKERRQ(DMLabelViewFromOptionsOnComm_Private(label, "-ipdm_part_boundary_view", comm));
  CHKERRQ(DMLabelGetStratumISOnComm_Private(label, value, comm, &part_boundary_faces_is));
  CHKERRQ(DMRemoveLabelBySelf(ipdm, &label, PETSC_TRUE));
  CHKERRQ(DMLabelDestroy(&label));

  /* remove ipdm whole-domain boundary (boundary_faces_is) from ipdm partition boundary (part_boundary_faces_is), resulting just in inter-partition interface */
  CHKERRQ(ISDifference(part_boundary_faces_is,boundary_faces_is,interface_faces_is));
  CHKERRQ(ISDestroy(&part_boundary_faces_is));
  PetscFunctionReturn(0);
}

/* compute inter-partition interface including edges and vertices */
static PetscErrorCode DMPlexComputeCompleteInterface_Private(DM ipdm, IS interface_faces_is, IS *interface_is)
{
  DMLabel                label;
  PetscInt               value = 1;
  const char             interfaceName[] = "DMPlexDistributeInterpolateMarkInterface_interface";
  DMPlexInterpolatedFlag intp;
  MPI_Comm               comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)ipdm, &comm));
  CHKERRQ(DMPlexIsInterpolatedCollective(ipdm, &intp));
  PetscCheckFalse(intp != DMPLEX_INTERPOLATED_FULL,comm, PETSC_ERR_ARG_WRONG, "only for fully interpolated DMPlex");

  CHKERRQ(DMLabelCreate(PETSC_COMM_SELF, interfaceName, &label));
  CHKERRQ(DMAddLabel(ipdm, label));
  CHKERRQ(DMLabelSetStratumIS(label, value, interface_faces_is));
  CHKERRQ(DMLabelViewFromOptionsOnComm_Private(label, "-interface_faces_view", comm));
  CHKERRQ(DMPlexLabelComplete(ipdm, label));
  CHKERRQ(DMLabelViewFromOptionsOnComm_Private(label, "-interface_view", comm));
  CHKERRQ(DMLabelGetStratumISOnComm_Private(label, value, comm, interface_is));
  CHKERRQ(PetscObjectSetName((PetscObject)*interface_is, "interface_is"));
  CHKERRQ(ISViewFromOptions(*interface_is, NULL, "-interface_is_view"));
  CHKERRQ(DMRemoveLabelBySelf(ipdm, &label, PETSC_TRUE));
  CHKERRQ(DMLabelDestroy(&label));
  PetscFunctionReturn(0);
}

static PetscErrorCode PointSFGetOutwardInterfacePoints(PetscSF sf, IS *is)
{
  PetscInt        n;
  const PetscInt  *arr;

  PetscFunctionBegin;
  CHKERRQ(PetscSFGetGraph(sf, NULL, &n, &arr, NULL));
  CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)sf), n, arr, PETSC_USE_POINTER, is));
  PetscFunctionReturn(0);
}

static PetscErrorCode PointSFGetInwardInterfacePoints(PetscSF sf, IS *is)
{
  PetscInt        n;
  const PetscInt  *rootdegree;
  PetscInt        *arr;

  PetscFunctionBegin;
  CHKERRQ(PetscSFSetUp(sf));
  CHKERRQ(PetscSFComputeDegreeBegin(sf, &rootdegree));
  CHKERRQ(PetscSFComputeDegreeEnd(sf, &rootdegree));
  CHKERRQ(PetscSFComputeMultiRootOriginalNumbering(sf, rootdegree, &n, &arr));
  CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)sf), n, arr, PETSC_OWN_POINTER, is));
  PetscFunctionReturn(0);
}

static PetscErrorCode PointSFGetInterfacePoints_Private(PetscSF pointSF, IS *is)
{
  IS pointSF_out_is, pointSF_in_is;

  PetscFunctionBegin;
  CHKERRQ(PointSFGetOutwardInterfacePoints(pointSF, &pointSF_out_is));
  CHKERRQ(PointSFGetInwardInterfacePoints(pointSF, &pointSF_in_is));
  CHKERRQ(ISExpand(pointSF_out_is, pointSF_in_is, is));
  CHKERRQ(ISDestroy(&pointSF_out_is));
  CHKERRQ(ISDestroy(&pointSF_in_is));
  PetscFunctionReturn(0);
}

#define CHKERRMY(ierr) PetscCheck(!ierr,PETSC_COMM_SELF, PETSC_ERR_PLIB, "PointSF is wrong. Unable to show details!")

static PetscErrorCode ViewPointsWithType_Internal(DM dm, IS pointsIS, PetscViewer v)
{
  DMLabel         label;
  PetscSection    coordsSection;
  Vec             coordsVec;
  PetscScalar     *coordsScalar;
  PetscInt        coneSize, depth, dim, i, p, npoints;
  const PetscInt  *points;

  PetscFunctionBegin;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMGetCoordinateSection(dm, &coordsSection));
  CHKERRQ(DMGetCoordinatesLocal(dm, &coordsVec));
  CHKERRQ(VecGetArray(coordsVec, &coordsScalar));
  CHKERRQ(ISGetLocalSize(pointsIS, &npoints));
  CHKERRQ(ISGetIndices(pointsIS, &points));
  CHKERRQ(DMPlexGetDepthLabel(dm, &label));
  CHKERRQ(PetscViewerASCIIPushTab(v));
  for (i=0; i<npoints; i++) {
    p = points[i];
    CHKERRQ(DMLabelGetValue(label, p, &depth));
    if (!depth) {
      PetscInt        n, o;
      char            coordstr[128];

      CHKERRQ(PetscSectionGetDof(coordsSection, p, &n));
      CHKERRQ(PetscSectionGetOffset(coordsSection, p, &o));
      CHKERRQ(coord2str(coordstr, sizeof(coordstr), n, &coordsScalar[o], 1.0));
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(v, "vertex %D w/ coordinates %s\n", p, coordstr));
    } else {
      char            entityType[16];

      switch (depth) {
        case 1: CHKERRQ(PetscStrcpy(entityType, "edge")); break;
        case 2: CHKERRQ(PetscStrcpy(entityType, "face")); break;
        case 3: CHKERRQ(PetscStrcpy(entityType, "cell")); break;
        default: SETERRQ(PetscObjectComm((PetscObject)v), PETSC_ERR_SUP, "Only for depth <= 3");
      }
      if (depth == dim && dim < 3) {
        CHKERRQ(PetscStrlcat(entityType, " (cell)", sizeof(entityType)));
      }
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(v, "%s %D\n", entityType, p));
    }
    CHKERRQ(DMPlexGetConeSize(dm, p, &coneSize));
    if (coneSize) {
      const PetscInt *cone;
      IS             coneIS;

      CHKERRQ(DMPlexGetCone(dm, p, &cone));
      CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, coneSize, cone, PETSC_USE_POINTER, &coneIS));
      CHKERRQ(ViewPointsWithType_Internal(dm, coneIS, v));
      CHKERRQ(ISDestroy(&coneIS));
    }
  }
  CHKERRQ(PetscViewerASCIIPopTab(v));
  CHKERRQ(VecRestoreArray(coordsVec, &coordsScalar));
  CHKERRQ(ISRestoreIndices(pointsIS, &points));
  PetscFunctionReturn(0);
}

static PetscErrorCode ViewPointsWithType(DM dm, IS points, PetscViewer v)
{
  PetscBool       flg;
  PetscInt        npoints;
  PetscMPIInt     rank;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)v, PETSCVIEWERASCII, &flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)v), PETSC_ERR_SUP, "Only for ASCII viewer");
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)v), &rank));
  CHKERRQ(PetscViewerASCIIPushSynchronized(v));
  CHKERRQ(ISGetLocalSize(points, &npoints));
  if (npoints) {
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(v, "[%d] --------\n", rank));
    CHKERRQ(ViewPointsWithType_Internal(dm, points, v));
  }
  CHKERRQ(PetscViewerFlush(v));
  CHKERRQ(PetscViewerASCIIPopSynchronized(v));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexComparePointSFWithInterface_Private(DM ipdm, IS interface_is)
{
  PetscSF         pointsf;
  IS              pointsf_is;
  PetscBool       flg;
  MPI_Comm        comm;
  PetscMPIInt     size;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)ipdm, &comm));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRQ(DMGetPointSF(ipdm, &pointsf));
  if (pointsf) {
    PetscInt nroots;
    CHKERRQ(PetscSFGetGraph(pointsf, &nroots, NULL, NULL, NULL));
    if (nroots < 0) pointsf = NULL; /* uninitialized SF */
  }
  if (!pointsf) {
    PetscInt N=0;
    if (interface_is) CHKERRQ(ISGetSize(interface_is, &N));
    PetscCheck(!N,comm, PETSC_ERR_PLIB, "interface_is should be NULL or empty for PointSF being NULL");
    PetscFunctionReturn(0);
  }

  /* get PointSF points as IS pointsf_is */
  CHKERRQ(PointSFGetInterfacePoints_Private(pointsf, &pointsf_is));

  /* compare pointsf_is with interface_is */
  CHKERRQ(ISEqual(interface_is, pointsf_is, &flg));
  CHKERRMPI(MPI_Allreduce(MPI_IN_PLACE,&flg,1,MPIU_BOOL,MPI_LAND,comm));
  if (!flg) {
    IS pointsf_extra_is, pointsf_missing_is;
    PetscViewer errv = PETSC_VIEWER_STDERR_(comm);
    CHKERRMY(ISDifference(interface_is, pointsf_is, &pointsf_missing_is));
    CHKERRMY(ISDifference(pointsf_is, interface_is, &pointsf_extra_is));
    CHKERRMY(PetscViewerASCIIPrintf(errv, "Points missing in PointSF:\n"));
    CHKERRMY(ViewPointsWithType(ipdm, pointsf_missing_is, errv));
    CHKERRMY(PetscViewerASCIIPrintf(errv, "Extra points in PointSF:\n"));
    CHKERRMY(ViewPointsWithType(ipdm, pointsf_extra_is, errv));
    CHKERRMY(ISDestroy(&pointsf_extra_is));
    CHKERRMY(ISDestroy(&pointsf_missing_is));
    SETERRQ(comm, PETSC_ERR_PLIB, "PointSF is wrong! See details above.");
  }
  CHKERRQ(ISDestroy(&pointsf_is));
  PetscFunctionReturn(0);
}

/* remove faces & edges from label, leave just vertices */
static PetscErrorCode DMPlexISFilterVertices_Private(DM dm, IS points)
{
  PetscInt        vStart, vEnd;
  MPI_Comm        comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)dm, &comm));
  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  CHKERRQ(ISGeneralFilter(points, vStart, vEnd));
  PetscFunctionReturn(0);
}

/*
  DMPlexCheckPointSFHeavy - Thoroughly test that the PointSF after parallel DMPlexInterpolate() includes exactly all interface points.

  Collective

  Input Parameters:
. dm - The DMPlex object

  Notes:
  The input DMPlex must be serial (one partition has all points, the other partitions have no points).
  This is a heavy test which involves DMPlexInterpolate() if the input DM is not interpolated yet, and depends on having a representation of the whole-domain boundary (PortableBoundary), which can be obtained only by DMPlexGetExpandedBoundary_Private() (which involves DMPlexInterpolate() of a sequential DM).
  This is mainly intended for debugging/testing purposes.

  Algorithm:
  1. boundary faces of the serial version of the whole mesh are found using DMPlexMarkBoundaryFaces()
  2. boundary faces are translated into vertices using DMPlexGetConeRecursive() and these are translated into coordinates - this description (aka PortableBoundary) is completely independent of partitioning and point numbering
  3. the mesh is distributed or loaded in parallel
  4. boundary faces of the distributed mesh are reconstructed from PortableBoundary using DMPlexFindVertices()
  5. partition boundary faces of the parallel mesh are found using DMPlexMarkBoundaryFaces()
  6. partition interfaces are computed as set difference of partition boundary faces minus the reconstructed boundary
  7. check that interface covered by PointSF (union of inward and outward points) is equal to the partition interface for each rank, otherwise print the difference and throw an error

  Level: developer

.seealso: DMGetPointSF(), DMPlexCheckSymmetry(), DMPlexCheckSkeleton(), DMPlexCheckFaces()
*/
static PetscErrorCode DMPlexCheckPointSFHeavy(DM dm, PortableBoundary bnd)
{
  DM                     ipdm=NULL;
  IS                     boundary_faces_is, interface_faces_is, interface_is;
  DMPlexInterpolatedFlag intp;
  MPI_Comm               comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)dm, &comm));

  CHKERRQ(DMPlexIsInterpolatedCollective(dm, &intp));
  if (intp == DMPLEX_INTERPOLATED_FULL) {
    ipdm = dm;
  } else {
    /* create temporary interpolated DM if input DM is not interpolated */
    CHKERRQ(DMPlexSetOrientInterface_Private(dm, PETSC_FALSE));
    CHKERRQ(DMPlexInterpolate(dm, &ipdm)); /* with DMPlexOrientInterface_Internal() call skipped so that PointSF issues are left to DMPlexComparePointSFWithInterface_Private() below */
    CHKERRQ(DMPlexSetOrientInterface_Private(dm, PETSC_TRUE));
  }
  CHKERRQ(DMViewFromOptions(ipdm, NULL, "-ipdm_view"));

  /* recover ipdm whole-domain boundary faces from the expanded vertices coordinates */
  CHKERRQ(DMPlexExpandedVerticesCoordinatesToFaces_Private(ipdm, bnd, &boundary_faces_is));
  /* get inter-partition interface faces (interface_faces_is)*/
  CHKERRQ(DMPlexGetInterfaceFaces_Private(ipdm, boundary_faces_is, &interface_faces_is));
  /* compute inter-partition interface including edges and vertices (interface_is) */
  CHKERRQ(DMPlexComputeCompleteInterface_Private(ipdm, interface_faces_is, &interface_is));
  /* destroy immediate ISs */
  CHKERRQ(ISDestroy(&boundary_faces_is));
  CHKERRQ(ISDestroy(&interface_faces_is));

  /* for uninterpolated case, keep just vertices in interface */
  if (!intp) {
    CHKERRQ(DMPlexISFilterVertices_Private(ipdm, interface_is));
    CHKERRQ(DMDestroy(&ipdm));
  }

  /* compare PointSF with the boundary reconstructed from coordinates */
  CHKERRQ(DMPlexComparePointSFWithInterface_Private(dm, interface_is));
  CHKERRQ(PetscPrintf(comm, "DMPlexCheckPointSFHeavy PASSED\n"));
  CHKERRQ(ISDestroy(&interface_is));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;

  CHKERRQ(PetscInitialize(&argc, &argv, NULL, help));
  CHKERRQ(PetscLogStageRegister("create",&stage[0]));
  CHKERRQ(PetscLogStageRegister("distribute",&stage[1]));
  CHKERRQ(PetscLogStageRegister("interpolate",&stage[2]));
  CHKERRQ(ProcessOptions(PETSC_COMM_WORLD, &user));
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  if (user.nPointsToExpand) {
    CHKERRQ(TestExpandPoints(dm, &user));
  }
  if (user.ncoords) {
    Vec coords;

    CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF, user.ncoords, user.ncoords, user.coords, &coords));
    CHKERRQ(ViewVerticesFromCoords(dm, coords, user.coordsTol, PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(VecDestroy(&coords));
  }
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    nsize: 2
    args: -dm_view ascii::ascii_info_detail
    args: -dm_plex_check_all
    test:
      suffix: 1_tri_dist0
      args: -distribute 0 -interpolate {{none create}separate output}
    test:
      suffix: 1_tri_dist1
      args: -distribute 1 -interpolate {{none create after_distribute}separate output}
    test:
      suffix: 1_quad_dist0
      args: -cell_simplex 0 -distribute 0 -interpolate {{none create}separate output}
    test:
      suffix: 1_quad_dist1
      args: -cell_simplex 0 -distribute 1 -interpolate {{none create after_distribute}separate output}
    test:
      suffix: 1_1d_dist1
      args: -dim 1 -distribute 1

  testset:
    nsize: 3
    args: -testnum 1 -interpolate create
    args: -dm_plex_check_all
    test:
      suffix: 2
      args: -dm_view ascii::ascii_info_detail
    test:
      suffix: 2a
      args: -dm_plex_check_cones_conform_on_interfaces_verbose
    test:
      suffix: 2b
      args: -test_expand_points 0,1,2,5,6
    test:
      suffix: 2c
      args: -test_expand_points 0,1,2,5,6 -test_expand_points_empty

  testset:
    # the same as 1% for 3D
    nsize: 2
    args: -dim 3 -dm_view ascii::ascii_info_detail
    args: -dm_plex_check_all
    test:
      suffix: 4_tet_dist0
      args: -distribute 0 -interpolate {{none create}separate output}
    test:
      suffix: 4_tet_dist1
      args: -distribute 1 -interpolate {{none create after_distribute}separate output}
    test:
      suffix: 4_hex_dist0
      args: -cell_simplex 0 -distribute 0 -interpolate {{none create}separate output}
    test:
      suffix: 4_hex_dist1
      args: -cell_simplex 0 -distribute 1 -interpolate {{none create after_distribute}separate output}

  test:
    # the same as 4_tet_dist0 but test different initial orientations
    suffix: 4_tet_test_orient
    nsize: 2
    args: -dim 3 -distribute 0
    args: -dm_plex_check_all
    args: -rotate_interface_0 {{0 1 2 11 12 13}}
    args: -rotate_interface_1 {{0 1 2 11 12 13}}

  testset:
    requires: exodusii
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/TwoQuads.exo
    args: -dm_view ascii::ascii_info_detail
    args: -dm_plex_check_all
    args: -custom_view
    test:
      suffix: 5_seq
      nsize: 1
      args: -distribute 0 -interpolate {{none create}separate output}
    test:
      # Detail viewing in a non-distributed mesh is broken because the DMLabelView() is collective, but the label is not shared
      suffix: 5_dist0
      nsize: 2
      args: -distribute 0 -interpolate {{none create}separate output} -dm_view
    test:
      suffix: 5_dist1
      nsize: 2
      args: -distribute 1 -interpolate {{none create after_distribute}separate output}

  testset:
    nsize: {{1 2 4}}
    args: -use_generator
    args: -dm_plex_check_all
    args: -distribute -interpolate none
    test:
      suffix: 6_tri
      requires: triangle
      args: -faces {{2,2  1,3  7,4}} -cell_simplex 1 -dm_generator triangle
    test:
      suffix: 6_quad
      args: -faces {{2,2  1,3  7,4}} -cell_simplex 0
    test:
      suffix: 6_tet
      requires: ctetgen
      args: -faces {{2,2,2  1,3,5  3,4,7}} -cell_simplex 1 -dm_generator ctetgen
    test:
      suffix: 6_hex
      args: -faces {{2,2,2  1,3,5  3,4,7}} -cell_simplex 0
  testset:
    nsize: {{1 2 4}}
    args: -use_generator
    args: -dm_plex_check_all
    args: -distribute -interpolate create
    test:
      suffix: 6_int_tri
      requires: triangle
      args: -faces {{2,2  1,3  7,4}} -cell_simplex 1 -dm_generator triangle
    test:
      suffix: 6_int_quad
      args: -faces {{2,2  1,3  7,4}} -cell_simplex 0
    test:
      suffix: 6_int_tet
      requires: ctetgen
      args: -faces {{2,2,2  1,3,5  3,4,7}} -cell_simplex 1 -dm_generator ctetgen
    test:
      suffix: 6_int_hex
      args: -faces {{2,2,2  1,3,5  3,4,7}} -cell_simplex 0
  testset:
    nsize: {{2 4}}
    args: -use_generator
    args: -dm_plex_check_all
    args: -distribute -interpolate after_distribute
    test:
      suffix: 6_parint_tri
      requires: triangle
      args: -faces {{2,2  1,3  7,4}} -cell_simplex 1 -dm_generator triangle
    test:
      suffix: 6_parint_quad
      args: -faces {{2,2  1,3  7,4}} -cell_simplex 0
    test:
      suffix: 6_parint_tet
      requires: ctetgen
      args: -faces {{2,2,2  1,3,5  3,4,7}} -cell_simplex 1 -dm_generator ctetgen
    test:
      suffix: 6_parint_hex
      args: -faces {{2,2,2  1,3,5  3,4,7}} -cell_simplex 0

  testset: # 7 EXODUS
    requires: exodusii
    args: -dm_plex_check_all
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo
    args: -distribute
    test: # seq load, simple partitioner
      suffix: 7_exo
      nsize: {{1 2 4 5}}
      args: -interpolate none
    test: # seq load, seq interpolation, simple partitioner
      suffix: 7_exo_int_simple
      nsize: {{1 2 4 5}}
      args: -interpolate create
    test: # seq load, seq interpolation, metis partitioner
      suffix: 7_exo_int_metis
      requires: parmetis
      nsize: {{2 4 5}}
      args: -interpolate create
      args: -petscpartitioner_type parmetis
    test: # seq load, simple partitioner, par interpolation
      suffix: 7_exo_simple_int
      nsize: {{2 4 5}}
      args: -interpolate after_distribute
    test: # seq load, metis partitioner, par interpolation
      suffix: 7_exo_metis_int
      requires: parmetis
      nsize: {{2 4 5}}
      args: -interpolate after_distribute
      args: -petscpartitioner_type parmetis

  testset: # 7 HDF5 SEQUANTIAL LOAD
    requires: hdf5 !complex
    args: -dm_plex_check_all
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.h5 -dm_plex_create_from_hdf5_xdmf
    args: -dm_plex_hdf5_force_sequential
    args: -distribute
    test: # seq load, simple partitioner
      suffix: 7_seq_hdf5_simple
      nsize: {{1 2 4 5}}
      args: -interpolate none
    test: # seq load, seq interpolation, simple partitioner
      suffix: 7_seq_hdf5_int_simple
      nsize: {{1 2 4 5}}
      args: -interpolate after_create
    test: # seq load, seq interpolation, metis partitioner
      nsize: {{2 4 5}}
      suffix: 7_seq_hdf5_int_metis
      requires: parmetis
      args: -interpolate after_create
      args: -petscpartitioner_type parmetis
    test: # seq load, simple partitioner, par interpolation
      suffix: 7_seq_hdf5_simple_int
      nsize: {{2 4 5}}
      args: -interpolate after_distribute
    test: # seq load, metis partitioner, par interpolation
      nsize: {{2 4 5}}
      suffix: 7_seq_hdf5_metis_int
      requires: parmetis
      args: -interpolate after_distribute
      args: -petscpartitioner_type parmetis

  testset: # 7 HDF5 PARALLEL LOAD
    requires: hdf5 !complex
    nsize: {{2 4 5}}
    args: -dm_plex_check_all
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.h5 -dm_plex_create_from_hdf5_xdmf
    test: # par load
      suffix: 7_par_hdf5
      args: -interpolate none
    test: # par load, par interpolation
      suffix: 7_par_hdf5_int
      args: -interpolate after_create
    test: # par load, parmetis repartitioner
      TODO: Parallel partitioning of uninterpolated meshes not supported
      suffix: 7_par_hdf5_parmetis
      requires: parmetis
      args: -distribute -petscpartitioner_type parmetis
      args: -interpolate none
    test: # par load, par interpolation, parmetis repartitioner
      suffix: 7_par_hdf5_int_parmetis
      requires: parmetis
      args: -distribute -petscpartitioner_type parmetis
      args: -interpolate after_create
    test: # par load, parmetis partitioner, par interpolation
      TODO: Parallel partitioning of uninterpolated meshes not supported
      suffix: 7_par_hdf5_parmetis_int
      requires: parmetis
      args: -distribute -petscpartitioner_type parmetis
      args: -interpolate after_distribute

    test:
      suffix: 7_hdf5_hierarch
      requires: hdf5 ptscotch !complex
      nsize: {{2 3 4}separate output}
      args: -distribute
      args: -interpolate after_create
      args: -petscpartitioner_type matpartitioning -petscpartitioner_view ::ascii_info
      args: -mat_partitioning_type hierarch -mat_partitioning_hierarchical_nfineparts 2
      args: -mat_partitioning_hierarchical_coarseparttype ptscotch -mat_partitioning_hierarchical_fineparttype ptscotch

  test:
    suffix: 8
    requires: hdf5 !complex
    nsize: 4
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.h5 -dm_plex_create_from_hdf5_xdmf
    args: -distribute 0 -interpolate after_create
    args: -view_vertices_from_coords 0.,1.,0.,-0.5,1.,0.,0.583,-0.644,0.,-2.,-2.,-2. -view_vertices_from_coords_tol 1e-3
    args: -dm_plex_check_all
    args: -custom_view

  testset: # 9 HDF5 SEQUANTIAL LOAD
    requires: hdf5 !complex datafilespath
    args: -dm_plex_check_all
    args: -filename ${DATAFILESPATH}/meshes/cube-hexahedra-refined.h5 -dm_plex_create_from_hdf5_xdmf -dm_plex_hdf5_topology_path /cells -dm_plex_hdf5_geometry_path /coordinates
    args: -dm_plex_hdf5_force_sequential
    args: -distribute
    test: # seq load, simple partitioner
      suffix: 9_seq_hdf5_simple
      nsize: {{1 2 4 5}}
      args: -interpolate none
    test: # seq load, seq interpolation, simple partitioner
      suffix: 9_seq_hdf5_int_simple
      nsize: {{1 2 4 5}}
      args: -interpolate after_create
    test: # seq load, seq interpolation, metis partitioner
      nsize: {{2 4 5}}
      suffix: 9_seq_hdf5_int_metis
      requires: parmetis
      args: -interpolate after_create
      args: -petscpartitioner_type parmetis
    test: # seq load, simple partitioner, par interpolation
      suffix: 9_seq_hdf5_simple_int
      nsize: {{2 4 5}}
      args: -interpolate after_distribute
    test: # seq load, simple partitioner, par interpolation
      # This is like 9_seq_hdf5_simple_int but testing error output of DMPlexCheckPointSFHeavy().
      # Once 9_seq_hdf5_simple_int gets fixed, this one gets broken.
      # We can then provide an intentionally broken mesh instead.
      TODO: This test is broken because PointSF is fixed.
      suffix: 9_seq_hdf5_simple_int_err
      nsize: 4
      args: -interpolate after_distribute
      filter: sed -e "/PETSC ERROR/,$$d"
    test: # seq load, metis partitioner, par interpolation
      nsize: {{2 4 5}}
      suffix: 9_seq_hdf5_metis_int
      requires: parmetis
      args: -interpolate after_distribute
      args: -petscpartitioner_type parmetis

  testset: # 9 HDF5 PARALLEL LOAD
    requires: hdf5 !complex datafilespath
    nsize: {{2 4 5}}
    args: -dm_plex_check_all
    args: -filename ${DATAFILESPATH}/meshes/cube-hexahedra-refined.h5 -dm_plex_create_from_hdf5_xdmf -dm_plex_hdf5_topology_path /cells -dm_plex_hdf5_geometry_path /coordinates
    test: # par load
      suffix: 9_par_hdf5
      args: -interpolate none
    test: # par load, par interpolation
      suffix: 9_par_hdf5_int
      args: -interpolate after_create
    test: # par load, parmetis repartitioner
      TODO: Parallel partitioning of uninterpolated meshes not supported
      suffix: 9_par_hdf5_parmetis
      requires: parmetis
      args: -distribute -petscpartitioner_type parmetis
      args: -interpolate none
    test: # par load, par interpolation, parmetis repartitioner
      suffix: 9_par_hdf5_int_parmetis
      requires: parmetis
      args: -distribute -petscpartitioner_type parmetis
      args: -interpolate after_create
    test: # par load, parmetis partitioner, par interpolation
      TODO: Parallel partitioning of uninterpolated meshes not supported
      suffix: 9_par_hdf5_parmetis_int
      requires: parmetis
      args: -distribute -petscpartitioner_type parmetis
      args: -interpolate after_distribute

  testset: # 10 HDF5 PARALLEL LOAD
    requires: hdf5 !complex datafilespath
    nsize: {{2 4 7}}
    args: -dm_plex_check_all
    args: -filename ${DATAFILESPATH}/meshes/cube-hexahedra-refined2.h5 -dm_plex_create_from_hdf5_xdmf -dm_plex_hdf5_topology_path /topo -dm_plex_hdf5_geometry_path /geom
    test: # par load, par interpolation
      suffix: 10_par_hdf5_int
      args: -interpolate after_create
TEST*/
